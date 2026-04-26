# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Smart Calendar Agent Environment — Round 2.

CalendarEnv is a thin orchestrator that delegates to:
  CalendarBuilder     — weekly calendar construction and obstacle seeding
  ObservationBuilder  — agent-visible metadata (free slots, dep graph)
  RewardCalculator    — five independent reward signals
  ActionValidator     — schema and dependency validation
  CommandHandlerFactory + handlers — Strategy pattern for command execution
"""

import hashlib
import logging
import uuid
from typing import Any, Dict, List, Optional

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import (
        Calendar,
        MyCalendarAction,
        MyCalendarObservation,
        MyCalendarState,
        Slot,
    )
    from ..task_definitions import (
        EASY_SPEC,
        FULL_DEPENDENCY_GRAPH,
        HARD_SPEC,
        MEDIUM_SPEC,
        SUPER_HARD_SPEC,
        THREE_ATTENDEES,
        TaskDifficulty,
        TaskSpec,
    )
    from .calendar_builder import CalendarBuilder
    from .handlers import CommandHandlerFactory
    from .observation_builder import ObservationBuilder
    from .rewards import RewardCalculator
    from .slot_utils import SlotUtils
    from .state import EpisodeState
    from .validators import ActionValidator
except ImportError:
    from models import (
        Calendar,
        MyCalendarAction,
        MyCalendarObservation,
        MyCalendarState,
        Slot,
    )
    from task_definitions import (
        EASY_SPEC,
        FULL_DEPENDENCY_GRAPH,
        HARD_SPEC,
        MEDIUM_SPEC,
        SUPER_HARD_SPEC,
        THREE_ATTENDEES,
        TaskDifficulty,
        TaskSpec,
    )
    from server.calendar_builder import CalendarBuilder
    from server.handlers import CommandHandlerFactory
    from server.observation_builder import ObservationBuilder
    from server.rewards import RewardCalculator
    from server.slot_utils import SlotUtils
    from server.state import EpisodeState
    from server.validators import ActionValidator

log = logging.getLogger(__name__)

_SPEC_MAP: Dict[TaskDifficulty, TaskSpec] = {
    TaskDifficulty.EASY: EASY_SPEC,
    TaskDifficulty.MEDIUM: MEDIUM_SPEC,
    TaskDifficulty.HARD: HARD_SPEC,
    TaskDifficulty.SUPER_HARD: SUPER_HARD_SPEC,
}


class CalendarEnv(Environment):
    """Round-2 Smart Calendar RL environment.

    Orchestrates a five-day weekly scheduling task with multi-attendee
    personas, a meeting dependency graph, and five independent reward signals.

    The raw UTC slot matrix is never exposed to the agent — only
    local-time free-slot strings and dependency status labels are surfaced.
    """

    def __init__(self) -> None:
        """Initialise instance variables; a full episode begins with reset()."""
        self._state: Optional[EpisodeState] = None
        self._action_history: List[str] = []
        self._episode_id: str = str(uuid.uuid4())

    # ── public interface ──────────────────────────────────────────────────

    def reset(self, task: TaskDifficulty = TaskDifficulty.EASY) -> MyCalendarObservation:
        """Reset to a fresh episode for the given difficulty.

        Args:
            task: Curriculum difficulty level (EASY / MEDIUM / HARD).
                  Accepts TaskDifficulty enum or its string value (e.g. "medium").

        Returns:
            Initial observation with task description and full visible state.
        """
        if isinstance(task, str):
            task = TaskDifficulty(task)
        spec = _SPEC_MAP[task]
        week_dates = CalendarBuilder.get_week_dates()
        week = CalendarBuilder.build_weekly(week_dates)
        CalendarBuilder.preseed_obstacles(week, task)

        attendee_map = {a.name: a for a in THREE_ATTENDEES}
        attendees = [attendee_map[n] for n in spec.attendee_names if n in attendee_map]

        self._state = EpisodeState(spec=spec, week=week, attendees=attendees)
        self._action_history = []
        self._episode_id = str(uuid.uuid4())

        objective = (
            f"Schedule {len(spec.meetings)} meeting(s) in dependency order: "
            + " → ".join(spec.meetings)
            + ". Respect all dependencies and attendee preferred hours."
        )

        return MyCalendarObservation(
            message=objective,
            reward=0.0,
            done=False,
            metadata={
                "free_slots": ObservationBuilder.free_slots(week, spec, attendees),
                "attendees": [
                    {
                        "name": a.name,
                        "timezone": a.timezone,
                        "preferred_hours": (
                            f"{a.preferred_start_hour}am-{a.preferred_end_hour}pm local"
                        ),
                    }
                    for a in attendees
                ],
                "scheduled_meetings": [],
                "dependency_graph": ObservationBuilder.dependency_graph(spec, []),
                "task_objective": objective,
                "target_meetings": len(spec.meetings),
                "max_steps": spec.max_steps,
                "steps_remaining": spec.max_steps,
                "episode_id": self._episode_id,
            },
        )

    def step(self, action: MyCalendarAction) -> MyCalendarObservation:
        """Apply one agent action and return the updated observation.

        Processing order:
          1. Hard step-limit guard
          2. Schema validation (ActionValidator)
          3. Duplicate detection (SHA-256 fingerprint)
          4. Command dispatch (CommandHandlerFactory → handler.execute)
          5. Update higher-level state (scheduled_meeting_ids, details)
          6. Compute five reward signals (RewardCalculator)
          7. Build and return observation

        All five reward signals live in metadata only; the top-level
        reward field is always 0.0 (signals are never composited).

        Args:
            action: Agent's calendar action.

        Returns:
            Observation with five reward signals and updated visible state.
        """
        assert self._state is not None, "call reset() before step()"
        state = self._state
        state.step_count += 1

        # 1. Hard step-limit
        if state.step_count > state.spec.max_steps:
            return self._build_obs(
                message="Episode terminated: max steps exceeded.",
                action_valid=False,
                rejection_reason="max_steps_exceeded",
                rewards=RewardCalculator.zero(),
                state=state,
                done=True,
            )

        # 2. Schema validation
        valid, reason = ActionValidator.schema(action)
        if not valid:
            log.warning("schema validation failed: %s", reason)
            return self._build_obs(
                message=f"Invalid action: {reason}",
                action_valid=False,
                rejection_reason=reason,
                rewards=RewardCalculator.zero(),
                state=state,
                done=False,
            )

        # 3. Duplicate detection
        fingerprint = self._fingerprint(action)
        if fingerprint in self._action_history:
            log.warning("duplicate action: %s", action.expected_action.command)
            return self._build_obs(
                message="Duplicate action — no reward.",
                action_valid=False,
                rejection_reason="duplicate_action",
                rewards=RewardCalculator.zero(),
                state=state,
                done=False,
            )
        self._action_history.append(fingerprint)

        # 4. Dispatch to strategy handler
        handler = CommandHandlerFactory.get(action.expected_action.command)
        result = handler.execute(action, state)

        if not result.action_valid:
            rewards = RewardCalculator.zero()
            # Surface the specific violated signal for informative training signal
            if result.rejection_reason == "slot_conflict":
                rewards["reward_conflict_free"] = 0.0
            if result.rejection_reason and "dependency" in result.rejection_reason:
                rewards["reward_dependency_respected"] = 0.0
            return self._build_obs(
                message=result.message,
                action_valid=False,
                rejection_reason=result.rejection_reason,
                rewards=rewards,
                state=state,
                done=False,
            )

        # 5. Update higher-level state (handlers only mutate slots)
        command = action.expected_action.command
        is_search = command == "search_slot"
        is_delete = command == "delete_event"
        if not is_search and result.meeting_id:
            if is_delete:
                # delete_event: unregister the meeting
                state.scheduled_meeting_ids = [
                    m for m in state.scheduled_meeting_ids if m != result.meeting_id
                ]
                state.scheduled_details = [
                    d for d in state.scheduled_details if d["meeting_id"] != result.meeting_id
                ]
            elif result.meeting_id not in state.scheduled_meeting_ids:
                # add_event: register the new booking
                state.scheduled_meeting_ids.append(result.meeting_id)
                state.scheduled_details.append(
                    {
                        "meeting_id": result.meeting_id,
                        "day": result.day,
                        "utc_start": f"{result.utc_hour:02d}:00",
                        "utc_end": f"{(result.utc_hour or 0) + 1:02d}:00",
                    }
                )
            else:
                # move_event: update existing detail in place
                for detail in state.scheduled_details:
                    if detail["meeting_id"] == result.meeting_id:
                        detail["day"] = result.day
                        detail["utc_start"] = f"{result.utc_hour:02d}:00"
                        detail["utc_end"] = f"{(result.utc_hour or 0) + 1:02d}:00"
                        break

        # 6. Compute rewards
        if is_search:
            rewards = {
                "reward_conflict_free": RewardCalculator.conflict_free(state.week),
                "reward_dependency_respected": 0.3,
                "reward_attendee_satisfied": 0.0,
                "reward_efficiency": RewardCalculator.efficiency(
                    state.step_count, state.spec.max_steps
                ),
                "reward_objective_progress": RewardCalculator.objective_progress(state),
            }
        elif is_delete:
            # Deletion is corrective: no meeting scheduled, no dependency satisfied
            rewards = {
                "reward_conflict_free": RewardCalculator.conflict_free(state.week),
                "reward_dependency_respected": 0.0,
                "reward_attendee_satisfied": 0.0,
                "reward_efficiency": RewardCalculator.efficiency(
                    state.step_count, state.spec.max_steps
                ),
                "reward_objective_progress": RewardCalculator.objective_progress(state),
            }
        else:
            rewards = RewardCalculator.all_rewards(
                meeting_id=result.meeting_id or "",
                utc_hour=result.utc_hour or 0,
                state=state,
            )

        done = (
            len(state.scheduled_meeting_ids) >= len(state.spec.meetings)
            or state.step_count >= state.spec.max_steps
        )

        # 7. Build observation
        return self._build_obs(
            message=result.message,
            action_valid=True,
            rejection_reason=None,
            rewards=rewards,
            state=state,
            done=done,
        )

    # ── private helpers ───────────────────────────────────────────────────

    def _build_obs(
        self,
        message: str,
        action_valid: bool,
        rejection_reason: Optional[str],
        rewards: Dict[str, Any],
        state: EpisodeState,
        done: bool,
    ) -> MyCalendarObservation:
        """Assemble the full observation returned to the agent.

        Guarantees hard termination: overrides done=True whenever
        step_count >= max_steps regardless of the calling path.

        Args:
            message: Human-readable outcome.
            action_valid: Whether the action was accepted.
            rejection_reason: Reason string if rejected, else None.
            rewards: Five-signal reward dict.
            state: Current episode state.
            done: Episode-done flag from the caller.

        Returns:
            MyCalendarObservation with metadata containing all signals.
        """
        effective_done = done or state.step_count >= state.spec.max_steps
        return MyCalendarObservation(
            message=message,
            reward=0.0,
            done=effective_done,
            metadata={
                **rewards,
                "action_valid": action_valid,
                "rejection_reason": rejection_reason,
                "free_slots": ObservationBuilder.free_slots(
                    state.week, state.spec, state.attendees
                ),
                "scheduled_meetings": list(state.scheduled_details),
                "dependency_graph": ObservationBuilder.dependency_graph(
                    state.spec, state.scheduled_meeting_ids
                ),
                "steps_remaining": max(0, state.spec.max_steps - state.step_count),
                "target_meetings": len(state.spec.meetings),
            },
        )

    @staticmethod
    def _fingerprint(action: MyCalendarAction) -> str:
        """SHA-256 fingerprint for duplicate-action detection.

        Args:
            action: Agent action to fingerprint.

        Returns:
            Hex digest string.
        """
        return hashlib.sha256(action.model_dump_json().encode()).hexdigest()

    # ── state property ────────────────────────────────────────────────────

    @property
    def state(self) -> MyCalendarState:
        """Current environment state as a MyCalendarState snapshot.

        Returns:
            MyCalendarState with all fields populated.

        Raises:
            AssertionError: If reset() has not been called.
        """
        assert self._state is not None, "call reset() before accessing state"
        s = self._state

        all_slots: List[Slot] = [
            slot
            for day_name in s.spec.days
            for slot in (
                s.week.days[day_name].slots if day_name in s.week.days else []
            )
        ]
        flat_free: List[str] = [
            (
                f"{day_name} "
                f"{SlotUtils.parse_utc(slot.start_time).strftime('%H:%M')}-"
                f"{SlotUtils.parse_utc(slot.end_time).strftime('%H:%M')} UTC"
            )
            for day_name in s.spec.days
            for slot in (
                s.week.days[day_name].slots if day_name in s.week.days else []
            )
            if slot.event is None
        ]

        return MyCalendarState(
            episode_id=self._episode_id,
            step_count=s.step_count,
            calendar=Calendar(slots=all_slots),
            task_objective=(
                f"Schedule {len(s.spec.meetings)} meeting(s): "
                + " → ".join(s.spec.meetings)
            ),
            task_goal=f"schedule {len(s.spec.meetings)} meetings",
            events=list(s.scheduled_meeting_ids),
            free_slots=flat_free,
            target_meetings=len(s.spec.meetings),
            scheduled_meetings=len(s.scheduled_meeting_ids),
            objective_progress=RewardCalculator.objective_progress(s),
            failed_steps=0,
            week=s.week,
            attendees=list(s.attendees),
            cascade_conflicts=[],
            attendee_satisfaction=0.0,
            dependency_graph={
                mid: FULL_DEPENDENCY_GRAPH.get(mid, {}).get("deps", [])
                for mid in s.spec.meetings
            },
            scheduled_meeting_ids=list(s.scheduled_meeting_ids),
        )
