# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Five independent reward signals for the Round-2 calendar environment.

Per CLAUDE.md: reward signals are NEVER combined into a composite score.
Each method is a separate, independently interpretable signal.
"""

from typing import Any, Dict

try:
    from ..models import WeeklyCalendar, is_within_preference
    from ..task_definitions import FULL_DEPENDENCY_GRAPH
    from .state import EpisodeState
except ImportError:
    from models import WeeklyCalendar, is_within_preference
    from task_definitions import FULL_DEPENDENCY_GRAPH
    from server.state import EpisodeState


class RewardCalculator:
    """Five independent reward signals — never combined into a composite score."""

    @staticmethod
    def conflict_free(week: WeeklyCalendar) -> float:
        """1.0 if no slot is double-booked in the entire calendar; else 0.0.

        Args:
            week: Weekly calendar to inspect.

        Returns:
            1.0 or 0.0.
        """
        seen: set = set()
        for day_cal in week.days.values():
            for slot in day_cal.slots:
                if slot.event is None:
                    continue
                if slot.start_time in seen:
                    return 0.0
                seen.add(slot.start_time)
        return 1.0

    @staticmethod
    def dependency_respected(
        meeting_id: str, state: EpisodeState, is_search: bool
    ) -> float:
        """Reward for correct dependency ordering.

        Args:
            meeting_id: Meeting being evaluated.
            state: Current episode state.
            is_search: True when the triggering action was search_slot.

        Returns:
            1.0 if all deps satisfied, 0.3 for search, 0.0 if any dep missing.
        """
        if is_search:
            return 0.3
        deps = FULL_DEPENDENCY_GRAPH.get(meeting_id, {}).get("deps", [])
        for dep in deps:
            if dep in state.spec.meetings and dep not in state.scheduled_meeting_ids:
                return 0.0
        return 1.0

    @staticmethod
    def attendee_satisfied(
        meeting_id: str, utc_hour: int, state: EpisodeState
    ) -> float:
        """Fraction of required attendees whose preferred local hours contain utc_hour.

        Args:
            meeting_id: Meeting being scheduled.
            utc_hour: Start hour in UTC.
            state: Current episode state (provides attendee roster).

        Returns:
            Float in [0.0, 1.0].
        """
        required = FULL_DEPENDENCY_GRAPH.get(meeting_id, {}).get("attendees", [])
        if not required:
            return 1.0
        attendee_map = {a.name: a for a in state.attendees}
        satisfied = sum(
            1
            for name in required
            if name in attendee_map
            and is_within_preference(utc_hour, attendee_map[name])
        )
        return satisfied / len(required)

    @staticmethod
    def efficiency(step_count: int, max_steps: int) -> float:
        """Linearly decaying reward: ~1.0 early, 0.0 at max_steps.

        Args:
            step_count: Steps used so far.
            max_steps: Episode step budget.

        Returns:
            Float in [0.0, 1.0].
        """
        return max(0.0, 1.0 - step_count / max_steps)

    @staticmethod
    def objective_progress(state: EpisodeState) -> float:
        """Progressive non-zero signal: scheduled meetings / target meetings.

        Args:
            state: Current episode state.

        Returns:
            Float in [0.0, 1.0].
        """
        target = len(state.spec.meetings)
        return len(state.scheduled_meeting_ids) / target if target > 0 else 0.0

    @classmethod
    def all_rewards(
        cls,
        meeting_id: str,
        utc_hour: int,
        state: EpisodeState,
        is_search: bool = False,
    ) -> Dict[str, Any]:
        """Compute all five signals for a successful scheduling or move action.

        Args:
            meeting_id: Meeting that was just acted on.
            utc_hour: UTC start hour of the acted-on slot.
            state: Updated episode state (after any slot mutation).
            is_search: True when the triggering command was search_slot.

        Returns:
            Dict with five float reward keys.
        """
        return {
            "reward_conflict_free": cls.conflict_free(state.week),
            "reward_dependency_respected": cls.dependency_respected(
                meeting_id, state, is_search
            ),
            "reward_attendee_satisfied": cls.attendee_satisfied(
                meeting_id, utc_hour, state
            ),
            "reward_efficiency": cls.efficiency(state.step_count, state.spec.max_steps),
            "reward_objective_progress": cls.objective_progress(state),
        }

    @staticmethod
    def zero() -> Dict[str, Any]:
        """Return a rewards dict with all five signals set to 0.0.

        Returns:
            Dict with five zero-valued reward keys.
        """
        return {
            "reward_conflict_free": 0.0,
            "reward_dependency_respected": 0.0,
            "reward_attendee_satisfied": 0.0,
            "reward_efficiency": 0.0,
            "reward_objective_progress": 0.0,
        }
