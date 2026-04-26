# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Command handler hierarchy (Strategy Pattern) for calendar actions.

Each concrete handler encapsulates the execution logic for one command type.
Handlers may mutate WeeklyCalendar slots in EpisodeState but must not
touch scheduled_meeting_ids or scheduled_details — those are managed by
CalendarEnv after the handler returns.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict

try:
    from ..models import CalendarEvent, MyCalendarAction
    from ..task_definitions import FULL_DEPENDENCY_GRAPH
    from .slot_utils import SlotUtils
    from .state import CommandResult, EpisodeState
    from .validators import ActionValidator
except ImportError:
    from models import CalendarEvent, MyCalendarAction
    from task_definitions import FULL_DEPENDENCY_GRAPH
    from server.slot_utils import SlotUtils
    from server.state import CommandResult, EpisodeState
    from server.validators import ActionValidator

log = logging.getLogger(__name__)


# ── abstract strategy ────────────────────────────────────────────────────────

class CommandHandler(ABC):
    """Abstract strategy for handling a specific calendar command."""

    @abstractmethod
    def execute(
        self, action: MyCalendarAction, state: EpisodeState
    ) -> CommandResult:
        """Execute the command; may mutate state.week slots only.

        Args:
            action: Agent action carrying command parameters.
            state: Shared episode state.

        Returns:
            CommandResult describing success or rejection details.
        """


# ── concrete strategies ──────────────────────────────────────────────────────

class SearchSlotHandler(CommandHandler):
    """Handles search_slot — informational query, no state mutation."""

    def execute(
        self, action: MyCalendarAction, state: EpisodeState
    ) -> CommandResult:
        """Return free-slot availability for the requested day.

        Day existence is already verified by ActionValidator before dispatch,
        so this handler always returns a valid result.

        Args:
            action: Agent action with command='search_slot'.
            state: Current episode state (not mutated).

        Returns:
            Valid CommandResult with the queried day name.
        """
        day = (action.expected_action.day or "").lower().strip()
        return CommandResult(
            action_valid=True,
            message=f"Free slots on {day} returned.",
            day=day,
        )


class AddEventHandler(CommandHandler):
    """Handles add_event and reschedule_event — books a new meeting slot."""

    def execute(
        self, action: MyCalendarAction, state: EpisodeState
    ) -> CommandResult:
        """Validate dependencies and conflicts, then book the target slot.

        Args:
            action: Agent action with meeting_id, day, and slot.
            state: Current episode state; target slot.event is set on success.

        Returns:
            CommandResult with booking details on success, or rejection reason.
        """
        exp = action.expected_action
        meeting_id = (exp.event_id or "").strip()
        day = (exp.day or "").lower().strip()

        if meeting_id not in state.spec.meetings:
            return CommandResult(
                action_valid=False,
                message=f"Rejected: meeting_id '{meeting_id}' not in task spec",
                rejection_reason=f"unknown_meeting:{meeting_id}",
            )
        if meeting_id in state.scheduled_meeting_ids:
            return CommandResult(
                action_valid=False,
                message=f"Rejected: '{meeting_id}' already scheduled",
                rejection_reason="already_scheduled",
            )
        if day not in state.week.days:
            return CommandResult(
                action_valid=False,
                message=f"Rejected: day '{day}' not in this task",
                rejection_reason=f"unknown_day:{day}",
            )

        dep_ok, dep_reason = ActionValidator.dependencies(meeting_id, state)
        if not dep_ok:
            log.warning("dependency bypass attempt: %s — %s", meeting_id, dep_reason)
            return CommandResult(
                action_valid=False,
                message=f"Rejected: {dep_reason}",
                rejection_reason=dep_reason,
            )

        try:
            utc_hour = SlotUtils.to_utc_hour(exp.slot.start_time)  # type: ignore[union-attr]
        except (ValueError, AttributeError) as exc:
            return CommandResult(
                action_valid=False,
                message=f"Rejected: unparseable slot time — {exc}",
                rejection_reason="invalid_slot_time",
            )

        target_slot = SlotUtils.find_in_day(state.week, day, utc_hour)
        if target_slot is None:
            return CommandResult(
                action_valid=False,
                message=f"Rejected: {utc_hour:02d}:00 UTC not in working hours (08–18)",
                rejection_reason="outside_working_hours",
            )
        if target_slot.event is not None:
            return CommandResult(
                action_valid=False,
                message=f"Rejected: slot {utc_hour:02d}:00 UTC on {day} is occupied",
                rejection_reason="slot_conflict",
            )

        gap = state.spec.min_gap_hours
        if gap > 0:
            for check_hour in range(utc_hour - gap, utc_hour + gap + 1):
                if check_hour == utc_hour:
                    continue
                adjacent = SlotUtils.find_in_day(state.week, day, check_hour)
                if adjacent is not None and adjacent.event is not None:
                    return CommandResult(
                        action_valid=False,
                        message=(
                            f"Rejected: spacing constraint — must leave a {gap}h gap "
                            f"between meetings (conflict at {check_hour:02d}:00 UTC on {day})"
                        ),
                        rejection_reason="spacing_violation",
                    )

        target_slot.event = CalendarEvent(
            event_id=meeting_id,
            title=meeting_id.replace("_", " ").title(),
        )
        return CommandResult(
            action_valid=True,
            message=f"Scheduled '{meeting_id}' on {day} at {utc_hour:02d}:00 UTC.",
            meeting_id=meeting_id,
            utc_hour=utc_hour,
            day=day,
        )


class MoveEventHandler(CommandHandler):
    """Handles move_event — relocates an already-scheduled meeting."""

    def execute(
        self, action: MyCalendarAction, state: EpisodeState
    ) -> CommandResult:
        """Clear the existing slot and book the new one.

        Args:
            action: Agent action with meeting_id, new day, and new slot.
            state: Current episode state; slots are mutated on success.

        Returns:
            CommandResult with updated booking details on success.
        """
        exp = action.expected_action
        meeting_id = (exp.event_id or "").strip()
        new_day = (exp.day or "").lower().strip()

        if meeting_id not in state.scheduled_meeting_ids:
            return CommandResult(
                action_valid=False,
                message=f"Rejected: '{meeting_id}' is not currently scheduled",
                rejection_reason="not_scheduled",
            )
        if new_day not in state.week.days:
            return CommandResult(
                action_valid=False,
                message=f"Rejected: day '{new_day}' not in this task",
                rejection_reason=f"unknown_day:{new_day}",
            )

        try:
            new_utc_hour = SlotUtils.to_utc_hour(exp.slot.start_time)  # type: ignore[union-attr]
        except (ValueError, AttributeError) as exc:
            return CommandResult(
                action_valid=False,
                message=f"Rejected: unparseable slot time — {exc}",
                rejection_reason="invalid_slot_time",
            )

        old_slot = SlotUtils.find_by_meeting_id(state.week, meeting_id)
        if old_slot is None:
            return CommandResult(
                action_valid=False,
                message=f"Rejected: cannot locate current slot for '{meeting_id}'",
                rejection_reason="slot_not_found",
            )

        new_slot = SlotUtils.find_in_day(state.week, new_day, new_utc_hour)
        if new_slot is None:
            return CommandResult(
                action_valid=False,
                message=f"Rejected: {new_utc_hour:02d}:00 UTC on {new_day} not in working hours",
                rejection_reason="outside_working_hours",
            )
        if new_slot.event is not None and new_slot.event.event_id != meeting_id:
            return CommandResult(
                action_valid=False,
                message=(
                    f"Rejected: slot {new_utc_hour:02d}:00 UTC on {new_day} is occupied"
                ),
                rejection_reason="slot_conflict",
            )

        gap = state.spec.min_gap_hours
        if gap > 0:
            for check_hour in range(new_utc_hour - gap, new_utc_hour + gap + 1):
                if check_hour == new_utc_hour:
                    continue
                adjacent = SlotUtils.find_in_day(state.week, new_day, check_hour)
                # Exclude the meeting's own current slot — it will be freed on move
                if (
                    adjacent is not None
                    and adjacent.event is not None
                    and adjacent.event.event_id != meeting_id
                ):
                    return CommandResult(
                        action_valid=False,
                        message=(
                            f"Rejected: spacing constraint — must leave a {gap}h gap "
                            f"between meetings (conflict at {check_hour:02d}:00 UTC on {new_day})"
                        ),
                        rejection_reason="spacing_violation",
                    )

        old_slot.event = None
        new_slot.event = CalendarEvent(
            event_id=meeting_id,
            title=meeting_id.replace("_", " ").title(),
        )
        return CommandResult(
            action_valid=True,
            message=f"Moved '{meeting_id}' to {new_day} at {new_utc_hour:02d}:00 UTC.",
            meeting_id=meeting_id,
            utc_hour=new_utc_hour,
            day=new_day,
        )


class DeleteEventHandler(CommandHandler):
    """Handles delete_event — frees a previously booked meeting slot."""

    def execute(
        self, action: MyCalendarAction, state: EpisodeState
    ) -> CommandResult:
        """Locate and clear the slot occupied by the requested meeting.

        Args:
            action: Agent action with meeting_id to delete.
            state: Current episode state; target slot.event is cleared on success.

        Returns:
            CommandResult confirming deletion, or rejection reason.
        """
        meeting_id = (action.expected_action.event_id or "").strip()

        if meeting_id not in state.scheduled_meeting_ids:
            return CommandResult(
                action_valid=False,
                message=f"Rejected: '{meeting_id}' is not currently scheduled",
                rejection_reason="not_scheduled",
            )

        slot = SlotUtils.find_by_meeting_id(state.week, meeting_id)
        if slot is None:
            return CommandResult(
                action_valid=False,
                message=f"Rejected: cannot locate slot for '{meeting_id}'",
                rejection_reason="slot_not_found",
            )

        slot.event = None
        log.info("deleted meeting '%s' from calendar", meeting_id)
        return CommandResult(
            action_valid=True,
            message=f"Deleted '{meeting_id}' from calendar.",
            meeting_id=meeting_id,
        )


# ── factory ──────────────────────────────────────────────────────────────────

class CommandHandlerFactory:
    """Maps command strings to singleton handler instances."""

    _registry: Dict[str, CommandHandler] = {
        "add_event": AddEventHandler(),
        "reschedule_event": AddEventHandler(),
        "move_event": MoveEventHandler(),
        "search_slot": SearchSlotHandler(),
        "delete_event": DeleteEventHandler(),
    }

    @classmethod
    def get(cls, command: str) -> CommandHandler:
        """Return the handler registered for command.

        Args:
            command: Command name from the agent action.

        Returns:
            Matching CommandHandler instance.

        Raises:
            ValueError: If the command is not registered.
        """
        handler = cls._registry.get(command)
        if handler is None:
            raise ValueError(f"No handler registered for command '{command}'")
        return handler
