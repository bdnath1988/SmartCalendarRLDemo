# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Smart Calendar Agent Environment Implementation.

A simple test environment that echoes back messages sent to it.
Perfect for testing HTTP server infrastructure.
"""

# from openenv.core.env_server.interfaces import Environment
# import uuid
# from datetime import datetime
# from typing import Tuple, Dict, Any, List

# try:
#     from ..models import MyCalendarAction, MyCalendarObservation, CalendarEvent, MyCalendarState
# except ImportError:
#     from models import MyCalendarAction, MyCalendarObservation, CalendarEvent, MyCalendarState


# class CalendarEnv(Environment):
#     def __init__(self):
#         self.state = MyCalendarState(
#             task_id=0,
#             steps_taken=0,
#         )

#     # ---------------- RESET ----------------
#     def reset(self) -> MyCalendarObservation:
#         self.state = MyCalendarState(
#             task_id=0,
#             steps_taken=0,
#         )

#         return MyCalendarObservation(
#             action_status="reset",
#             score=0.0,
#             done=True,
#         )

#     # ---------------- STEP ----------------
#     def step(self, action: MyCalendarAction) -> MyCalendarObservation:
#         self.state.task_id+=1
#         self.state.steps_taken+=1
#         response_score = len(action.response)
#         command_score = 0
#         calendar_score = 0
#         command = action.command
#         if command == 'block':
#             command_score = 1
#         elif command == "free":
#             command_score = 2
#         else:
#             command_score = 3
#         calendar = action.calendar
#         for item in calendar:
#             calendar_score+=len(item)
#         score = response_score+command_score+calendar_score
#         return MyCalendarObservation(
#             action_status="reset",
#             score=score,
#             done=False,
#             reward=score,
#         )
    
#     # ----------------- STATE -------------------- #
#     def state(self) -> MyCalendarState:
#         return self.state
    
from openenv.core.env_server.interfaces import Environment
from typing import Tuple, List, Optional
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import uuid

try:
    from ..models import MyCalendarAction, MyCalendarObservation,  MyCalendarState, Calendar, Slot, ExpectedAction, PerformedAction
except ImportError:
    from models import MyCalendarAction, MyCalendarObservation, MyCalendarState, Calendar, Slot, ExpectedAction, PerformedAction


# ---------------- ACTION HANDLERS (Strategy Pattern) ----------------
class ActionHandler(ABC):
    """Abstract base class for handling different calendar actions."""

    def __init__(self, calendar: Calendar):
        self.calendar = calendar

    @abstractmethod
    def execute(self, expected: ExpectedAction, performed: PerformedAction) -> Tuple[float, str]:
        """Execute the action and return (reward, message)."""
        pass

    # Helper methods moved from environment
    def _is_slot_available(self, slot: Slot) -> bool:
        """Check if a slot is available (no event assigned)."""
        for cal_slot in self.calendar.slots:
            if cal_slot.start_time == slot.start_time and cal_slot.end_time == slot.end_time:
                return cal_slot.event is None
        return False

    def _update_slot(self, slot: Slot):
        """Update the calendar slot with the given slot."""
        for cal_slot in self.calendar.slots:
            if cal_slot.start_time == slot.start_time and cal_slot.end_time == slot.end_time:
                cal_slot.event = slot.event
                break

    def _find_slot_by_event_id(self, event_id: str) -> Optional[Slot]:
        """Find the slot containing the event with the given ID."""
        for slot in self.calendar.slots:
            if slot.event and slot.event.id == event_id:
                return slot
        return None

    def _find_first_available_slot(self) -> Optional[Slot]:
        """Find the first available slot (no event)."""
        for slot in self.calendar.slots:
            if slot.event is None:
                return slot
        return None


class AddEventHandler(ActionHandler):
    """Handles adding events to the calendar."""

    def execute(self, expected: ExpectedAction, performed: PerformedAction) -> Tuple[float, str]:
        slot_available = self._is_slot_available(expected.slot)
        if performed.success and slot_available:
            self._update_slot(performed.slot)
            return 1.0, "Event added successfully"
        return -1.0, "Failed to add event"


class DeleteEventHandler(ActionHandler):
    """Handles deleting events from the calendar."""

    def execute(self, expected: ExpectedAction, performed: PerformedAction) -> Tuple[float, str]:
        slot = self._find_slot_by_event_id(expected.event_id)
        if slot and performed.success:
            slot.event = None
            return 1.0, "Event deleted successfully"
        return -1.0, "Failed to delete event"


class MoveEventHandler(ActionHandler):
    """Handles moving events between slots."""

    def execute(self, expected: ExpectedAction, performed: PerformedAction) -> Tuple[float, str]:
        current_slot = self._find_slot_by_event_id(expected.event_id)
        target_available = performed.slot is not None and self._is_slot_available(performed.slot)
        if performed.success and current_slot and target_available:
            performed.slot.event = current_slot.event
            self._update_slot(performed.slot)
            current_slot.event = None
            return 1.0, "Event moved successfully"
        return -1.0, "Failed to move event"


class SearchSlotHandler(ActionHandler):
    """Handles searching for available slots."""

    def execute(self, expected: ExpectedAction, performed: PerformedAction) -> Tuple[float, str]:
        available_slot = self._find_first_available_slot()
        if performed.success and performed.slot and available_slot and performed.slot.start_time == available_slot.start_time:
            return 1.0, "Slot found correctly"
        return -1.0, "Incorrect slot found"


# ---------------- COMMAND FACTORY ----------------
class ActionHandlerFactory:
    """Factory for creating action handlers."""

    _handlers = {
        "add_event": AddEventHandler,
        "delete_event": DeleteEventHandler,
        "move_event": MoveEventHandler,
        "search_slot": SearchSlotHandler,
    }

    @staticmethod
    def get_handler(action_type: str, calendar: Calendar) -> ActionHandler:
        """Get the appropriate handler for the action type."""
        handler_class = ActionHandlerFactory._handlers.get(action_type)
        if not handler_class:
            raise ValueError(f"Unknown action type: {action_type}")
        return handler_class(calendar)


class CalendarEnv(Environment):
    def __init__(self):
        self.current_task_id = str(uuid.uuid4())
        self.steps = 0
        self.max_steps = 10
        self.calendar = self._build_daily_calendar()

    # ---------------- RESET ----------------
    def reset(self) -> MyCalendarObservation:
        self.calendar = self._build_daily_calendar()
        self.current_task_id = str(uuid.uuid4())
        self.steps = 0

        return MyCalendarObservation(
            message=f"Environment reset (Task {self.current_task_id})",
            reward=0.0,
            done=False
        )

    # ---------------- STEP ----------------
    def step(self, action: MyCalendarAction) -> MyCalendarObservation:
        self.steps += 1
        self.current_task_id = str(uuid.uuid4())
        done = self.steps >= self.max_steps

        expected = action.expected_action
        performed = action.performed_action

        # Get the appropriate handler and execute
        try:
            handler = ActionHandlerFactory.get_handler(expected.command, self.calendar)
            reward, message = handler.execute(expected, performed)
        except ValueError as e:
            reward, message = -1.0, str(e)

        return MyCalendarObservation(
            message=message,
            reward=reward,
            done=done
        )

    # ---------------- DAILY CALENDAR BUILDER ----------------
    def _build_daily_calendar(self) -> Calendar:
        start_of_day = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        slots: List[Slot] = []
        for hour in range(24):
            slot_start = start_of_day + timedelta(hours=hour)
            slot_end = slot_start + timedelta(hours=1)
            slots.append(Slot(start_time=slot_start, end_time=slot_end))
        return Calendar(slots=slots)

    # ---------------- STATE ----------------
    @property
    def state(self) -> MyCalendarState:
        return MyCalendarState(
            episode_id=self.current_task_id,
            step_count=self.steps,
            calendar=self.calendar
        )
