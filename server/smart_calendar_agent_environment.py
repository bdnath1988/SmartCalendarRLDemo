# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Smart Calendar Agent Environment Implementation.

A RL-based scheduling environment that routes agent actions through
specific handlers and evaluates them against formal task metrics.
"""

from openenv.core.env_server.interfaces import Environment
from typing import Tuple, List, Optional, Any
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import random
import uuid

try:
    from ..models import MyCalendarAction, MyCalendarObservation,  MyCalendarState, Calendar, Slot, ExpectedAction, PerformedAction, CalendarEvent
except ImportError:
    from models import MyCalendarAction, MyCalendarObservation, MyCalendarState, Calendar, Slot, ExpectedAction, PerformedAction, CalendarEvent

try:
    from task_definitions import get_task_by_level
except ImportError:
    from ..task_definitions import get_task_by_level

try:
    from server.grader import Grader
except ImportError:
    from .grader import Grader


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
    def _parse_time(self, t: Any) -> datetime:
        if isinstance(t, datetime):
            return t
        if isinstance(t, str):
            return datetime.fromisoformat(t)
        return t

    def _is_same_time(self, t1: Any, t2: Any) -> bool:
        """Check if two times are within 60 seconds of each other."""
        if t1 is None or t2 is None:
            return t1 == t2
        dt1 = self._parse_time(t1)
        dt2 = self._parse_time(t2)
        return abs((dt1 - dt2).total_seconds()) < 60

    def _is_slot_available(self, slot: Slot) -> bool:
        """Check if a slot is available (no event assigned)."""
        for cal_slot in self.calendar.slots:
            if self._is_same_time(cal_slot.start_time, slot.start_time) and self._is_same_time(cal_slot.end_time, slot.end_time):
                return cal_slot.event is None
        return False

    def _update_slot(self, slot: Slot):
        """Update the calendar slot with the given slot."""
        for cal_slot in self.calendar.slots:
            if self._is_same_time(cal_slot.start_time, slot.start_time) and self._is_same_time(cal_slot.end_time, slot.end_time):
                cal_slot.event = slot.event
                break

    def _find_slot_by_event_id(self, event_id: str) -> Optional[Slot]:
        """Find the slot containing the event with the given ID."""
        for slot in self.calendar.slots:
            if slot.event and slot.event.event_id == event_id:
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
        # Duplicate event IDs should be explicitly penalized.
        if expected.event_id and self._find_slot_by_event_id(expected.event_id):
            return 0.0, "Duplicate event id"

        slot_available = self._is_slot_available(expected.slot)

        if not slot_available:
            return 0.0, "Time slot already occupied"
        
        # Full success: both slot available and operation succeeded
        if slot_available and performed.success:
            event_slot = performed.slot or expected.slot
            if event_slot and event_slot.event is None:
                event_id = performed.event_id or expected.event_id or "event"
                title = None
                if performed.slot and performed.slot.event and performed.slot.event.title:
                    title = performed.slot.event.title
                elif expected.slot and expected.slot.event and expected.slot.event.title:
                    title = expected.slot.event.title
                else:
                    title = expected.event_id or performed.event_id or "event"
                event_slot.event = CalendarEvent(event_id=event_id, title=title)
            self._update_slot(event_slot)
            return 0.5, "Event added successfully"
        
        # Partial credit: correct slot identified but operation failed
        if slot_available and not performed.success:
            return 0.15, "Event not added (slot identified but execution failed)"


class DeleteEventHandler(ActionHandler):
    """Handles deleting events from the calendar."""

    def execute(self, expected: ExpectedAction, performed: PerformedAction) -> Tuple[float, str]:
        slot = self._find_slot_by_event_id(expected.event_id)
        
        # Full success: event found and deleted successfully
        if slot and performed.success:
            slot.event = None
            return 0.5, "Event deleted successfully"
        
        # Partial credit: event found but deletion failed
        if slot and not performed.success:
            return 0.15, "Event found but deletion failed"
        
        # Wrong logic: deletion succeeded but event not found
        if not slot and performed.success:
            return 0.0, "Event deletion succeeded but event not found"
        
        # Complete failure: event not found and deletion failed
        return 0.0, "Failed to delete event (not found)"


class MoveEventHandler(ActionHandler):
    """Handles moving events between slots."""

    def execute(self, expected: ExpectedAction, performed: PerformedAction) -> Tuple[float, str]:
        current_slot = self._find_slot_by_event_id(expected.event_id)
        target_available = performed.slot is not None and self._is_slot_available(performed.slot)
        
        # Full success: current slot found, target available, and move successful
        if performed.success and current_slot and target_available:
            performed.slot.event = current_slot.event
            self._update_slot(performed.slot)
            current_slot.event = None
            return 0.5, "Event moved successfully"
        
        # Partial credit: current slot found but move failed
        if not performed.success and current_slot:
            return 0.15, "Current event located but move failed"
        
        # Partial credit: current slot and target valid but move failed
        if not performed.success and current_slot and target_available:
            return 0.2, "Event and target slot valid but move execution failed"
        
        # Wrong logic: move succeeded but prerequisites not met
        if performed.success and (not current_slot or not target_available):
            return 0.0, "Move succeeded but event or target slot invalid"
        
        # Complete failure
        return 0.0, "Failed to move event"


class SearchSlotHandler(ActionHandler):
    """Handles searching for available slots."""

    def execute(self, expected: ExpectedAction, performed: PerformedAction) -> Tuple[float, str]:
        available_slot = self._find_first_available_slot()
        
        # Full success: found available slot and returned correct one
        if performed.success and performed.slot and available_slot and self._is_same_time(performed.slot.start_time, available_slot.start_time):
            return 0.5, "Slot found correctly"
        
        # Partial credit: search succeeded but returned wrong slot
        if performed.success and performed.slot and available_slot and not self._is_same_time(performed.slot.start_time, available_slot.start_time):
            return 0.1, "Search succeeded but returned wrong slot"
        
        # Partial credit: search succeeded but returned something when slots available
        if performed.success and performed.slot is None and available_slot:
            return 0.05, "No slot returned but slots are available"
        
        # Wrong logic: search succeeded but no available slots exist (shouldn't happen)
        if performed.success and not available_slot:
            return 0.0, "Slot search succeeded but no slots are available"
        
        # Complete failure: search failed
        return 0.0, "Search failed"


# ---------------- COMMAND FACTORY ----------------
class ActionHandlerFactory:
    """Factory for creating action handlers based on action type."""

    _handlers = {
        "add_event": AddEventHandler,
        "delete_event": DeleteEventHandler,
        "move_event": MoveEventHandler,
        "search_slot": SearchSlotHandler,
    }

    @staticmethod
    def get_handler(action_type: str, calendar: Calendar) -> ActionHandler:
        """Return the handler instance for the requested action type.

        Args:
            action_type: The command name for the calendar action.
            calendar: The current calendar state to operate on.

        Returns:
            An ActionHandler instance for the requested action.

        Raises:
            ValueError: If the action type is not recognized.
        """
        handler_class = ActionHandlerFactory._handlers.get(action_type)
        if not handler_class:
            raise ValueError(f"Unknown action type: {action_type}")
        return handler_class(calendar)


class CalendarEnv(Environment):
    """A simple smart calendar environment for OpenEnv-style agents.

    The environment maintains a daily calendar, tracks episode and step state,
    and routes agent actions through a strategy-based action handler.
    """

    def __init__(self):
        self.current_task_id = str(uuid.uuid4())
        self.steps = 0
        self.failed_steps = 0
        self.max_steps = 10
        self.task_type = "medium"
        
        # Load formal task definition
        self.task_def = get_task_by_level(self.task_type)
        self.task_objective = self.task_def.objective
        self.task_goal = self.task_def.goal
        self.target_meetings = self.task_def.metrics.target_meetings

        self.calendar = self._build_daily_calendar()
        self.previous_action_signature = ""

    # ---------------- RESET ----------------
    def reset(self) -> MyCalendarObservation:
        """Reset the environment to a fresh daily calendar and new task ID.

        Returns:
            A MyCalendarObservation indicating the reset state.
        """
        self.calendar = self._build_daily_calendar()
        self.current_task_id = str(uuid.uuid4())
        self.steps = 0
        self.failed_steps = 0
        self.previous_action_signature = ""
        self._assign_task()

        return MyCalendarObservation(
            message=f"Environment reset (Task {self.current_task_id}) | Objective: {self.task_objective}",
            reward=0.0,
            done=False
        )

    # ---------------- STEP ----------------
    def step(self, action: MyCalendarAction) -> MyCalendarObservation:
        """Advance the environment by applying the agent's calendar action.

        The action is routed through the appropriate handler based on the
        expected command, and the calendar state is updated in place.

        Args:
            action: The incoming MyCalendarAction from the agent.

        Returns:
            A MyCalendarObservation with reward, done status, and message.
        """
        self.steps += 1

        action_signature = action.model_dump_json()

        expected = action.expected_action
        performed = action.performed_action

        # Get the appropriate handler and execute
        try:
            handler = ActionHandlerFactory.get_handler(expected.command, self.calendar)
            base_reward, message = handler.execute(expected, performed)
        except ValueError as e:
            base_reward, message = 0.0, str(e)

        # Reward shaping: strictly 0.0-1.0 range based on Phase 2 requirements.
        # base_reward from handlers is [0.0, 0.5].
        # score from Grader is [0.0, 1.0].
        score = self.compute_score()
        reward = base_reward + (score * 0.5)

        if action_signature == self.previous_action_signature:
            reward *= 0.5 # Penalize repeat actions without going negative

        reward = max(0.0, min(1.0, float(reward)))

        if reward < 0.1:
            self.failed_steps += 1

        self.previous_action_signature = action_signature

        # Episode ends when max steps are reached, the goal is achieved, or too many failures occur.
        done = self.steps >= self.max_steps or scheduled >= self.target_meetings or self.failed_steps >= 3

        return MyCalendarObservation(
            message=message,
            reward=reward,
            done=done,
            metadata={"score": score}
        )

    # ---------------- DAILY CALENDAR BUILDER ----------------
    def _build_daily_calendar(self) -> Calendar:
        """Create an empty calendar with 24 hourly slots for the current day."""
        start_of_day = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        slots: List[Slot] = []
        for hour in range(24):
            slot_start = start_of_day + timedelta(hours=hour)
            slot_end = slot_start + timedelta(hours=1)
            slots.append(Slot(start_time=slot_start.isoformat(), end_time=slot_end.isoformat()))
        return Calendar(slots=slots)

    def _count_scheduled_meetings(self) -> int:
        """Count how many slots currently contain an event."""
        return sum(1 for slot in self.calendar.slots if slot.event is not None)

    def _build_free_slot_labels(self) -> List[str]:
        """Build compact labels for free slots to help prompting."""
        free_slots: List[str] = []
        for slot in self.calendar.slots:
            if slot.event is None:
                st = datetime.fromisoformat(slot.start_time).strftime('%H:%M') if isinstance(slot.start_time, str) else slot.start_time.strftime('%H:%M')
                en = datetime.fromisoformat(slot.end_time).strftime('%H:%M') if isinstance(slot.end_time, str) else slot.end_time.strftime('%H:%M')
                free_slots.append(f"{st}-{en}")
        return free_slots

    def _assign_task(self) -> None:
        """Assign one of the explicit evaluation tasks for this episode."""
        self.task_type = random.choice(["easy", "medium", "hard"])
        self.task_def = get_task_by_level(self.task_type)
        self.target_meetings = self.task_def.metrics.target_meetings
        self.task_objective = self.task_def.objective
        self.task_goal = self.task_def.goal

    def compute_score(self) -> float:
        """Compute deterministic task score in [0, 1] using the Grader class."""
        return Grader.compute_score(self.calendar, self.task_def)

    def _task_reward_adjustment(self) -> float:
        """Return task-specific reward adjustment based on Grader."""
        # Calculate current score representation as part of continuous reward shaping
        current_score = self.compute_score()
        return max(0.0, current_score * 0.5)

    # ---------------- STATE ----------------
    @property
    def state(self) -> MyCalendarState:
        """Return the current environment state as a MyCalendarState object."""
        scheduled = self._count_scheduled_meetings()
        progress = min(1.0, scheduled / self.target_meetings) if self.target_meetings > 0 else 0.0
        events = [slot.event.event_id for slot in self.calendar.slots if slot.event is not None and slot.event.event_id]
        return MyCalendarState(
            episode_id=self.current_task_id,
            step_count=self.steps,
            calendar=self.calendar,
            task_objective=self.task_objective,
            task_goal=self.task_goal,
            events=events,
            free_slots=self._build_free_slot_labels(),
            target_meetings=self.target_meetings,
            scheduled_meetings=scheduled,
            objective_progress=progress,
            failed_steps=self.failed_steps,
        )
