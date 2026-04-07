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
from typing import Tuple, List
from datetime import datetime
import random

try:
    from ..models import MyCalendarAction, MyCalendarObservation, CalendarEvent, MyCalendarState
except ImportError:
    from models import MyCalendarAction, MyCalendarObservation, CalendarEvent, MyCalendarState


class CalendarEnv(Environment):
    def __init__(self):
        self.events: List[CalendarEvent] = []
        self.current_task_id = 0
        self.steps = 0
        self.max_steps = 10

    # ---------------- RESET ----------------
    def reset(self) -> MyCalendarObservation:
        self.events = []
        self.current_task_id = random.choice([0, 1])  # multiple tasks
        self.steps = 0

        return MyCalendarObservation(
            events=self.events,
            message=f"Environment reset (Task {self.current_task_id})",
            reward=0.0,
            done=False
        )

    # ---------------- STEP ----------------
    def step(self, action: MyCalendarAction) -> MyCalendarObservation:
        self.steps += 1
        reward = 0
        message = ""

        # -------- ADD EVENT --------
        if action.action_type == "add_event":
            if action.event is None:
                return MyCalendarObservation(
                    events=self.events,
                    message="No event provided ❌",
                    reward=-1,
                    done=False
                )

            conflict = self._has_conflict(action.event)

            if conflict:
                reward -= 2
                message = "Conflict detected ❌"
            else:
                self.events.append(action.event)
                reward += 1
                message = "Event added ✅"

        # -------- MOVE EVENT --------
        elif action.action_type == "move_event":
            found = False
            for e in self.events:
                if e.id == action.event_id:
                    found = True

                    if action.new_start is None or action.new_end is None:
                        return MyCalendarObservation(
                            events=self.events,
                            message="Invalid move parameters ❌",
                            reward=-1,
                            done=False
                        )

                    e.start = action.new_start
                    e.end = action.new_end
                    reward += 1
                    message = "Event moved 🔄"

            if not found:
                reward -= 1
                message = "Event not found ❌"

        # -------- DELETE EVENT --------
        elif action.action_type == "delete_event":
            before = len(self.events)
            self.events = [e for e in self.events if e.id != action.event_id]

            if len(self.events) < before:
                reward += 0.5
                message = "Event deleted 🗑️"
            else:
                reward -= 1
                message = "Event not found ❌"

        # -------- SMART PENALTY (no back-to-back meetings) --------
        if len(self.events) >= 2:
            self.events.sort(key=lambda x: x.start)

            for i in range(len(self.events) - 1):
                gap = (self.events[i + 1].start - self.events[i].end).total_seconds() / 60
                if gap < 15:
                    reward -= 0.5

        # -------- TASK GRADING --------
        task_reward, task_done = self._grade()
        reward += task_reward

        done = task_done or self.steps >= self.max_steps

        return MyCalendarObservation(
            events=self.events,
            message=message,
            reward=reward,
            done=done
        )

    # ---------------- CONFLICT CHECK ----------------
    def _has_conflict(self, new_event: CalendarEvent) -> bool:
        for e in self.events:
            if not (new_event.end <= e.start or new_event.start >= e.end):
                return True
        return False

    # ---------------- TASK GRADER ----------------
    def _grade(self) -> Tuple[float, bool]:

        # 🟢 Task 0: Add at least one valid event
        if self.current_task_id == 0:
            if len(self.events) > 0:
                return 1.0, True

        # 🟡 Task 1: Ensure no overlaps
        if self.current_task_id == 1:
            for i in range(len(self.events)):
                for j in range(i + 1, len(self.events)):
                    if not (
                        self.events[i].end <= self.events[j].start or
                        self.events[i].start >= self.events[j].end
                    ):
                        return -1.0, True
            return 1.0, True

        return 0.0, False

    # ---------------- STATE ----------------
    @property
    def state(self) -> MyCalendarState:
        return MyCalendarState(
            episode_id="calendar-session",
            step_count=self.steps,
            task_id=self.current_task_id,
            steps_taken=self.steps
        )
