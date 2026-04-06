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

from openenv.core.env_server.interfaces import Environment
import uuid
from datetime import datetime
from typing import Tuple, Dict, Any, List

try:
    from ..models import MyCalendarAction, MyCalendarObservation, CalendarEvent, MyCalendarState
except ImportError:
    from models import MyCalendarAction, MyCalendarObservation, CalendarEvent, MyCalendarState


class CalendarEnv(Environment):
    def __init__(self):
        self.state = MyCalendarState(
            task_id=0,
            steps_taken=0,
        )

    # ---------------- RESET ----------------
    def reset(self) -> MyCalendarObservation:
        self.state = MyCalendarState(
            task_id=0,
            steps_taken=0,
        )

        return MyCalendarObservation(
            action_status="reset",
            score=0.0,
            done=True,
        )

    # ---------------- STEP ----------------
    def step(self, action: MyCalendarAction) -> MyCalendarObservation:
        self.state.task_id+=1
        self.state.steps_taken+=1
        response_score = len(action.response)
        command_score = 0
        calendar_score = 0
        command = action.command
        if command == 'block':
            command_score = 1
        elif command == "free":
            command_score = 2
        else:
            command_score = 3
        calendar = action.calendar
        for item in calendar:
            calendar_score+=len(item)
        score = response_score+command_score+calendar_score
        return MyCalendarObservation(
            action_status="reset",
            score=score,
            done=False,
            reward=score,
        )
    
    # ----------------- STATE -------------------- #
    def state(self) -> MyCalendarState:
        return self.state
    
    # # ---------------- CONFLICT CHECK ----------------
    # def _has_conflict(self, new_event: CalendarEvent) -> bool:
    #     for e in self.db:
    #         if not (new_event.end <= e.start or new_event.start >= e.end):
    #             return True
    #     return False

    # # ---------------- OBS ----------------
    # def _get_obs(self, status: str, done: bool) -> Observation:
    #     return Observation(
    #         current_time="2026-04-06T09:00:00",
    #         events=self.db,
    #         user_preferences=str(self.tasks[self.current_task_id]),
    #         last_action_status=status,
    #         done=done
    #     )

    # # ---------------- GRADER ----------------
    # def _grade(self, conflict: bool) -> Tuple[float, bool]:

    #     # 🟢 Task 0: Coffee at 2PM
    #     if self.current_task_id == 0:
    #         for e in self.db:
    #             if "Coffee" in e.title and e.start.hour == 14:
    #                 return 1.0, True

    #     # 🟡 Task 1: No overlap
    #     if self.current_task_id == 1:
    #         for i in range(len(self.db)):
    #             for j in range(i + 1, len(self.db)):
    #                 if not (
    #                     self.db[i].end <= self.db[j].start or
    #                     self.db[i].start >= self.db[j].end
    #                 ):
    #                     return -1.0, True
    #         return 1.0, True

    #     # 🔴 Task 2: Move lunch after 3PM
    #     if self.current_task_id == 2:
    #         for e in self.db:
    #             if e.id == "lunch-1" and e.start.hour >= 15:
    #                 return 1.0, True

    #     # ⏳ Step limit
    #     if self.steps >= self.max_steps:
    #         return 0.0, True

    #     # 🔄 Reward shaping
    #     reward = 0.1
    #     if conflict:
    #         reward -= 0.5

    #     return reward, False

    # # ---------------- INTERNAL STATE ----------------
    # def get_internal_state(self) -> State:
    #     return State(
    #         task_id=self.current_task_id,
    #         steps_taken=self.steps
    #     )