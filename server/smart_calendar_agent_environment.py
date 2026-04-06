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

import uuid
from datetime import datetime
from typing import Tuple, Dict, Any, List

try:
    from ..models import Action, Observation, CalendarEvent, State
except ImportError:
    from models import Action, Observation, CalendarEvent, State


class CalendarEnv:
    def __init__(self):
        self.tasks = {
            0: {"goal": "Coffee at 2PM", "time": "2026-04-06T14:00:00", "duration": 30},
            1: {"goal": "No overlapping events"},
            2: {"goal": "Reschedule Lunch", "target_id": "lunch-1"}
        }

        self.db: List[CalendarEvent] = []
        self.current_task_id = 0
        self.steps = 0
        self.max_steps = 10

    # ---------------- RESET ----------------
    def reset(self, task_id: int = 0) -> Observation:
        self.current_task_id = task_id
        self.steps = 0

        self.db = [
            CalendarEvent(
                id="lunch-1",
                title="Lunch",
                start=datetime.fromisoformat("2026-04-06T12:00:00"),
                end=datetime.fromisoformat("2026-04-06T13:00:00")
            )
        ]

        return self._get_obs("Initial State", False)

    # ---------------- STEP ----------------
    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict]:
        self.steps += 1
        msg = "Action Executed"
        conflict = False

        try:
            if action.command == "create":

                required = ["title", "start", "end"]
                if not all(k in action.args for k in required):
                    raise ValueError("Missing required fields")

                new_event = CalendarEvent(
                    id=str(uuid.uuid4())[:8],
                    title=action.args["title"],
                    start=datetime.fromisoformat(action.args["start"]),
                    end=datetime.fromisoformat(action.args["end"])
                )

                if self._has_conflict(new_event):
                    msg = "Conflict detected"
                    conflict = True
                else:
                    self.db.append(new_event)

            elif action.command == "delete":
                target_id = action.args.get("id")
                self.db = [e for e in self.db if e.id != target_id]

            elif action.command == "wait":
                msg = "Skipped turn"

        except Exception as e:
            msg = f"Error: {str(e)}"

        reward, done = self._grade(conflict)

        return self._get_obs(msg, done), reward, done, {}

    # ---------------- CONFLICT CHECK ----------------
    def _has_conflict(self, new_event: CalendarEvent) -> bool:
        for e in self.db:
            if not (new_event.end <= e.start or new_event.start >= e.end):
                return True
        return False

    # ---------------- OBS ----------------
    def _get_obs(self, status: str, done: bool) -> Observation:
        return Observation(
            current_time="2026-04-06T09:00:00",
            events=self.db,
            user_preferences=str(self.tasks[self.current_task_id]),
            last_action_status=status,
            done=done
        )

    # ---------------- GRADER ----------------
    def _grade(self, conflict: bool) -> Tuple[float, bool]:

        # 🟢 Task 0: Coffee at 2PM
        if self.current_task_id == 0:
            for e in self.db:
                if "Coffee" in e.title and e.start.hour == 14:
                    return 1.0, True

        # 🟡 Task 1: No overlap
        if self.current_task_id == 1:
            for i in range(len(self.db)):
                for j in range(i + 1, len(self.db)):
                    if not (
                        self.db[i].end <= self.db[j].start or
                        self.db[i].start >= self.db[j].end
                    ):
                        return -1.0, True
            return 1.0, True

        # 🔴 Task 2: Move lunch after 3PM
        if self.current_task_id == 2:
            for e in self.db:
                if e.id == "lunch-1" and e.start.hour >= 15:
                    return 1.0, True

        # ⏳ Step limit
        if self.steps >= self.max_steps:
            return 0.0, True

        # 🔄 Reward shaping
        reward = 0.1
        if conflict:
            reward -= 0.5

        return reward, False

    # ---------------- INTERNAL STATE ----------------
    def get_internal_state(self) -> State:
        return State(
            task_id=self.current_task_id,
            steps_taken=self.steps
        )