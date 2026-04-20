# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Smart Calendar Agent Environment Client."""

from typing import Dict
from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from models import (
    Calendar,
    MyCalendarAction,
    MyCalendarObservation,
    MyCalendarState,
)


class SmartCalendarEnv(
    EnvClient[MyCalendarAction, MyCalendarObservation, MyCalendarState]
):
    """
    Client for Smart Calendar Environment (WebSocket-based, persistent state).
    
    Maintains a persistent connection to the environment server,
    enabling efficient multi-step interactions.
    """

    # -------- SEND ACTION --------
    def _step_payload(self, action: MyCalendarAction) -> Dict:
        return action.model_dump(mode="json")

    # -------- PARSE RESPONSE --------
    def _parse_result(self, payload: Dict) -> StepResult[MyCalendarObservation]:
        obs_data = payload.get("observation", {})

        observation = MyCalendarObservation(
            message=obs_data.get("message", ""),
            reward=payload.get("reward", 0),
            done=payload.get("done", False),
            metadata=obs_data.get("metadata", {})
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0),
            done=payload.get("done", False),
        )

    # -------- PARSE STATE --------
    def _parse_state(self, payload: Dict) -> MyCalendarState:
        calendar_data = payload.get("calendar", {"slots": []})
        calendar = Calendar.model_validate(calendar_data)
        
        return MyCalendarState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            calendar=calendar,
            task_objective=payload.get("task_objective", "Schedule 3 meetings efficiently"),
            task_goal=payload.get("task_goal", "schedule 3 meetings"),
            events=payload.get("events", []),
            free_slots=payload.get("free_slots", []),
            target_meetings=payload.get("target_meetings", 3),
            scheduled_meetings=payload.get("scheduled_meetings", 0),
            objective_progress=payload.get("objective_progress", 0.0),
            failed_steps=payload.get("failed_steps", 0),
        )