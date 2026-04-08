# # Copyright (c) Meta Platforms, Inc. and affiliates.
# # All rights reserved.
# #
# # This source code is licensed under the BSD-style license found in the
# # LICENSE file in the root directory of this source tree.

# """Smart Calendar Agent Environment Client."""

# from typing import Dict

# from openenv.core import EnvClient
# from openenv.core.client_types import StepResult
# from openenv.core.env_server.types import State

# from .models import SmartCalendarAgentAction, SmartCalendarAgentObservation


# class SmartCalendarAgentEnv(
#     EnvClient[SmartCalendarAgentAction, SmartCalendarAgentObservation, State]
# ):
#     """
#     Client for the Smart Calendar Agent Environment.

#     This client maintains a persistent WebSocket connection to the environment server,
#     enabling efficient multi-step interactions with lower latency.
#     Each client instance has its own dedicated environment session on the server.

#     Example:
#         >>> # Connect to a running server
#         >>> with SmartCalendarAgentEnv(base_url="http://localhost:8000") as client:
#         ...     result = client.reset()
#         ...     print(result.observation.echoed_message)
#         ...
#         ...     result = client.step(SmartCalendarAgentAction(message="Hello!"))
#         ...     print(result.observation.echoed_message)

#     Example with Docker:
#         >>> # Automatically start container and connect
#         >>> client = SmartCalendarAgentEnv.from_docker_image("smart_calendar_agent-env:latest")
#         >>> try:
#         ...     result = client.reset()
#         ...     result = client.step(SmartCalendarAgentAction(message="Test"))
#         ... finally:
#         ...     client.close()
#     """

#     def _step_payload(self, action: SmartCalendarAgentAction) -> Dict:
#         """
#         Convert SmartCalendarAgentAction to JSON payload for step message.

#         Args:
#             action: SmartCalendarAgentAction instance

#         Returns:
#             Dictionary representation suitable for JSON encoding
#         """
#         return {
#             "message": action.message,
#         }

#     def _parse_result(self, payload: Dict) -> StepResult[SmartCalendarAgentObservation]:
#         """
#         Parse server response into StepResult[SmartCalendarAgentObservation].

#         Args:
#             payload: JSON response data from server

#         Returns:
#             StepResult with SmartCalendarAgentObservation
#         """
#         obs_data = payload.get("observation", {})
#         observation = SmartCalendarAgentObservation(
#             echoed_message=obs_data.get("echoed_message", ""),
#             message_length=obs_data.get("message_length", 0),
#             done=payload.get("done", False),
#             reward=payload.get("reward"),
#             metadata=obs_data.get("metadata", {}),
#         )

#         return StepResult(
#             observation=observation,
#             reward=payload.get("reward"),
#             done=payload.get("done", False),
#         )

#     def _parse_state(self, payload: Dict) -> State:
#         """
#         Parse server response into State object.

#         Args:
#             payload: JSON response from state request

#         Returns:
#             State object with episode_id and step_count
#         """
#         return State(
#             episode_id=payload.get("episode_id"),
#             step_count=payload.get("step_count", 0),
#         )
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
    Client for Smart Calendar Environment (WebSocket-based, persistent state)
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
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0),
            done=payload.get("done", False),
        )

    # -------- PARSE STATE --------
    def _parse_state(self, payload: Dict) -> MyCalendarState:
        return MyCalendarState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            calendar=Calendar.model_validate(payload.get("calendar", {"slots": []})),
            task_objective=payload.get("task_objective", "Schedule 3 meetings efficiently"),
            target_meetings=payload.get("target_meetings", 3),
            scheduled_meetings=payload.get("scheduled_meetings", 0),
            objective_progress=payload.get("objective_progress", 0.0),
        )