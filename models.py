# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Smart Calendar Agent Environment.

The smart_calendar_agent environment is a simple test environment that echoes back messages.
"""

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field, ValidationInfo, field_validator
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime

# 1. Calendar Event
# Define how a calender event actually looks like
class CalendarEvent(BaseModel):
    id: str = Field(description="Unique ID for the event")
    title: str = Field(description="Name of the meeting")
    start: datetime = Field(description="Start time (ISO 8601)")
    end: datetime = Field(description="End time (ISO 8601)")

    @field_validator("end")
    def check_time(cls, v, info: ValidationInfo):
        start = info.data.get("start") if info.data else None
        if start is not None and v <= start:
            raise ValueError("End must be after start")
        return v


# 2. Action
class MyCalendarAction(Action):
    # 'create' = Block time / Schedule meeting
    # 'delete' = Cancel meeting / Make time free
    # 'search' = Check for free slots or existing events
    command: Literal["create", "delete", "search"]
    #args holds the 'payload'
    args: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Parameters for the command. Use 'title', 'start', 'end' for create; 'id' for delete."
    )
    response: Dict[str, str]


# 3. Observation
class MyCalendarObservation(Observation):
    current_time: datetime
    visible_events: List[CalendarEvent]
    instruction: str # The specific task (Easy, Medium, or Hard)
    action_status: str
    score: float
    done: bool


# 4. Internal State
class MyCalendarState(State):
    task_id: int
    steps_taken: int
    # To store the actual database of events inside the state
    calendar_db: List[CalendarEvent] = []