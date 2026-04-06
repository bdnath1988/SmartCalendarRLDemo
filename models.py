# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Smart Calendar Agent Environment.

The smart_calendar_agent environment is a simple test environment that echoes back messages.
"""

from pydantic import BaseModel, Field, ValidationInfo, field_validator
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime

# 1. Calendar Event
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
class Action(BaseModel):
    command: Literal["create", "delete", "search", "wait"]
    args: Dict[str, Any] = Field(default_factory=dict)


# 3. Observation
class Observation(BaseModel):
    current_time: str
    events: List[CalendarEvent]
    user_preferences: str
    last_action_status: str
    done: bool


# 4. Internal State
class State(BaseModel):
    task_id: int
    steps_taken: int
    max_steps: int = 10