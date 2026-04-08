# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data models for the Smart Calendar Agent Environment."""

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime


# 1. Calendar Event
class CalendarEvent(BaseModel):
    """Represents a calendar event with a unique identifier and title."""
    event_id: str = Field(description="Unique ID for the event")
    title: str = Field(description="Title of the event")

# 2. Slot
class Slot(BaseModel):
    """Represents a time slot, optionally linked to a calendar event."""
    start_time: datetime = Field(description="Start time (ISO 8601)")
    end_time: datetime = Field(description="End time (ISO 8601)")
    event: Optional[CalendarEvent] = Field(default=None, description="Optional event assigned to this slot")

# 3. Expected Action
class ExpectedAction(BaseModel):
    """Describes an action the calendar agent is expected to perform."""
    command: Literal["add_event", "move_event", "delete_event", "search_slot"] = Field(description="command (mandatory)")
    slot: Optional[Slot] = Field(default=None, description="Time slot for the action")
    event_id: Optional[str] = Field(default=None, description="Event ID")

# 4. Performed Action
class PerformedAction(BaseModel):
    """Represents the result of a performed action, including success and optional slot details."""
    success: bool = Field(description="Whether the action was successful")
    event_id: Optional[str] = Field(default=None, description="Optional ID")
    slot: Optional[Slot] = Field(default=None, description="Optional time range")


# 5. Calendar
class Calendar(BaseModel):
    """Represents a calendar containing a list of available slots."""
    slots: List[Slot] = Field(description="List of slots")

# 6. My Calendar Action
class MyCalendarAction(Action):
    """Encapsulates a calendar action with expected and actual performed details."""
    expected_action: ExpectedAction = Field(description="Expected action to perform")
    performed_action: PerformedAction = Field(description="Action that was actually performed")


# 3. Observation
class MyCalendarObservation(Observation):
    """Represents observations returned from the calendar environment."""
    message: str = Field(description="Observation message")


# 4. Internal State
class MyCalendarState(State):
    """Tracks the internal state of the calendar environment."""
    calendar: Calendar = Field(description="Current calendar state")
    task_objective: str = Field(default="Schedule 3 meetings efficiently", description="Current task objective for the episode")
    target_meetings: int = Field(default=3, description="Target number of meetings for objective completion")
    scheduled_meetings: int = Field(default=0, description="Number of meetings currently scheduled")
    objective_progress: float = Field(default=0.0, description="Progress toward objective in range [0, 1]")