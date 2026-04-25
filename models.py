# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data models for the Smart Calendar Agent Environment."""

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Literal
from datetime import datetime, date, timezone as dt_timezone
from zoneinfo import ZoneInfo

from pydantic import model_validator
import json

# 1. Calendar Event
class CalendarEvent(BaseModel):
    """Represents a calendar event with a unique identifier and title."""
    event_id: str = Field(description="Unique ID for the event")
    title: str = Field(description="Title of the event")

# 2. Slot
class Slot(BaseModel):
    """Represents a time slot, optionally linked to a calendar event."""
    start_time: str = Field(description="Start time (ISO 8601)")
    end_time: str = Field(description="End time (ISO 8601)")
    event: Optional[CalendarEvent] = Field(default=None, description="Optional event assigned to this slot")

# 3. Expected Action
class ExpectedAction(BaseModel):
    """Describes an action the calendar agent is expected to perform."""
    command: Literal["add_event", "move_event", "delete_event", "search_slot", "reschedule_event"] = Field(description="command (mandatory)")
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



# 6. Attendee
class Attendee(BaseModel):
    """Represents a meeting attendee with timezone and scheduling preferences."""

    name: str = Field(description="Attendee name")
    timezone: str = Field(description="IANA timezone string, e.g. 'Asia/Kolkata'")
    priority: int = Field(description="Priority level: 1=CEO, 2=Director, 3=Manager")
    preferred_start_hour: int = Field(description="Preferred meeting start hour in local time")
    preferred_end_hour: int = Field(description="Preferred meeting end hour in local time")


# 7. DayCalendar
class DayCalendar(BaseModel):
    """Represents a single work day with 10 hourly slots covering 8am-6pm UTC."""

    day_name: str = Field(description="Day of week, e.g. 'monday'")
    slots: List[Slot] = Field(description="10 hourly slots from 8am-6pm UTC")


# 8. WeeklyCalendar
class WeeklyCalendar(BaseModel):
    """Represents a five-day work week with per-day calendars and attendee roster."""

    days: Dict[str, DayCalendar] = Field(description="Per-day calendars keyed by day name")
    attendees: List[Attendee] = Field(description="Attendees participating this week")


# 9. MeetingNode
class MeetingNode(BaseModel):
    """Node in a meeting dependency graph."""

    meeting_id: str = Field(description="Unique meeting identifier")
    title: str = Field(description="Human-readable meeting title")
    required_attendees: List[str] = Field(description="Names of required attendees")
    dependencies: List[str] = Field(default_factory=list, description="Meeting IDs that must be scheduled first")
    duration_hours: int = Field(default=1, description="Duration in hours")


# ---- Timezone utility functions (zoneinfo only, no extra packages) ----

def utc_to_local(utc_hour: int, timezone: str) -> int:
    """Convert a UTC hour to the local hour for a given IANA timezone.

    Args:
        utc_hour: Hour in UTC (0-23).
        timezone: IANA timezone string, e.g. 'Asia/Kolkata'.

    Returns:
        Local hour (0-23), truncated — not rounded.
    """
    today = date.today()
    dt_utc = datetime(today.year, today.month, today.day, utc_hour, 0, 0, tzinfo=dt_timezone.utc)
    return dt_utc.astimezone(ZoneInfo(timezone)).hour


def local_to_utc(local_hour: int, timezone: str) -> int:
    """Convert a local hour to UTC for a given IANA timezone.

    Args:
        local_hour: Hour in local time (0-23).
        timezone: IANA timezone string, e.g. 'Asia/Kolkata'.

    Returns:
        UTC hour (0-23), truncated — not rounded.
    """
    today = date.today()
    dt_local = datetime(today.year, today.month, today.day, local_hour, 0, 0, tzinfo=ZoneInfo(timezone))
    return dt_local.astimezone(dt_timezone.utc).hour


def is_within_preference(utc_hour: int, attendee: Attendee) -> bool:
    """Check whether a UTC slot hour falls within an attendee's preferred local working hours.

    Preferred hours are soft constraints used for satisfaction scoring only;
    this function does not block slot booking.

    Args:
        utc_hour: Slot start hour in UTC.
        attendee: Attendee whose preferences to check.

    Returns:
        True if local_hour is in [preferred_start_hour, preferred_end_hour).
    """
    local_hour = utc_to_local(utc_hour, attendee.timezone)
    return attendee.preferred_start_hour <= local_hour < attendee.preferred_end_hour


class MyCalendarAction(Action):
    """Encapsulates a calendar action with expected and actual performed details."""
    expected_action: ExpectedAction = Field(description="Expected action to perform")
    performed_action: PerformedAction = Field(description="Action that was actually performed")

    @model_validator(mode='before')
    @classmethod
    def parse_stringified_actions(cls, values: Any) -> Any:
        if isinstance(values, dict):
            for field in ["expected_action", "performed_action"]:
                val = values.get(field)
                if isinstance(val, str):
                    try:
                        values[field] = json.loads(val)
                    except Exception:
                        pass
        return values


# 3. Observation
class MyCalendarObservation(Observation):
    """Represents observations returned from the calendar environment."""
    message: str = Field(description="Observation message")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional observation metadata")


# 4. Internal State
class MyCalendarState(State):
    """Tracks the internal state of the calendar environment."""
    calendar: Calendar = Field(description="Current calendar state")
    task_objective: str = Field(default="Schedule 3 meetings efficiently", description="Current task objective for the episode")
    task_goal: str = Field(default="schedule 3 meetings", description="Short task goal string for prompting")
    events: List[str] = Field(default_factory=list, description="List of scheduled event IDs")
    free_slots: List[str] = Field(default_factory=list, description="List of currently free slot labels")
    target_meetings: int = Field(default=3, description="Target number of meetings for objective completion")
    scheduled_meetings: int = Field(default=0, description="Number of meetings currently scheduled")
    objective_progress: float = Field(default=0.0, description="Progress toward objective in range [0, 1]")
    failed_steps: int = Field(default=0, description="Number of failed steps in the current episode")
    week: Optional[WeeklyCalendar] = Field(default=None, description="Weekly calendar for multi-day tasks")
    attendees: List[Attendee] = Field(default_factory=list, description="Attendees for the current episode")
    cascade_conflicts: List[str] = Field(default_factory=list, description="Meeting IDs with unresolved dependency conflicts")
    attendee_satisfaction: float = Field(default=0.0, description="Weighted satisfaction score across attendees in [0, 1]")
    dependency_graph: Dict[str, List[str]] = Field(default_factory=dict, description="Meeting dependency adjacency list")
    scheduled_meeting_ids: List[str] = Field(default_factory=list, description="Meeting IDs successfully scheduled this episode")