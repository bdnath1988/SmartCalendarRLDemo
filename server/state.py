# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Value objects for the Round-2 calendar environment.

EpisodeState  — mutable shared state passed to command handlers.
CommandResult — immutable return value from command handlers.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    from ..models import Attendee, WeeklyCalendar
    from ..task_definitions import TaskSpec
except ImportError:
    from models import Attendee, WeeklyCalendar
    from task_definitions import TaskSpec


@dataclass
class EpisodeState:
    """Mutable episode state shared between CalendarEnv and command handlers.

    Attributes:
        spec: Active task specification.
        week: Weekly calendar whose slots may be mutated by handlers.
        attendees: Attendee personas active this episode.
        scheduled_meeting_ids: Ordered list of successfully booked meeting IDs.
        scheduled_details: Per-meeting booking metadata (day, UTC start/end).
        step_count: Number of steps taken so far this episode.
    """

    spec: TaskSpec
    week: WeeklyCalendar
    attendees: List[Attendee]
    scheduled_meeting_ids: List[str] = field(default_factory=list)
    scheduled_details: List[Dict[str, Any]] = field(default_factory=list)
    step_count: int = 0


@dataclass
class CommandResult:
    """Immutable return value from a command handler.

    Attributes:
        action_valid: Whether the command was accepted.
        message: Human-readable outcome description.
        rejection_reason: Populated when action_valid is False.
        meeting_id: The meeting that was acted on, if applicable.
        utc_hour: UTC start hour of the booked/moved slot, if applicable.
        day: Day name of the booked/moved slot, if applicable.
    """

    action_valid: bool
    message: str
    rejection_reason: Optional[str] = None
    meeting_id: Optional[str] = None
    utc_hour: Optional[int] = None
    day: Optional[str] = None
