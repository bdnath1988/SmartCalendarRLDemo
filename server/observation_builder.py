# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Constructs agent-visible metadata from episode state.

The raw UTC slot matrix is never included — only local-time strings and
dependency status labels are surfaced to the agent.
"""

from typing import Any, Dict, List
from zoneinfo import ZoneInfo

try:
    from ..models import Attendee, WeeklyCalendar
    from ..task_definitions import FULL_DEPENDENCY_GRAPH, TaskSpec
    from .slot_utils import SlotUtils
except ImportError:
    from models import Attendee, WeeklyCalendar
    from task_definitions import FULL_DEPENDENCY_GRAPH, TaskSpec
    from server.slot_utils import SlotUtils


class ObservationBuilder:
    """Static helpers that build agent-visible metadata from the calendar state."""

    @staticmethod
    def free_slots(
        week: WeeklyCalendar,
        spec: TaskSpec,
        attendees: List[Attendee],
    ) -> Dict[str, Dict[str, List[str]]]:
        """Return free-slot strings per day per attendee in each attendee's local timezone.

        Only days listed in spec.days are included. Each slot appears as a
        'HH:MM-HH:MM' string in the attendee's IANA timezone — the raw UTC
        slot matrix is never exposed.

        Args:
            week: Current weekly calendar.
            spec: Active task spec (determines which days to include).
            attendees: Attendees whose timezones to convert to.

        Returns:
            {day_name: {attendee_name: ["HH:MM-HH:MM", ...]}}.
        """
        result: Dict[str, Dict[str, List[str]]] = {}
        for day_name in spec.days:
            day_cal = week.days.get(day_name)
            if day_cal is None:
                continue
            result[day_name] = {}
            for attendee in attendees:
                local_slots: List[str] = []
                for slot in day_cal.slots:
                    if slot.event is not None:
                        continue
                    start_dt = SlotUtils.parse_utc(slot.start_time)
                    end_dt = SlotUtils.parse_utc(slot.end_time)
                    local_start = start_dt.astimezone(ZoneInfo(attendee.timezone))
                    local_end = end_dt.astimezone(ZoneInfo(attendee.timezone))
                    local_slots.append(
                        f"{local_start.strftime('%H:%M')}-{local_end.strftime('%H:%M')}"
                    )
                result[day_name][attendee.name] = local_slots
        return result

    @staticmethod
    def dependency_graph(
        spec: TaskSpec,
        scheduled: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """Build the dependency graph restricted to spec meetings with status labels.

        Status values:
            'scheduled' — already booked this episode.
            'available' — all deps satisfied, ready to schedule.
            'locked'    — at least one dep still unscheduled.

        Args:
            spec: Active task spec.
            scheduled: Meeting IDs already scheduled this episode.

        Returns:
            {meeting_id: {"deps": [...], "status": "scheduled|available|locked"}}.
        """
        graph: Dict[str, Dict[str, Any]] = {}
        for mid in spec.meetings:
            raw_deps = FULL_DEPENDENCY_GRAPH.get(mid, {}).get("deps", [])
            relevant = [d for d in raw_deps if d in spec.meetings]
            if mid in scheduled:
                status = "scheduled"
            elif all(d in scheduled for d in relevant):
                status = "available"
            else:
                status = "locked"
            graph[mid] = {"deps": relevant, "status": status}
        return graph
