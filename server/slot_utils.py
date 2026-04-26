# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Low-level helpers for slot lookup and time parsing."""

from datetime import datetime, timezone as dt_timezone
from typing import Optional

try:
    from ..models import Slot, WeeklyCalendar
except ImportError:
    from models import Slot, WeeklyCalendar


class SlotUtils:
    """Static utilities for calendar slot operations and time conversion."""

    @staticmethod
    def find_in_day(week: WeeklyCalendar, day: str, utc_hour: int) -> Optional[Slot]:
        """Return the Slot at utc_hour on day, or None if not found.

        Args:
            week: Weekly calendar to search.
            day: Day name, e.g. 'monday'.
            utc_hour: Target UTC hour (0–23).

        Returns:
            Matching Slot, or None if day is absent or hour is out of range.
        """
        day_cal = week.days.get(day)
        if day_cal is None:
            return None
        for slot in day_cal.slots:
            if SlotUtils.parse_utc(slot.start_time).hour == utc_hour:
                return slot
        return None

    @staticmethod
    def find_by_meeting_id(week: WeeklyCalendar, meeting_id: str) -> Optional[Slot]:
        """Search all days for a booked slot whose event_id matches meeting_id.

        Args:
            week: Weekly calendar to search.
            meeting_id: Event ID to match.

        Returns:
            Matching Slot, or None.
        """
        for day_cal in week.days.values():
            for slot in day_cal.slots:
                if slot.event and slot.event.event_id == meeting_id:
                    return slot
        return None

    @staticmethod
    def to_utc_hour(time_str: str) -> int:
        """Parse an ISO-8601 or HH:MM string and return the UTC hour.

        Accepts full ISO datetimes (with or without tzinfo) and plain
        HH:MM / HH:MM:SS shorthand strings (treated as UTC).

        Args:
            time_str: Time string from an agent action slot.

        Returns:
            UTC hour as int.

        Raises:
            ValueError: If the string cannot be parsed.
        """
        if not isinstance(time_str, str):
            raise ValueError(f"expected str, got {type(time_str)}")
        if len(time_str) <= 8 and "T" not in time_str:
            return int(time_str.split(":")[0])
        dt = datetime.fromisoformat(time_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=dt_timezone.utc)
        return dt.astimezone(dt_timezone.utc).hour

    @staticmethod
    def parse_utc(t: str) -> datetime:
        """Parse an ISO datetime string, assuming UTC when no tzinfo is present.

        Args:
            t: ISO datetime string stored in a Slot.

        Returns:
            Timezone-aware datetime in UTC.
        """
        dt = datetime.fromisoformat(t)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=dt_timezone.utc)
        return dt.astimezone(dt_timezone.utc)
