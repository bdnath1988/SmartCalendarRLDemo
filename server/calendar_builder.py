# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Factory helpers for constructing and pre-seeding the weekly calendar."""

from datetime import date, datetime, timedelta, timezone as dt_timezone
from typing import Dict, List, Tuple

try:
    from ..models import CalendarEvent, DayCalendar, Slot, WeeklyCalendar
    from ..task_definitions import THREE_ATTENDEES, TaskDifficulty
    from .slot_utils import SlotUtils
except ImportError:
    from models import CalendarEvent, DayCalendar, Slot, WeeklyCalendar
    from task_definitions import THREE_ATTENDEES, TaskDifficulty
    from server.slot_utils import SlotUtils


_WEEK_ORDER: List[str] = ["monday", "tuesday", "wednesday", "thursday", "friday"]

# Fixed background events pre-seeded per difficulty.
# Format: (day_name, utc_hour, event_title)
_OBSTACLES: Dict[TaskDifficulty, List[Tuple[str, int, str]]] = {
    TaskDifficulty.EASY: [
        ("monday", 9, "Standup call"),
        ("monday", 14, "Exec sync"),
    ],
    TaskDifficulty.MEDIUM: [
        ("monday", 9, "Standup call"),
        ("tuesday", 11, "Team lunch"),
        ("wednesday", 15, "1:1 review"),
    ],
    TaskDifficulty.HARD: [
        ("monday", 9, "Standup call"),
        ("tuesday", 11, "Team lunch"),
        ("wednesday", 15, "1:1 review"),
        ("thursday", 9, "Sprint planning"),
    ],
    TaskDifficulty.SUPER_HARD: [
        ("monday", 9, "Standup call"),
        ("tuesday", 11, "Team lunch"),
        ("wednesday", 15, "1:1 review"),
        ("thursday", 9, "Sprint planning"),
        ("friday", 16, "Retro"),
    ],
}


class CalendarBuilder:
    """Factory for constructing and pre-seeding the weekly calendar."""

    @staticmethod
    def get_week_dates() -> Dict[str, date]:
        """Return ISO dates for monday–friday of the current calendar week.

        Returns:
            Dict mapping day name to the corresponding date.
        """
        today = date.today()
        monday = today - timedelta(days=today.weekday())
        return {name: monday + timedelta(days=i) for i, name in enumerate(_WEEK_ORDER)}

    @staticmethod
    def build_weekly(week_dates: Dict[str, date]) -> WeeklyCalendar:
        """Create an empty 5-day × 10-slot calendar covering 08:00–18:00 UTC.

        Args:
            week_dates: Map of day name to ISO date (from get_week_dates).

        Returns:
            WeeklyCalendar with one empty DayCalendar per weekday.
        """
        days: Dict[str, DayCalendar] = {}
        for day_name in _WEEK_ORDER:
            d = week_dates[day_name]
            slots: List[Slot] = [
                Slot(
                    start_time=datetime(
                        d.year, d.month, d.day, hour, 0, 0, tzinfo=dt_timezone.utc
                    ).isoformat(),
                    end_time=datetime(
                        d.year, d.month, d.day, hour + 1, 0, 0, tzinfo=dt_timezone.utc
                    ).isoformat(),
                )
                for hour in range(8, 18)
            ]
            days[day_name] = DayCalendar(day_name=day_name, slots=slots)
        return WeeklyCalendar(days=days, attendees=list(THREE_ATTENDEES))

    @staticmethod
    def preseed_obstacles(week: WeeklyCalendar, difficulty: TaskDifficulty) -> None:
        """Block fixed slots with background commitments the agent must route around.

        These obstacles are never counted as task meetings. Their event IDs
        follow the pattern 'obstacle_N'.

        Args:
            week: Weekly calendar to mutate in place.
            difficulty: Determines which obstacle set to apply.
        """
        for idx, (day, utc_hour, title) in enumerate(
            _OBSTACLES.get(difficulty, [])
        ):
            slot = SlotUtils.find_in_day(week, day, utc_hour)
            if slot is not None:
                slot.event = CalendarEvent(event_id=f"obstacle_{idx}", title=title)
