# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for Round 2 models, timezone utilities, and curriculum definitions."""

import pytest
from datetime import datetime, date, timezone as dt_timezone, timedelta
from zoneinfo import ZoneInfo

from models import (
    Attendee,
    DayCalendar,
    WeeklyCalendar,
    Slot,
    utc_to_local,
    local_to_utc,
    is_within_preference,
)
from task_definitions import (
    TaskDifficulty,
    THREE_ATTENDEES,
    HACKATHON_CURRICULUM,
    FULL_CURRICULUM,
)


# ---- Helpers ----

def _make_day(day_name: str, date_str: str) -> DayCalendar:
    """Build a DayCalendar with 10 empty slots from 8am-6pm UTC."""
    base = datetime.fromisoformat(f"{date_str}T08:00:00")
    slots = [
        Slot(
            start_time=(base + timedelta(hours=i)).isoformat(),
            end_time=(base + timedelta(hours=i + 1)).isoformat(),
        )
        for i in range(10)
    ]
    return DayCalendar(day_name=day_name, slots=slots)


def _expected_local(utc_hour: int, tz: str) -> int:
    """Reference implementation for expected local hour, used in DST-sensitive assertions."""
    today = date.today()
    dt_utc = datetime(today.year, today.month, today.day, utc_hour, 0, 0, tzinfo=dt_timezone.utc)
    return dt_utc.astimezone(ZoneInfo(tz)).hour


# ---- Timezone conversion: utc_to_local ----

def test_utc_to_local_alice_ist():
    """IST is UTC+5:30 (no DST) — UTC 8 always maps to local hour 13 (13:30 truncated)."""
    assert utc_to_local(8, "Asia/Kolkata") == 13


def test_utc_to_local_bob_london():
    """London observes BST (+1) in summer and GMT (+0) in winter; validate against reference."""
    assert utc_to_local(8, "Europe/London") == _expected_local(8, "Europe/London")


def test_utc_to_local_carol_la():
    """Los Angeles observes PDT (−7) in summer and PST (−8) in winter; validate against reference."""
    assert utc_to_local(16, "America/Los_Angeles") == _expected_local(16, "America/Los_Angeles")


# ---- Timezone conversion: local_to_utc ----

def test_local_to_utc_alice_ist():
    """IST 10:00 = UTC 04:30 → truncated UTC hour 4."""
    assert local_to_utc(10, "Asia/Kolkata") == 4


def test_local_to_utc_roundtrip_bob():
    """local_to_utc(utc_to_local(h)) should return the original hour for whole-hour timezones."""
    utc_hour = 10
    local_hour = utc_to_local(utc_hour, "Europe/London")
    assert local_to_utc(local_hour, "Europe/London") == utc_hour


# ---- is_within_preference ----

def test_is_within_preference_alice_in_range():
    """UTC 9 → IST 14 (always, no DST). Alice prefers 10-16. 10 ≤ 14 < 16 → True."""
    alice = THREE_ATTENDEES[0]
    assert is_within_preference(9, alice) is True


def test_is_within_preference_alice_out_of_range_early():
    """UTC 3 → IST 8 (always). Alice prefers 10-16. 8 < 10 → False."""
    alice = THREE_ATTENDEES[0]
    assert is_within_preference(3, alice) is False


def test_is_within_preference_carol_out_of_range():
    """UTC 9 → PDT 2 or PST 1. Carol prefers 9-18. Both values < 9 → always False."""
    carol = THREE_ATTENDEES[2]
    assert is_within_preference(9, carol) is False


def test_is_within_preference_boundary_excluded():
    """preferred_end_hour is exclusive. UTC hour that maps to exactly preferred_end → False."""
    alice = THREE_ATTENDEES[0]
    # Alice preferred_end_hour=16 IST. UTC 10 → IST 15:30 → hour 15. Still True.
    # UTC 11 → IST 16:30 → hour 16. 16 < 16 is False.
    assert is_within_preference(11, alice) is False


# ---- WeeklyCalendar structure ----

def test_weekly_calendar_50_slots():
    """Five days × 10 slots each must equal exactly 50 total slots."""
    schedule = {
        "monday":    "2026-04-27",
        "tuesday":   "2026-04-28",
        "wednesday": "2026-04-29",
        "thursday":  "2026-04-30",
        "friday":    "2026-05-01",
    }
    days = {name: _make_day(name, date_str) for name, date_str in schedule.items()}
    wc = WeeklyCalendar(days=days, attendees=THREE_ATTENDEES)

    total = sum(len(d.slots) for d in wc.days.values())
    assert total == 50


def test_weekly_calendar_five_days():
    """WeeklyCalendar must contain exactly 5 day entries."""
    schedule = {
        "monday":    "2026-04-27",
        "tuesday":   "2026-04-28",
        "wednesday": "2026-04-29",
        "thursday":  "2026-04-30",
        "friday":    "2026-05-01",
    }
    days = {name: _make_day(name, date_str) for name, date_str in schedule.items()}
    wc = WeeklyCalendar(days=days, attendees=THREE_ATTENDEES)
    assert len(wc.days) == 5


def test_weekly_calendar_three_attendees():
    """WeeklyCalendar built with THREE_ATTENDEES must carry all three personas."""
    schedule = {"monday": "2026-04-27"}
    days = {"monday": _make_day("monday", "2026-04-27")}
    wc = WeeklyCalendar(days=days, attendees=THREE_ATTENDEES)
    names = {a.name for a in wc.attendees}
    assert names == {"Alice", "Bob", "Carol"}


# ---- Curriculum membership ----

def test_hackathon_curriculum_excludes_super_hard():
    """SUPER_HARD must NOT appear in HACKATHON_CURRICULUM."""
    assert TaskDifficulty.SUPER_HARD not in HACKATHON_CURRICULUM


def test_hackathon_curriculum_length():
    """HACKATHON_CURRICULUM must contain exactly 3 difficulty levels."""
    assert len(HACKATHON_CURRICULUM) == 3


def test_full_curriculum_contains_all_four():
    """FULL_CURRICULUM must contain all four TaskDifficulty values."""
    assert set(FULL_CURRICULUM) == set(TaskDifficulty)


def test_full_curriculum_length():
    """FULL_CURRICULUM must contain exactly 4 difficulty levels."""
    assert len(FULL_CURRICULUM) == 4
