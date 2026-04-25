# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the Round-2 CalendarEnv environment.

Covers: reset per difficulty, all 5 rewards on valid add_event,
dependency violation, slot conflict, duplicate action, max-steps
termination, and timezone preference validation.
"""

import pytest
from datetime import datetime, timezone

from models import Attendee, ExpectedAction, PerformedAction, Slot, MyCalendarAction
from models import is_within_preference
from task_definitions import TaskDifficulty
from server.smart_calendar_agent_environment import CalendarEnv


# ── helpers ──────────────────────────────────────────────────────────────────

def make_add_action(meeting_id: str, day: str, utc_hour: int, env: CalendarEnv) -> MyCalendarAction:
    """Build a well-formed add_event action for a given UTC hour.

    Args:
        meeting_id: Meeting to schedule.
        day: Day name, e.g. 'monday'.
        utc_hour: Target UTC start hour.
        env: Live environment (provides week dates for the ISO timestamp).

    Returns:
        MyCalendarAction ready to pass to env.step().
    """
    assert env._state is not None
    d = env._state.week.days[day].slots[0]
    # Derive the date from the first slot of that day
    base_dt = datetime.fromisoformat(d.start_time)
    start = datetime(
        base_dt.year, base_dt.month, base_dt.day, utc_hour, 0, 0, tzinfo=timezone.utc
    ).isoformat()
    end = datetime(
        base_dt.year, base_dt.month, base_dt.day, utc_hour + 1, 0, 0, tzinfo=timezone.utc
    ).isoformat()
    return MyCalendarAction(
        expected_action=ExpectedAction(
            command="add_event",
            event_id=meeting_id,
            day=day,
            slot=Slot(start_time=start, end_time=end),
        ),
        performed_action=PerformedAction(success=True, event_id=meeting_id),
    )


def make_search_action(day: str) -> MyCalendarAction:
    """Build a search_slot action.

    Args:
        day: Day to search.

    Returns:
        MyCalendarAction with command='search_slot'.
    """
    return MyCalendarAction(
        expected_action=ExpectedAction(command="search_slot", day=day),
        performed_action=PerformedAction(success=True),
    )


# ── reset tests ───────────────────────────────────────────────────────────────

class TestReset:
    """Verify reset() produces correct initial state per difficulty."""

    def test_reset_easy(self):
        """EASY: 1 attendee (Alice), 1 day (monday), 1 target meeting."""
        env = CalendarEnv()
        obs = env.reset(TaskDifficulty.EASY)

        meta = obs.metadata
        assert meta["target_meetings"] == 1
        assert meta["max_steps"] == 10
        assert list(meta["free_slots"].keys()) == ["monday"]
        assert "Alice" in meta["free_slots"]["monday"]
        assert len(meta["attendees"]) == 1
        assert meta["attendees"][0]["name"] == "Alice"
        assert meta["scheduled_meetings"] == []
        assert "kickoff" in meta["dependency_graph"]

    def test_reset_medium(self):
        """MEDIUM: 2 attendees, 3 days, 3 target meetings."""
        env = CalendarEnv()
        obs = env.reset(TaskDifficulty.MEDIUM)

        meta = obs.metadata
        assert meta["target_meetings"] == 3
        assert meta["max_steps"] == 20
        assert set(meta["free_slots"].keys()) == {"monday", "tuesday", "wednesday"}
        attendee_names = {a["name"] for a in meta["attendees"]}
        assert attendee_names == {"Alice", "Bob"}
        assert set(meta["dependency_graph"].keys()) == {
            "kickoff", "requirements", "backend_design"
        }

    def test_reset_hard(self):
        """HARD: 3 attendees, 5 days, 7 target meetings with dependency chain."""
        env = CalendarEnv()
        obs = env.reset(TaskDifficulty.HARD)

        meta = obs.metadata
        assert meta["target_meetings"] == 7
        assert meta["max_steps"] == 30
        assert set(meta["free_slots"].keys()) == {
            "monday", "tuesday", "wednesday", "thursday", "friday"
        }
        attendee_names = {a["name"] for a in meta["attendees"]}
        assert attendee_names == {"Alice", "Bob", "Carol"}
        assert "launch_review" in meta["dependency_graph"]

    def test_reset_obstacles_block_slots(self):
        """Pre-seeded obstacles should appear as occupied (not free) for Alice."""
        env = CalendarEnv()
        env.reset(TaskDifficulty.EASY)

        # EASY obstacles: monday 09:00 and 14:00 UTC
        # Alice is IST (+5:30): 09:00 UTC = 14:30 IST, 14:00 UTC = 19:30 IST
        # Neither should appear in Alice's free_slots
        free = env._state.week.days["monday"].slots
        occupied_hours = {
            datetime.fromisoformat(s.start_time).hour
            for s in free
            if s.event is not None
        }
        assert 9 in occupied_hours
        assert 14 in occupied_hours


# ── step: valid add_event ─────────────────────────────────────────────────────

class TestStepAddEvent:
    """Verify all 5 reward signals are present on a valid add_event."""

    def test_all_five_rewards_returned(self):
        """Valid kickoff scheduling should return all five named reward keys."""
        env = CalendarEnv()
        env.reset(TaskDifficulty.EASY)
        action = make_add_action("kickoff", "monday", 10, env)
        obs = env.step(action)

        assert obs.metadata["action_valid"] is True
        assert obs.metadata["rejection_reason"] is None
        for key in (
            "reward_conflict_free",
            "reward_dependency_respected",
            "reward_attendee_satisfied",
            "reward_efficiency",
            "reward_objective_progress",
        ):
            assert key in obs.metadata, f"missing reward key: {key}"

    def test_objective_progress_increments(self):
        """After scheduling kickoff, objective_progress should be 1.0 (EASY target=1)."""
        env = CalendarEnv()
        env.reset(TaskDifficulty.EASY)
        action = make_add_action("kickoff", "monday", 10, env)
        obs = env.step(action)

        assert obs.metadata["reward_objective_progress"] == pytest.approx(1.0)
        assert obs.done is True

    def test_conflict_free_is_one_after_valid_booking(self):
        """No double-booking means conflict_free should be 1.0."""
        env = CalendarEnv()
        env.reset(TaskDifficulty.EASY)
        obs = env.step(make_add_action("kickoff", "monday", 10, env))

        assert obs.metadata["reward_conflict_free"] == pytest.approx(1.0)

    def test_top_level_reward_is_always_zero(self):
        """Top-level reward must stay 0.0 — signals are never composited."""
        env = CalendarEnv()
        env.reset(TaskDifficulty.EASY)
        obs = env.step(make_add_action("kickoff", "monday", 10, env))

        assert obs.reward == pytest.approx(0.0)


# ── step: dependency violation ────────────────────────────────────────────────

class TestDependencyViolation:
    """Attempting to schedule a meeting before its dependency must be rejected."""

    def test_requirements_before_kickoff_rejected(self):
        """requirements depends on kickoff; scheduling it first must fail."""
        env = CalendarEnv()
        env.reset(TaskDifficulty.MEDIUM)
        # Try to schedule requirements before kickoff
        action = make_add_action("requirements", "tuesday", 10, env)
        obs = env.step(action)

        assert obs.metadata["action_valid"] is False
        assert "dependency" in (obs.metadata["rejection_reason"] or "")
        assert obs.metadata["reward_dependency_respected"] == pytest.approx(0.0)

    def test_kickoff_has_no_deps_and_succeeds(self):
        """kickoff has no dependencies and must always be schedulable."""
        env = CalendarEnv()
        env.reset(TaskDifficulty.MEDIUM)
        obs = env.step(make_add_action("kickoff", "monday", 10, env))

        assert obs.metadata["action_valid"] is True


# ── step: slot conflict ───────────────────────────────────────────────────────

class TestSlotConflict:
    """Scheduling into an occupied slot must be rejected."""

    def test_obstacle_slot_conflict(self):
        """monday 09:00 UTC is pre-seeded as 'Standup call' in EASY."""
        env = CalendarEnv()
        env.reset(TaskDifficulty.EASY)
        action = make_add_action("kickoff", "monday", 9, env)
        obs = env.step(action)

        assert obs.metadata["action_valid"] is False
        assert obs.metadata["rejection_reason"] == "slot_conflict"
        assert obs.metadata["reward_conflict_free"] == pytest.approx(0.0)

    def test_double_booking_same_slot_rejected(self):
        """Scheduling two different meetings to the same slot must reject the second."""
        env = CalendarEnv()
        env.reset(TaskDifficulty.MEDIUM)
        # Schedule kickoff at monday 10:00
        env.step(make_add_action("kickoff", "monday", 10, env))
        # requirements needs kickoff first — schedule it
        env.step(make_add_action("requirements", "monday", 11, env))
        # Now try to put backend_design (dep: requirements) on the same slot as requirements
        obs = env.step(make_add_action("backend_design", "monday", 11, env))

        assert obs.metadata["action_valid"] is False
        assert obs.metadata["rejection_reason"] == "slot_conflict"


# ── step: duplicate action ────────────────────────────────────────────────────

class TestDuplicateAction:
    """Sending the exact same action twice must zero all rewards on the repeat."""

    def test_duplicate_search_slot_zeroes_rewards(self):
        """Second identical search_slot must return rejection_reason=duplicate_action."""
        env = CalendarEnv()
        env.reset(TaskDifficulty.EASY)
        action = make_search_action("monday")

        env.step(action)  # first — valid
        obs2 = env.step(action)  # duplicate

        assert obs2.metadata["action_valid"] is False
        assert obs2.metadata["rejection_reason"] == "duplicate_action"
        for key in (
            "reward_conflict_free",
            "reward_dependency_respected",
            "reward_attendee_satisfied",
            "reward_efficiency",
            "reward_objective_progress",
        ):
            assert obs2.metadata[key] == pytest.approx(0.0)


# ── max steps termination ─────────────────────────────────────────────────────

class TestMaxSteps:
    """Episode must terminate (done=True) when the step budget is exhausted."""

    def test_done_after_max_steps(self):
        """EASY has max_steps=10; after 10 steps done must be True."""
        env = CalendarEnv()
        env.reset(TaskDifficulty.EASY)
        action = make_search_action("monday")

        obs = None
        for _ in range(10):
            obs = env.step(action)

        assert obs is not None
        assert obs.done is True
        assert env._state.step_count == 10  # type: ignore[union-attr]

    def test_step_after_limit_returns_terminated_message(self):
        """Step 11 on EASY must immediately return the termination message."""
        env = CalendarEnv()
        env.reset(TaskDifficulty.EASY)
        for _ in range(10):
            env.step(make_search_action("monday"))

        obs = env.step(make_search_action("monday"))  # step 11
        assert obs.done is True
        assert obs.metadata["rejection_reason"] == "max_steps_exceeded"


# ── timezone preference ───────────────────────────────────────────────────────

class TestTimezonePreference:
    """Validate that UTC→local conversion and preference checks are correct."""

    def test_alice_8am_utc_is_satisfied(self):
        """8am UTC = 1:30pm IST, which is within Alice's 10am–4pm preference."""
        alice = Attendee(
            name="Alice",
            timezone="Asia/Kolkata",
            priority=1,
            preferred_start_hour=10,
            preferred_end_hour=16,
        )
        assert is_within_preference(8, alice) is True

    def test_alice_3am_utc_is_not_satisfied(self):
        """3am UTC = 8:30am IST, which is before Alice's 10am start."""
        alice = Attendee(
            name="Alice",
            timezone="Asia/Kolkata",
            priority=1,
            preferred_start_hour=10,
            preferred_end_hour=16,
        )
        assert is_within_preference(3, alice) is False

    def test_attendee_satisfied_reward_reflects_preference(self):
        """Scheduling kickoff (Alice only) at 10am UTC should give satisfied=1.0."""
        # 10am UTC = 3:30pm IST — within Alice's 10am–4pm window
        env = CalendarEnv()
        env.reset(TaskDifficulty.EASY)
        obs = env.step(make_add_action("kickoff", "monday", 10, env))

        assert obs.metadata["action_valid"] is True
        assert obs.metadata["reward_attendee_satisfied"] == pytest.approx(1.0)
