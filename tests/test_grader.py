import pytest
from datetime import datetime, timedelta
from models import Calendar, Slot, CalendarEvent
from server.grader import Grader
from task_definitions import CalendarTask, TaskDifficultyMetrics

# Dummy Tasks
easy_task = CalendarTask(
    task_id="t1", level="easy", objective="easy obj", goal="easy goal",
    metrics=TaskDifficultyMetrics(target_meetings=1, allow_soft_conflicts=True, enforce_spacing=False)
)

medium_task = CalendarTask(
    task_id="t2", level="medium", objective="med obj", goal="med goal",
    metrics=TaskDifficultyMetrics(target_meetings=3, allow_soft_conflicts=False, enforce_spacing=False)
)

hard_task = CalendarTask(
    task_id="t3", level="hard", objective="hard obj", goal="hard goal",
    metrics=TaskDifficultyMetrics(target_meetings=5, allow_soft_conflicts=False, enforce_spacing=True, min_gap_hours=1.0)
)

def build_slot(idx: int, start: str, end: str, has_event: bool = True) -> Slot:
    event = CalendarEvent(event_id=str(idx), title=f"Event {idx}") if has_event else None
    return Slot(start_time=start, end_time=end, event=event)

def test_grader_easy_success():
    """Test 1: Easy task requires only 1 meeting. Ensures binary scoring."""
    cal = Calendar(slots=[
        build_slot(1, "2026-04-11T09:00:00", "2026-04-11T10:00:00")
    ])
    score = Grader.compute_score(cal, easy_task)
    assert score == 1.0

def test_grader_hard_fail_overlaps():
    """Test 2: Determinism & Edge Case - Overlapping events should result in 0.0 score."""
    cal = Calendar(slots=[
        build_slot(1, "2026-04-11T09:00:00", "2026-04-11T10:00:00"),
        # Overlaps perfectly
        build_slot(2, "2026-04-11T09:30:00", "2026-04-11T10:30:00")
    ])
    score = Grader.compute_score(cal, medium_task)
    assert score == 0.0

def test_grader_medium_proportional():
    """Test 3: Medium task scoring should be directly proportional to scheduled count."""
    cal = Calendar(slots=[
        build_slot(1, "2026-04-11T09:00:00", "2026-04-11T10:00:00"),
        build_slot(2, "2026-04-11T11:00:00", "2026-04-11T12:00:00")
    ])
    # Target is 3, we have 2, no overlaps -> 2/3 = 0.666...
    score = Grader.compute_score(cal, medium_task)
    assert abs(score - 0.666) < 0.01

def test_grader_hard_spacing_violations():
    """Test 4: Strict boundary condition - events without required 1-hour gap."""
    cal = Calendar(slots=[
        build_slot(1, "2026-04-11T09:00:00", "2026-04-11T10:00:00"),
        # Gap is only 30m, which violates the 1-hour hard constraint
        build_slot(2, "2026-04-11T10:30:00", "2026-04-11T11:30:00"),
        # Valid gap 1 hr
        build_slot(3, "2026-04-11T12:30:00", "2026-04-11T13:30:00"),
    ])
    
    # 3/5 completion = 0.6. Spacing = 1 violation out of max 0.4 score
    score = Grader.compute_score(cal, hard_task)
    # Base completion: 0.6 * (3/5) = 0.36
    # Gap score: max(0.0, 0.4 - (0.1 * 1)) = 0.3
    # Total = 0.660
    assert abs(score - 0.660) < 0.001

def test_grader_hard_perfect():
    """Test 5: Edge case handling bounded exactly to 1.0 when perfectly executed."""
    cal = Calendar(slots=[
        build_slot(1, "2026-04-11T08:00:00", "2026-04-11T09:00:00"),
        build_slot(2, "2026-04-11T10:00:00", "2026-04-11T11:00:00"),
        build_slot(3, "2026-04-11T12:00:00", "2026-04-11T13:00:00"),
        build_slot(4, "2026-04-11T14:00:00", "2026-04-11T15:00:00"),
        build_slot(5, "2026-04-11T16:00:00", "2026-04-11T17:00:00")
    ])
    score = Grader.compute_score(cal, hard_task)
    assert score == 1.0
