# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Formal task definitions with increasing difficulty for evaluation progression.
"""

from enum import Enum
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal

try:
    from models import Attendee
except ImportError:
    from ..models import Attendee


# ---- Existing models (kept for backward compatibility with server/ and grader) ----

class TaskDifficultyMetrics(BaseModel):
    """Metrics defining the difficulty of a specific task."""
    target_meetings: int = Field(description="Number of meetings to schedule")
    allow_soft_conflicts: bool = Field(description="Whether soft conflicts/rescheduling are allowed")
    enforce_spacing: bool = Field(description="Whether to enforce gaps between meetings")
    min_gap_hours: float = Field(default=0.0, description="Minimum gap between meetings in hours")

class CalendarTask(BaseModel):
    """Defines a formal evaluation task for the Smart Calendar Agent."""
    task_id: str = Field(description="Unique string identifier for the task, e.g., 'easy-01'")
    level: str = Field(description="Difficulty level: 'easy', 'medium', or 'hard'")
    objective: str = Field(description="Full text instruction provided to the agent")
    goal: str = Field(description="Short representation of the goal")
    metrics: TaskDifficultyMetrics = Field(description="Quantitative metrics for grading the task")


# Pre-defined Formal Progression Tasks
EVALUATION_TASKS = [
    CalendarTask(
        task_id="task-easy-1",
        level="easy",
        objective="Task 1 (Easy): Schedule exactly 1 meeting at any available time.",
        goal="schedule 1 meeting",
        metrics=TaskDifficultyMetrics(
            target_meetings=1,
            allow_soft_conflicts=True,
            enforce_spacing=False
        )
    ),
    CalendarTask(
        task_id="task-medium-1",
        level="medium",
        objective="Task 2 (Medium): Schedule 3 distinct meetings without any overlaps.",
        goal="schedule 3 distinct meetings",
        metrics=TaskDifficultyMetrics(
            target_meetings=3,
            allow_soft_conflicts=False,
            enforce_spacing=False
        )
    ),
    CalendarTask(
        task_id="task-hard-1",
        level="hard",
        objective="Task 3 (Hard): Schedule 5 meetings with strict constraints. You must leave at least a 1-hour gap between any two meetings.",
        goal="schedule 5 meetings with 1 hour spacing",
        metrics=TaskDifficultyMetrics(
            target_meetings=5,
            allow_soft_conflicts=False,
            enforce_spacing=True,
            min_gap_hours=1.0
        )
    )
]

def get_task_by_level(level: str) -> CalendarTask:
    for task in EVALUATION_TASKS:
        if task.level == level:
            return task
    return EVALUATION_TASKS[0]


# ---- Round 2: TaskDifficulty ----

class TaskDifficulty(Enum):
    """Curriculum difficulty levels. SUPER_HARD is research-only."""
    EASY       = "easy"
    MEDIUM     = "medium"
    HARD       = "hard"
    SUPER_HARD = "super_hard"


# ---- Round 2: TaskSpec ----

class TaskSpec(BaseModel):
    """Full specification for a Round 2 multi-attendee scheduling task."""

    difficulty: TaskDifficulty = Field(description="Curriculum difficulty level")
    meetings: List[str] = Field(description="Ordered list of meeting IDs to schedule")
    days: List[str] = Field(description="Work days available for scheduling")
    attendee_names: List[str] = Field(description="Names of attendees required for this task")
    max_steps: int = Field(description="Maximum agent steps allowed per episode")
    reward_mode: Literal["dense", "sparse"] = Field(description="'dense' gives step-level signals; 'sparse' gives 0 until complete")
    research_mode: bool = Field(default=False, description="If True, excluded from HACKATHON_CURRICULUM")


# ---- Attendee Personas (fixed, as per CLAUDE.md) ----

THREE_ATTENDEES: List[Attendee] = [
    Attendee(
        name="Alice",
        timezone="Asia/Kolkata",
        priority=1,
        preferred_start_hour=10,
        preferred_end_hour=16,
    ),
    Attendee(
        name="Bob",
        timezone="Europe/London",
        priority=2,
        preferred_start_hour=9,
        preferred_end_hour=17,
    ),
    Attendee(
        name="Carol",
        timezone="America/Los_Angeles",
        priority=3,
        preferred_start_hour=9,
        preferred_end_hour=18,
    ),
]


# ---- Full Dependency Graph ----
# Each entry: {"deps": [meeting_ids that must be scheduled first], "attendees": [name list]}

FULL_DEPENDENCY_GRAPH: Dict[str, Dict[str, Any]] = {
    "kickoff":         {"deps": [],                                      "attendees": ["Alice"]},
    "standup_mon":     {"deps": [],                                      "attendees": ["Alice", "Bob", "Carol"]},
    "standup_tue":     {"deps": ["standup_mon"],                         "attendees": ["Alice", "Bob", "Carol"]},
    "standup_wed":     {"deps": ["standup_tue"],                         "attendees": ["Alice", "Bob", "Carol"]},
    "standup_thu":     {"deps": ["standup_wed"],                         "attendees": ["Alice", "Bob", "Carol"]},
    "requirements":    {"deps": ["kickoff"],                             "attendees": ["Alice", "Bob", "Carol"]},
    "backend_design":  {"deps": ["requirements"],                        "attendees": ["Bob"]},
    "frontend_design": {"deps": ["requirements"],                        "attendees": ["Carol"]},
    "integration":     {"deps": ["backend_design", "frontend_design"],   "attendees": ["Bob", "Carol"]},
    "qa_planning":     {"deps": ["integration"],                         "attendees": ["Bob", "Carol"]},
    "launch_review":   {"deps": ["qa_planning"],                         "attendees": ["Alice", "Bob", "Carol"]},
}


# ---- Task Spec Instances ----

EASY_SPEC = TaskSpec(
    difficulty=TaskDifficulty.EASY,
    meetings=["kickoff"],
    days=["monday"],
    attendee_names=["Alice"],
    max_steps=10,
    reward_mode="dense",
)

MEDIUM_SPEC = TaskSpec(
    difficulty=TaskDifficulty.MEDIUM,
    meetings=["kickoff", "requirements", "backend_design"],
    days=["monday", "tuesday", "wednesday"],
    attendee_names=["Alice", "Bob"],
    max_steps=20,
    reward_mode="dense",
)

HARD_SPEC = TaskSpec(
    difficulty=TaskDifficulty.HARD,
    meetings=[
        "kickoff",
        "requirements",
        "backend_design",
        "frontend_design",
        "integration",
        "qa_planning",
        "launch_review",
    ],
    days=["monday", "tuesday", "wednesday", "thursday", "friday"],
    attendee_names=["Alice", "Bob", "Carol"],
    max_steps=30,
    reward_mode="dense",
)

SUPER_HARD_SPEC = TaskSpec(
    difficulty=TaskDifficulty.SUPER_HARD,
    meetings=[
        "kickoff",
        "standup_mon",
        "standup_tue",
        "standup_wed",
        "standup_thu",
        "requirements",
        "backend_design",
        "frontend_design",
        "integration",
        "qa_planning",
        "launch_review",
    ],
    days=["monday", "tuesday", "wednesday", "thursday", "friday"],
    attendee_names=["Alice", "Bob", "Carol"],
    max_steps=40,
    reward_mode="sparse",
    research_mode=True,
)


# ---- Curricula ----

HACKATHON_CURRICULUM: List[TaskDifficulty] = [
    TaskDifficulty.EASY,
    TaskDifficulty.MEDIUM,
    TaskDifficulty.HARD,
]

FULL_CURRICULUM: List[TaskDifficulty] = [
    TaskDifficulty.EASY,
    TaskDifficulty.MEDIUM,
    TaskDifficulty.HARD,
    TaskDifficulty.SUPER_HARD,
]
