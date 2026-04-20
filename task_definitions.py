# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Formal task definitions with increasing difficulty for evaluation progression.
"""

from pydantic import BaseModel, Field
from typing import List

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
