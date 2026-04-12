from datetime import datetime
from typing import List, Tuple
from models import Calendar, Slot
from task_definitions import CalendarTask

class Grader:
    """
    Robust grader for evaluating the final calendar state against formal task metrics.
    Ensures deterministic scoring and strict penalty checks for invalid states.
    """

    @staticmethod
    def _parse_time(t) -> datetime:
        if isinstance(t, datetime):
            return t
        if isinstance(t, str):
            return datetime.fromisoformat(t)
        raise ValueError(f"Unparseable time format: {t}")

    @classmethod
    def check_overlaps(cls, occupied_slots: List[Slot]) -> int:
        """Returns the number of distinct pairwise overlaps."""
        overlaps = 0
        sorted_slots = sorted(occupied_slots, key=lambda s: cls._parse_time(s.start_time))
        for i in range(len(sorted_slots)):
            for j in range(i + 1, len(sorted_slots)):
                s1, s2 = sorted_slots[i], sorted_slots[j]
                s1_start, s1_end = cls._parse_time(s1.start_time), cls._parse_time(s1.end_time)
                s2_start, s2_end = cls._parse_time(s2.start_time), cls._parse_time(s2.end_time)
                
                # Check strict overlap (exclusive ends)
                if max(s1_start, s2_start) < min(s1_end, s2_end):
                    overlaps += 1
        return overlaps
    
    @classmethod
    def evaluate_spacing(cls, occupied_slots: List[Slot], min_gap_hours: float) -> Tuple[bool, int]:
        """Check if all adjacent events respect the minimum gap. returns (all_respected, number_of_violations)"""
        sorted_slots = sorted(occupied_slots, key=lambda s: cls._parse_time(s.start_time))
        violations = 0
        
        for i in range(len(sorted_slots) - 1):
            left, right = sorted_slots[i], sorted_slots[i+1]
            left_end = cls._parse_time(left.end_time)
            right_start = cls._parse_time(right.start_time)
            
            gap_hours = (right_start - left_end).total_seconds() / 3600.0
            if gap_hours < min_gap_hours:
                violations += 1
                
        return violations == 0, violations

    @classmethod
    def compute_score(cls, calendar: Calendar, task: CalendarTask) -> float:
        """
        Computes a deterministic score in [0.0, 1.0] for the calendar state.
        Penalizes for overlapping events, missing target counts, and spacing rule violations.
        """
        score = 0.0
        occupied_slots = [slot for slot in calendar.slots if slot.event is not None]
        scheduled_count = len(occupied_slots)
        
        metrics = task.metrics
        
        # Base completion calculation (capped at 1.0)
        completion_ratio = min(1.0, scheduled_count / max(1, metrics.target_meetings))
        
        # Severe penalty for any overlapping events
        overlaps = cls.check_overlaps(occupied_slots)
        if overlaps > 0:
            return 0.01  # Overlapping events is a hard fail for calendar validity in this environment (clamped > 0.0)
            
        if task.level == "easy":
            # Binary completion
            score = 0.99 if scheduled_count >= metrics.target_meetings else 0.01
            return score
            
        elif task.level == "medium":
            # Just proportional to scheduled count, assuming no overlaps
            score = float(completion_ratio)
            
        elif task.level == "hard":
            # Needs strict spacing
            score = 0.6 * completion_ratio
            all_respected, violations = cls.evaluate_spacing(occupied_slots, metrics.min_gap_hours)
            
            # Spacing gives max 0.4 score
            if scheduled_count >= 2:
                # Max gap score if all respected
                gap_score = 0.4 if all_respected else max(0.0, 0.4 - (0.1 * violations))
                score += gap_score
            elif scheduled_count == 1:
                # If only 1 scheduled, they get no gap points
                pass
                
        # Clamp to strictly between (0, 1) to pass criteria
        return max(0.01, min(0.99, round(score, 3)))
