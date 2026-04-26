# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Schema and dependency validation for incoming agent actions."""

from typing import Tuple

try:
    from ..models import MyCalendarAction
    from ..task_definitions import FULL_DEPENDENCY_GRAPH
    from .state import EpisodeState
except ImportError:
    from models import MyCalendarAction
    from task_definitions import FULL_DEPENDENCY_GRAPH
    from server.state import EpisodeState

_VALID_COMMANDS = frozenset({"add_event", "move_event", "reschedule_event", "search_slot", "delete_event"})


class ActionValidator:
    """Static validators for agent action correctness."""

    @staticmethod
    def schema(action: MyCalendarAction) -> Tuple[bool, str]:
        """Check that the action carries required fields for its command.

        Extra or unknown fields are silently ignored; only missing required
        fields and unknown commands are rejected.

        Args:
            action: Incoming agent action.

        Returns:
            (True, '') if valid; (False, reason_string) otherwise.
        """
        try:
            cmd = action.expected_action.command
        except AttributeError:
            return False, "missing expected_action.command"

        if cmd not in _VALID_COMMANDS:
            return False, f"unknown command '{cmd}'"

        exp = action.expected_action
        if cmd in {"add_event", "reschedule_event"}:
            if not exp.event_id:
                return False, "add_event requires event_id"
            if not exp.day:
                return False, "add_event requires day"
            if not exp.slot:
                return False, "add_event requires slot"
        if cmd == "move_event":
            if not exp.event_id:
                return False, "move_event requires event_id"
            if not exp.day:
                return False, "move_event requires day"
            if not exp.slot:
                return False, "move_event requires slot"
        if cmd == "search_slot":
            if not exp.day:
                return False, "search_slot requires day"
        if cmd == "delete_event":
            if not exp.event_id:
                return False, "delete_event requires event_id"
        return True, ""

    @staticmethod
    def dependencies(meeting_id: str, state: EpisodeState) -> Tuple[bool, str]:
        """Confirm all in-spec dependencies of meeting_id are already scheduled.

        Only checks dependencies that are present in the active task spec.
        Out-of-spec prerequisites (from other difficulty levels) are ignored.

        Args:
            meeting_id: Meeting about to be added.
            state: Current episode state.

        Returns:
            (True, '') if satisfied; (False, reason_string) otherwise.
        """
        deps = FULL_DEPENDENCY_GRAPH.get(meeting_id, {}).get("deps", [])
        for dep in deps:
            if dep in state.spec.meetings and dep not in state.scheduled_meeting_ids:
                return (
                    False,
                    f"dependency '{dep}' must be scheduled before '{meeting_id}'",
                )
        return True, ""
