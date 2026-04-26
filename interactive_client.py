from datetime import datetime

from client import SmartCalendarEnv
from models import ExpectedAction, MyCalendarAction, PerformedAction, Slot


def parse_time(t):
    today = datetime.now().date()
    return datetime.fromisoformat(f"{today.isoformat()}T{t}:00")


def build_slot(start, end):
    return Slot(start_time=parse_time(start), end_time=parse_time(end))


def print_result(res):
    print(f"-> {res.observation.message} | Reward: {res.reward} | Done: {res.done}")


with SmartCalendarEnv(base_url="http://localhost:8000").sync() as env:
    print("\nSmart Calendar CLI Started")
    print("Type 'help' for commands\n")

    while True:
        cmd = input(">> ").strip().split()

        if not cmd:
            continue

        if cmd[0] == "exit":
            print("Exiting...")
            break

        elif cmd[0] == "help":
            print(
                """
Commands:
  reset
  add <id> <title> <start> <end>
  move <id> <start> <end>
  delete <id>
  state
  exit
            """
            )

        elif cmd[0] == "reset":
            res = env.reset()
            print(f"Reset: {res.observation.message}")

        elif cmd[0] == "add":
            try:
                _, id, title, start, end = cmd
                slot = Slot(
                    start_time=parse_time(start),
                    end_time=parse_time(end),
                    event=None,
                )

                action = MyCalendarAction(
                    expected_action=ExpectedAction(
                        command="add_event",
                        slot=slot,
                        event_id=id,
                    ),
                    performed_action=PerformedAction(
                        success=True,
                        slot=slot,
                    ),
                )

                res = env.step(action)
                print_result(res)

            except Exception:
                print("Usage: add <id> <title> <start> <end>")

        elif cmd[0] == "move":
            try:
                _, id, start, end = cmd

                action = MyCalendarAction(
                    expected_action=ExpectedAction(
                        command="move_event",
                        event_id=id,
                    ),
                    performed_action=PerformedAction(
                        success=True,
                        event_id=id,
                        slot=build_slot(start, end),
                    ),
                )

                res = env.step(action)
                print_result(res)

            except Exception:
                print("Usage: move <id> <start> <end>")

        elif cmd[0] == "delete":
            try:
                _, id = cmd

                action = MyCalendarAction(
                    expected_action=ExpectedAction(
                        command="delete_event",
                        event_id=id,
                    ),
                    performed_action=PerformedAction(
                        success=True,
                        event_id=id,
                    ),
                )

                res = env.step(action)
                print_result(res)

            except Exception:
                print("Usage: delete <id>")

        elif cmd[0] == "state":
            state = env.state()
            print(f"Steps: {state.step_count}, Episode: {state.episode_id}")
            print(f"Objective: {state.task_objective}")
            print(
                f"Meetings: {state.scheduled_meetings}/{state.target_meetings} "
                f"| Progress: {state.objective_progress:.2f}"
            )

        else:
            print("Unknown command (type 'help')")
