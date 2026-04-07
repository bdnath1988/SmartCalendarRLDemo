from datetime import datetime
from client import SmartCalendarEnv
from models import MyCalendarAction, CalendarEvent


def parse_time(t):
    return datetime.fromisoformat(f"2026-04-07T{t}:00")


def print_events(events):
    if not events:
        print("📭 No events scheduled")
        return

    print("\n📅 Current Calendar:")
    for e in events:
        print(f"  🕒 {e.start.time()} - {e.end.time()} | {e.title} (id={e.id})")


with SmartCalendarEnv(base_url="http://localhost:8000").sync() as env:

    print("\n🟢 Smart Calendar CLI Started")
    print("Type 'help' for commands\n")

    while True:
        cmd = input(">> ").strip().split()

        if not cmd:
            continue

        if cmd[0] == "exit":
            print("👋 Exiting...")
            break

        elif cmd[0] == "help":
            print("""
Commands:
  reset
  add <id> <title> <start> <end>
  move <id> <start> <end>
  delete <id>
  state
  exit
            """)

        elif cmd[0] == "reset":
            res = env.reset()
            print(f"🔄 {res.observation.message}")

        elif cmd[0] == "add":
            try:
                _, id, title, start, end = cmd

                action = MyCalendarAction(
                    action_type="add_event",
                    event=CalendarEvent(
                        id=id,
                        title=title,
                        start=parse_time(start),
                        end=parse_time(end),
                    ),
                )

                res = env.step(action)
                print(f"➡️ {res.observation.message} | Reward: {res.reward}")
                print_events(res.observation.events)

            except:
                print("❌ Usage: add <id> <title> <start> <end>")

        elif cmd[0] == "move":
            try:
                _, id, start, end = cmd

                action = MyCalendarAction(
                    action_type="move_event",
                    event_id=id,
                    new_start=parse_time(start),
                    new_end=parse_time(end),
                )

                res = env.step(action)
                print(f"➡️ {res.observation.message} | Reward: {res.reward}")
                print_events(res.observation.events)

            except:
                print("❌ Usage: move <id> <start> <end>")

        elif cmd[0] == "delete":
            try:
                _, id = cmd

                action = MyCalendarAction(
                    action_type="delete_event",
                    event_id=id,
                )

                res = env.step(action)
                print(f"➡️ {res.observation.message} | Reward: {res.reward}")
                print_events(res.observation.events)

            except:
                print("❌ Usage: delete <id>")

        elif cmd[0] == "state":
            state = env.state()
            print(f"📊 Steps: {state.step_count}, Task: {state.task_id}")

        else:
            print("❌ Unknown command (type 'help')")