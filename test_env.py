import sys
import os
sys.path.append(os.path.dirname(__file__))

from server.smart_calendar_agent_environment import CalendarEnv
from models import Action


def run_tests():
    print("🚀 Starting environment test...\n")

    env = CalendarEnv()

    # 🔹 Reset environment
    state = env.reset(task_id=0)
    print("✅ Initial State:")
    print(state)
    print("-" * 50)

    # 🔹 Test 1: Create Coffee event at 2PM
    print("🧪 Test 1: Creating Coffee event...")
    state, reward, done, info = env.step(Action(
        command="create",
        args={
            "title": "Coffee",
            "start": "2026-04-06T14:00:00",
            "end": "2026-04-06T14:30:00"
        }
    ))

    print("State:", state)
    print("Reward:", reward)
    print("Done:", done)
    print("-" * 50)

    # 🔹 Test 2: Create overlapping event (should trigger conflict)
    print("🧪 Test 2: Creating conflicting event...")
    state, reward, done, info = env.step(Action(
        command="create",
        args={
            "title": "Meeting",
            "start": "2026-04-06T14:15:00",
            "end": "2026-04-06T15:00:00"
        }
    ))

    print("State:", state)
    print("Reward:", reward)
    print("Done:", done)
    print("-" * 50)

    # 🔹 Test 3: Delete event
    print("🧪 Test 3: Deleting event...")
    if state.events:
        event_id = state.events[0].id
        state, reward, done, info = env.step(Action(
            command="delete",
            args={"id": event_id}
        ))

        print("State:", state)
        print("Reward:", reward)
        print("Done:", done)

    print("\n🎉 Testing complete!")


if __name__ == "__main__":
    run_tests()