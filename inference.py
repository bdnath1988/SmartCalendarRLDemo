import os
import json
import asyncio
import random
from typing import List, Optional
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI

from client import SmartCalendarEnv
from models import MyCalendarAction, ExpectedAction, PerformedAction, Slot

load_dotenv()

# ================= CONFIG =================
IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "smart_calendar_agent_env")

# Use grader-injected proxy variables exactly as required by validator.
API_KEY = os.environ["API_KEY"]
API_BASE_URL = os.environ["API_BASE_URL"]
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

TASK_NAME = os.getenv("TASK_NAME", "smart-calendar")
BENCHMARK = os.getenv("BENCHMARK", "smart_calendar")
MAX_STEPS = int(os.getenv("MAX_STEPS", "8"))

SUCCESS_SCORE_THRESHOLD = 0.6


def _llm_client() -> OpenAI:
    return OpenAI(api_key=API_KEY, base_url=API_BASE_URL)


# ================= LOGGING =================
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ================= HELPERS =================
def parse_time(t: str) -> datetime:
    today = datetime.now().date()
    return datetime.fromisoformat(f"{today.isoformat()}T{t}:00")


def build_prompt(state) -> str:
    return f"""
You are a smart calendar scheduling agent.

Objective:
{state.task_objective}

Task goal:
{getattr(state, 'task_goal', 'schedule meetings')}

Current:
- Scheduled meetings: {state.scheduled_meetings}
- Target meetings: {state.target_meetings}
- Existing events: {getattr(state, 'events', [])}
- Free slots: {getattr(state, 'free_slots', [])[:8]}
- Failed steps: {getattr(state, 'failed_steps', 0)}

IMPORTANT:
- You can ONLY use these commands:
  - add_event
  - move_event
  - delete_event
  - search_slot
- Do NOT use any other command like remove_event.
- Do NOT repeat the same action if it gave no reward
- If an action fails or gives 0 reward, try a different strategy
- Maximize total reward

Before taking an action:
- Check which time slots are already used
- Do NOT reuse occupied slots
- Prefer new time slots
- Use "search_slot" to find free time before adding events.

Think step by step.

Action diversity hint:
- If the last step failed, change event_id or time slot
- Prefer free slots that are not already occupied
- Avoid duplicates and repeated timestamps

STRICT RULE:
Return ONLY valid JSON. No explanation.

Example:
{{
  "command": "add_event",
  "event_id": "4",
  "start_time": "12:00",
  "end_time": "13:00"
}}
"""


def get_llm_action(llm_client: OpenAI, state) -> dict:
    try:
        response = llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": build_prompt(state)}],
            temperature=0.7,
        )

        text = (response.choices[0].message.content or "").strip()

        import re

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return json.loads(text)
    except Exception:
        return fallback_action(step=state.scheduled_meetings)


def fallback_action(step: int) -> dict:
    hour = 9 + step
    return {
        "command": "add_event",
        "event_id": str(step),
        "start_time": f"{hour:02d}:00",
        "end_time": f"{hour + 1:02d}:00",
    }


def is_duplicate_action(action_json: dict, previous_action_json: Optional[dict]) -> bool:
    if not previous_action_json:
        return False
    return (
        action_json.get("command") == previous_action_json.get("command")
        and action_json.get("event_id") == previous_action_json.get("event_id")
        and action_json.get("start_time") == previous_action_json.get("start_time")
        and action_json.get("end_time") == previous_action_json.get("end_time")
    )


# ================= MAIN =================
async def main() -> None:
    llm_client = _llm_client()

    random.seed(42)
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    successful_events = 0
    previous_action_json: Optional[dict] = None

    log_start(TASK_NAME, BENCHMARK, MODEL_NAME)

    try:
        env = await SmartCalendarEnv.from_docker_image(IMAGE_NAME)
    except Exception:
        log_end(False, 0, 0.0, [])
        return

    try:
        _ = await env.reset()

        for step in range(1, MAX_STEPS + 1):
            state = await env.state()

            action_json = get_llm_action(llm_client, state)
            if action_json.get("command") == "remove_event":
                action_json["command"] = "delete_event"

            if is_duplicate_action(action_json, previous_action_json):
                action_json = fallback_action(step)

            try:
                slot = Slot(
                    start_time=parse_time(action_json["start_time"]).isoformat(),
                    end_time=parse_time(action_json["end_time"]).isoformat(),
                )
            except Exception:
                slot = Slot(
                    start_time=parse_time("09:00").isoformat(),
                    end_time=parse_time("10:00").isoformat(),
                )

            action = MyCalendarAction(
                expected_action=ExpectedAction(
                    command=action_json.get("command", "add_event"),
                    slot=slot,
                    event_id=action_json.get("event_id", str(step)),
                ),
                performed_action=PerformedAction(
                    success=True,
                    slot=slot,
                    event_id=action_json.get("event_id", str(step)),
                ),
            )

            try:
                result = await env.step(action)
                reward = result.reward or 0.0
                done = result.done
                error = getattr(result, "last_action_error", None)

                current_state = await env.state()
                score = getattr(current_state, "objective_progress", score)
                if reward >= 1.0:
                    successful_events += 1
            except Exception as exc:
                reward = 0.0
                done = True
                error = str(exc)

            log_step(step, json.dumps(action_json), reward, done, error)

            rewards.append(reward)
            steps_taken = step
            previous_action_json = action_json

            target_meetings = getattr(state, "target_meetings", 3)
            if successful_events >= target_meetings:
                done = True
                success = True

            if done:
                break

        if not success:
            success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception:
            pass

        score = min(max(score, 0.0), 1.0)
        log_end(success, steps_taken, score, rewards)


if __name__ == "__main__":
    asyncio.run(main())
