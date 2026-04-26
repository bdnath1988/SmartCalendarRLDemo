import os
import json
import asyncio
import re
from dotenv import load_dotenv
load_dotenv()
import textwrap
from typing import List, Optional
from datetime import datetime

from openai import OpenAI

# Importing the environment-specific models and client
from client import SmartCalendarEnv
from models import MyCalendarAction, ExpectedAction, PerformedAction, Slot

# ================= CONFIG =================
# Mandatory Environment Variables
IMAGE_NAME = os.getenv("IMAGE_NAME") or os.getenv("LOCAL_IMAGE_NAME", "smart_calendar_agent_env:latest")
EXISTING_BASE_URL = os.getenv("BASE_URL")  # if set, skip docker
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or ""
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

# Benchmark Configuration
TASKS = ("task-easy-1", "task-medium-1", "task-hard-1")
_TASK_DIFFICULTY = {
    "task-easy-1": "easy",
    "task-medium-1": "medium",
    "task-hard-1": "hard",
}
BENCHMARK = os.getenv("BENCHMARK", "smart_calendar")
MAX_STEPS = int(os.getenv("MAX_STEPS", "8"))
SUCCESS_SCORE_THRESHOLD = 0.6

def _llm_client() -> OpenAI:
    """Initialize the OpenAI client using the mandatory environment variables."""
    return OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

# ================= MANDATORY LOGGING =================
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
    dep_graph: dict = getattr(state, "dependency_graph", {})
    scheduled: list = getattr(state, "scheduled_meeting_ids", getattr(state, "events", []))
    free_slots: list = getattr(state, "free_slots", [])[:10]

    available = _available_meetings(dep_graph, scheduled)

    return textwrap.dedent(f"""
        You are a smart calendar scheduling agent.
        Objective: {state.task_objective}

        Progress: {state.scheduled_meetings} / {state.target_meetings} meetings scheduled
        Already scheduled: {scheduled}
        Ready to schedule now (dependencies met): {available}
        Dependency order: {dep_graph}

        Free slots (format: "day HH:MM-HH:MM UTC"): {free_slots}

        Commands (return ONE as JSON):
          add_event    — book a new meeting   (requires: event_id, day, start_time, end_time)
          search_slot  — query free slots     (requires: day)
          move_event   — reschedule a meeting (requires: event_id, day, start_time, end_time)
          delete_event — remove a meeting     (requires: event_id)

        Rules:
        - event_id MUST be one of the ready-to-schedule meetings listed above
        - day MUST be a weekday name, e.g. "monday"
        - start_time / end_time in HH:MM UTC format (working hours 08:00-18:00)
        - Schedule meetings in dependency order

        Return ONLY valid JSON, for example:
        {{
          "command": "add_event",
          "event_id": "{available[0] if available else 'kickoff'}",
          "day": "monday",
          "start_time": "09:00",
          "end_time": "10:00"
        }}
    """).strip()

def _deps_for_prompt(deps) -> list:
    """Normalize dependency graph values from state or observation metadata."""
    if isinstance(deps, dict):
        return deps.get("deps", [])
    return deps or []

def _available_meetings(dep_graph: dict, scheduled: list) -> list:
    return [
        mid for mid, deps in dep_graph.items()
        if mid not in scheduled and all(d in scheduled for d in _deps_for_prompt(deps))
    ]

def get_llm_action(llm_client: OpenAI, state) -> dict:    
    try:
        response = llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": build_prompt(state)}],
            temperature=0.7,
        )

        text = (response.choices[0].message.content or "").strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return json.loads(text)
    except Exception as e:
        # print(f"[DEBUG] Model request failed: {e}", flush=True)
        return fallback_action(step=state.scheduled_meetings, state=state)

def fallback_action(step: int, state=None) -> dict:
    dep_graph: dict = getattr(state, "dependency_graph", {}) if state else {}
    scheduled: list = getattr(state, "scheduled_meeting_ids", getattr(state, "events", [])) if state else []
    available = _available_meetings(dep_graph, scheduled)
    event_id = available[0] if available else "kickoff"
    day, hour = _first_usable_slot(state, step)
    return {
        "command": "add_event",
        "event_id": event_id,
        "day": day,
        "start_time": f"{hour:02d}:00",
        "end_time": f"{hour + 1:02d}:00",
    }

def repair_action(action_json: dict, state) -> dict:
    """Keep LLM output inside the current task's valid action frontier."""
    dep_graph: dict = getattr(state, "dependency_graph", {})
    scheduled: list = getattr(state, "scheduled_meeting_ids", getattr(state, "events", []))
    available = _available_meetings(dep_graph, scheduled)

    if action_json.get("command") == "remove_event":
        action_json["command"] = "delete_event"

    if action_json.get("command") != "add_event":
        return action_json

    event_id = str(action_json.get("event_id", ""))
    if event_id not in available:
        return fallback_action(step=getattr(state, "scheduled_meetings", 0), state=state)

    try:
        requested_hour = int(str(action_json.get("start_time", "")).split(":", 1)[0])
    except Exception:
        return fallback_action(step=getattr(state, "scheduled_meetings", 0), state=state)

    requested_day = str(action_json.get("day", "")).lower()
    if not _is_usable_slot(state, requested_day, requested_hour):
        fallback = fallback_action(step=getattr(state, "scheduled_meetings", 0), state=state)
        fallback["event_id"] = event_id
        return fallback

    action_json["day"] = requested_day
    action_json["event_id"] = event_id
    action_json["start_time"] = f"{requested_hour:02d}:00"
    action_json["end_time"] = f"{requested_hour + 1:02d}:00"
    return action_json

def _first_usable_slot(state, step: int) -> tuple[str, int]:
    week = getattr(state, "week", None)
    days = ["monday", "tuesday", "wednesday", "thursday", "friday"]
    enforce_gap = getattr(state, "target_meetings", 0) >= 7

    if week is not None and getattr(week, "days", None):
        for day in days:
            day_calendar = week.days.get(day)
            if day_calendar is None:
                continue
            for slot in day_calendar.slots:
                if slot.event is not None:
                    continue
                hour = datetime.fromisoformat(slot.start_time).hour
                if enforce_gap and _has_nearby_event(day_calendar.slots, hour):
                    continue
                return day, hour

    free_slots = getattr(state, "free_slots", []) if state else []
    for label in free_slots:
        match = re.match(r"(?P<day>\w+)\s+(?P<hour>\d{2}):\d{2}-", label)
        if match:
            return match.group("day").lower(), int(match.group("hour"))

    return days[step % len(days)], 10 + (step % 6)

def _is_usable_slot(state, day: str, hour: int) -> bool:
    week = getattr(state, "week", None)
    if week is None or not getattr(week, "days", None):
        return 8 <= hour < 18

    day_calendar = week.days.get(day)
    if day_calendar is None:
        return False

    for slot in day_calendar.slots:
        slot_hour = datetime.fromisoformat(slot.start_time).hour
        if slot_hour == hour:
            if slot.event is not None:
                return False
            enforce_gap = getattr(state, "target_meetings", 0) >= 7
            return not (enforce_gap and _has_nearby_event(day_calendar.slots, hour))
    return False

def _has_nearby_event(slots: list, hour: int) -> bool:
    for slot in slots:
        slot_hour = datetime.fromisoformat(slot.start_time).hour
        if abs(slot_hour - hour) > 1:
            continue
        if slot.event is not None:
            return True
    return False

# ================= MAIN LOOP =================
from typing import Any, Dict

async def run_episode(task: str, llm_client: OpenAI) -> Dict[str, Any]:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    
    log_start(task, BENCHMARK, MODEL_NAME)

    env = None
    try:
        if EXISTING_BASE_URL:
            try:
                env = SmartCalendarEnv(base_url=EXISTING_BASE_URL)
                await env.connect()
            except Exception as exc:
                if "localhost" not in EXISTING_BASE_URL and "127.0.0.1" not in EXISTING_BASE_URL:
                    raise
                print(
                    f"[DEBUG] Local BASE_URL unavailable ({exc}); falling back to Docker image {IMAGE_NAME}.",
                    flush=True,
                )
                env = await SmartCalendarEnv.from_docker_image(
                    IMAGE_NAME,
                    env_vars={"TASK_NAME": task}
                )
        else:
            env = await SmartCalendarEnv.from_docker_image(
                IMAGE_NAME,
                env_vars={"TASK_NAME": task}
            )
        
        await env.reset(task=_TASK_DIFFICULTY.get(task, "easy"))

        for step in range(1, MAX_STEPS + 1):
            state = await env.state()
            action_json = get_llm_action(llm_client, state)
            action_json = repair_action(action_json, state)

            # Parse slot times safely
            try:
                start_iso = parse_time(action_json["start_time"]).isoformat()
                end_iso = parse_time(action_json["end_time"]).isoformat()
            except:
                start_iso, end_iso = "2026-04-12T09:00:00", "2026-04-12T10:00:00"

            slot = Slot(start_time=start_iso, end_time=end_iso)
            event_id = str(action_json.get("event_id", step))
            action = MyCalendarAction(
                expected_action=ExpectedAction(
                    command=action_json.get("command", "add_event"),
                    slot=slot,
                    event_id=event_id,
                    day=action_json.get("day"),
                ),
                performed_action=PerformedAction(
                    success=True,
                    slot=slot,
                    event_id=event_id,
                )
            )

            previous_score = score
            result = await env.step(action)
            current_state = await env.state()

            # Round-2 rewards live in observation metadata. Some OpenEnv/Docker
            # response paths expose only the top-level 0.0 reward, so inference
            # falls back to visible objective progress for the demo reward log.
            obs = getattr(result, "observation", result)
            meta = getattr(obs, "metadata", None) or {}
            metadata_reward = meta.get("reward_objective_progress")
            top_level_reward = getattr(result, "reward", 0.0) or 0.0
            progress_reward = getattr(current_state, "objective_progress", previous_score)
            reward = (
                float(metadata_reward)
                if metadata_reward is not None
                else float(top_level_reward or progress_reward)
            )
            done = result.done
            error = meta.get("rejection_reason") or getattr(result, "last_action_error", None)

            # Mandatory STEP log
            log_step(step, json.dumps(action_json), reward, done, error)

            rewards.append(reward)
            steps_taken = step
            
            # Update score based on objective progress
            score = getattr(current_state, "objective_progress", 0.0)

            if done:
                break

        # Since openenv strictly forbids 0.0 and 1.0 logic, clamped score
        score = max(0.01, min(0.99, float(score)))
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Runtime Error: {e}", flush=True)
    finally:
        if env is not None:
            try:
                await env.close()
            except:
                pass
        # Mandatory END log
        log_end(success, steps_taken, score, rewards)

    return {"task": task, "score": score, "success": success, "steps": steps_taken}

async def main() -> None:
    if not API_KEY:
        print(
            "[DEBUG] No API key found in HF_TOKEN/API_KEY env var. "
            "LLM calls will fail and the script will use fallback no-op actions.",
            flush=True,
        )
    llm_client = _llm_client()

    results = []
    for task in TASKS:
        results.append(await run_episode(task, llm_client))

    # Friendly aggregate (printed AFTER the strict-format [END] lines, so it
    # doesn't break parsers that only consume the [START]/[STEP]/[END] grammar).
    avg = sum(r["score"] for r in results) / len(results)
    print(
        "[SUMMARY] "
        + " | ".join(f"{r['task']}={r['score']:.2f}" for r in results)
        + f" | avg={avg:.2f}",
        flush=True,
    )

if __name__ == "__main__":
    asyncio.run(main())
