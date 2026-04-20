import os
import json
import asyncio
import random
import re
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
    return textwrap.dedent(f"""
        You are a smart calendar scheduling agent.
        Objective: {state.task_objective}
        Goal: {getattr(state, 'task_goal', 'schedule meetings')}

        Current State:
        - Scheduled: {state.scheduled_meetings} / {state.target_meetings}
        - Existing events: {getattr(state, 'events', [])}
        - Free slots: {getattr(state, 'free_slots', [])[:8]}

        Return ONLY valid JSON:
        {{
          "command": "add_event",
          "event_id": "4",
          "start_time": "12:00",
          "end_time": "13:00"
        }}
    """).strip()

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
        return fallback_action(step=state.scheduled_meetings)

def fallback_action(step: int) -> dict:
    hour = 9 + (step % 8)
    return {
        "command": "add_event",
        "event_id": f"fb_{step}_{random.randint(10,99)}",
        "start_time": f"{hour:02d}:00",
        "end_time": f"{hour + 1:02d}:00",
    }

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
            env = SmartCalendarEnv(base_url=EXISTING_BASE_URL)
            await env.connect()
        else:
            env = await SmartCalendarEnv.from_docker_image(
                IMAGE_NAME,
                env_vars={"TASK_NAME": task}
            )
        
        await env.reset()

        for step in range(1, MAX_STEPS + 1):
            state = await env.state()
            action_json = get_llm_action(llm_client, state)

            # Map disallowed commands if necessary
            if action_json.get("command") == "remove_event":
                action_json["command"] = "delete_event"

            # Parse slot times safely
            try:
                start_iso = parse_time(action_json["start_time"]).isoformat()
                end_iso = parse_time(action_json["end_time"]).isoformat()
            except:
                start_iso, end_iso = "2026-04-12T09:00:00", "2026-04-12T10:00:00"

            slot = Slot(start_time=start_iso, end_time=end_iso)
            action = MyCalendarAction(
                expected_action=ExpectedAction(
                    command=action_json.get("command", "add_event"),
                    slot=slot,
                    event_id=str(action_json.get("event_id", step))
                ),
                performed_action=PerformedAction(
                    success=True,
                    slot=slot,
                    event_id=str(action_json.get("event_id", step))
                )
            )

            result = await env.step(action)
            
            reward = result.reward or 0.0
            done = result.done
            error = getattr(result, "last_action_error", None)

            # Mandatory STEP log
            log_step(step, json.dumps(action_json), reward, done, error)

            rewards.append(reward)
            steps_taken = step
            
            # Update score based on objective progress
            current_state = await env.state()
            score = getattr(current_state, "objective_progress", 0.0)

            if done:
                break

        # Since openenv strictly forbids 0.0 and 1.0 logic, clamped score
        score = max(0.01, min(0.99, float(score)))
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        # print(f"[DEBUG] Runtime Error: {e}", flush=True)
        pass
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