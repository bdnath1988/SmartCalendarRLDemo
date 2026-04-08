"""
Inference Script Example
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()
                     method

- Defaults are set only for API_BASE_URL and MODEL_NAME 
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
    MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")
    
- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each tasks should return score in [0, 1]

  Example:
    [START] task=click-test env=miniwob model=Qwen3-VL-30B
    [STEP] step=1 action=click('123') reward=0.00 done=false error=null
    [STEP] step=2 action=fill('456','text') reward=0.00 done=false error=null
    [STEP] step=3 action=click('789') reward=1.00 done=true error=null
    [END] success=true steps=3 score=1.00 rewards=0.00,0.00,1.00
"""

import os
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import requests
from models import CalendarEvent, ExpectedAction, MyCalendarAction, PerformedAction, Slot

API_KEY = (
    os.getenv("HF_TOKEN")
    or os.getenv("API_KEY")
    or os.getenv("OPENAI_API_KEY")
    or "deterministic-baseline-key"
)
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("TASK_NAME", "smart-calendar-scheduling")
BENCHMARK = os.getenv("BENCHMARK", "smart_calendar")
MAX_STEPS = int(os.getenv("MAX_STEPS", "8"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.6"))
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")

# Reward bounds from environment shaping: base in [-1.0, 1.0], progress bonus in [0.0, 0.5]
MIN_REWARD_PER_STEP = -1.0
MAX_REWARD_PER_STEP = 1.5


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
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def normalize_score(total_reward: float, steps_taken: int) -> float:
    if steps_taken <= 0:
        return 0.0
    min_total = steps_taken * MIN_REWARD_PER_STEP
    max_total = steps_taken * MAX_REWARD_PER_STEP
    if max_total <= min_total:
        return 0.0
    score = (total_reward - min_total) / (max_total - min_total)
    return min(max(score, 0.0), 1.0)


def build_slot_for_step(step: int, event_id: str, title: str) -> Tuple[Slot, str]:
    # Deterministic scheduling strategy: 09:00-10:00, 10:00-11:00, ...
    base_day = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_hour = 9 + ((step - 1) % 8)
    start_time = base_day + timedelta(hours=start_hour)
    end_time = start_time + timedelta(hours=1)

    slot = Slot(
        start_time=start_time,
        end_time=end_time,
        event=CalendarEvent(event_id=event_id, title=title),
    )
    action_repr = f"add_event(id={event_id},title={title},start={start_time.strftime('%H:%M')},end={end_time.strftime('%H:%M')})"
    return slot, action_repr


def main() -> None:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    graded_score: Optional[float] = None

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        res = requests.post(f"{BASE_URL}/reset", timeout=30)
        res.raise_for_status()
        result = res.json()
        task_objective = "unknown-objective"
        try:
            state_res = requests.get(f"{BASE_URL}/state", timeout=30)
            state_res.raise_for_status()
            task_objective = state_res.json().get("task_objective", task_objective)
        except Exception:
            pass

        for step in range(1, MAX_STEPS + 1):
            event_id = str(step)
            title = f"Meeting-{step}"
            slot, action_repr = build_slot_for_step(step, event_id, title)
            action_repr = f"{action_repr}|objective={task_objective}"

            action = MyCalendarAction(
                expected_action=ExpectedAction(
                    command="add_event",
                    slot=slot,
                    event_id=event_id,
                ),
                performed_action=PerformedAction(
                    success=True,
                    slot=slot,
                    event_id=event_id,
                ),
            )

            error: Optional[str] = None
            try:
                res = requests.post(
                    f"{BASE_URL}/step",
                    json={"action": action.model_dump(mode="json")},
                    timeout=30,
                )
                res.raise_for_status()
                result = res.json()
                reward = result.get("reward", 0.0) or 0.0
                done = result.get("done", False)
                # Prefer explicit environment grader score when exposed in metadata.
                graded_score = (
                    result.get("metadata", {}).get("score")
                    if isinstance(result.get("metadata"), dict)
                    else None
                )
                if graded_score is None and isinstance(result.get("observation"), dict):
                    graded_score = result.get("observation", {}).get("metadata", {}).get("score")
            except Exception as exc:
                reward = 0.0
                done = True
                error = str(exc)

            log_step(step=step, action=action_repr, reward=reward, done=done, error=error)

            rewards.append(reward)
            steps_taken = step

            if done:
                break

        score = graded_score if graded_score is not None else normalize_score(sum(rewards), steps_taken)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception:
        score = graded_score if graded_score is not None else normalize_score(sum(rewards), steps_taken)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    main()