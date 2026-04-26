import re
import json
import random
from datasets import Dataset
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer
from server.smart_calendar_agent_environment import CalendarEnv
from task_definitions import TaskDifficulty
from inference import build_prompt, parse_time
from models import MyCalendarAction, ExpectedAction, PerformedAction, Slot

# ── 1. Dataset ────────────────────────────────────────────────────
def generate_prompts(n=150):
    rows = []
    difficulties = (
        [TaskDifficulty.EASY] * 80 +
        [TaskDifficulty.MEDIUM] * 50 +
        [TaskDifficulty.HARD] * 20
    )
    random.shuffle(difficulties)
    for diff in difficulties[:n]:
        env = CalendarEnv()
        env.reset(diff)
        prompt = build_prompt(env.state)
        rows.append({
            "prompt": (
                "You are a smart calendar scheduling agent. "
                "Return only valid JSON.\n\n"
                f"USER:\n{prompt}\n\nASSISTANT:\n"
            ),
            "difficulty": diff.value,
        })
    return Dataset.from_list(rows)

dataset = generate_prompts(150)

# ── 2. Reward functions ───────────────────────────────────────────
def reward_valid_json(completions, prompts=None, **kwargs):
    results = []
    for c in completions:
        text = c[0]["content"] if isinstance(c, list) else c
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            results.append(-1.0)
            continue
        try:
            json.loads(match.group())
            results.append(0.3)
        except Exception:
            results.append(-0.3)
    return results

def reward_correct_time(completions, prompts=None, **kwargs):
    results = []
    for c in completions:
        text = c[0]["content"] if isinstance(c, list) else c
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            results.append(0.0)
            continue
        try:
            obj = json.loads(match.group())
            start = str(obj.get("start_time", ""))
            end   = str(obj.get("end_time", ""))
            if not re.match(r"^\d{2}:\d{2}$", start):
                results.append(-0.2)
                continue
            hour     = int(start.split(":")[0])
            end_hour = int(end.split(":")[0])
            results.append(1.0 if (8 <= hour <= 17 and end_hour > hour) else -0.3)
        except Exception:
            results.append(0.0)
    return results

def reward_env_objective(completions, prompts=None, difficulty=None, **kwargs):
    results = []
    for i, c in enumerate(completions):
        text     = c[0]["content"] if isinstance(c, list) else c
        diff_str = difficulty[i] if (difficulty and i < len(difficulty)) else "easy"
        try:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                results.append(0.0)
                continue
            action_json = json.loads(match.group())
            diff        = TaskDifficulty(diff_str)
            env         = CalendarEnv()
            env.reset(diff)
            start_iso = parse_time(action_json.get("start_time", "09:00")).isoformat()
            end_iso   = parse_time(action_json.get("end_time",   "10:00")).isoformat()
            slot      = Slot(start_time=start_iso, end_time=end_iso)
            event_id  = str(action_json.get("event_id", "kickoff"))
            action = MyCalendarAction(
                expected_action=ExpectedAction(
                    command=action_json.get("command", "add_event"),
                    slot=slot,
                    event_id=event_id,
                    day=action_json.get("day", "monday"),
                ),
                performed_action=PerformedAction(
                    success=True, slot=slot, event_id=event_id
                ),
            )
            obs    = env.step(action)
            meta   = obs.metadata or {}
            reward = float(meta.get("reward_objective_progress", 0.0))
            if not meta.get("action_valid", True):
                reward = -0.2
            results.append(reward)
        except Exception:
            results.append(0.0)
    return results

# ── 3. Training history collector ─────────────────────────────────
training_history = {
    "step":                  [],
    "loss":                  [],
    "reward_valid_json":     [],
    "reward_correct_time":   [],
    "reward_env_objective":  [],
}

class RewardHistoryCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        training_history["step"].append(state.global_step)
        training_history["loss"].append(logs.get("loss", 0.0))
        training_history["reward_valid_json"].append(
            logs.get("rewards/reward_valid_json/mean",
            logs.get("reward_valid_json", 0.0))
        )
        training_history["reward_correct_time"].append(
            logs.get("rewards/reward_correct_time/mean",
            logs.get("reward_correct_time", 0.0))
        )
        training_history["reward_env_objective"].append(
            logs.get("rewards/reward_env_objective/mean",
            logs.get("reward_env_objective", 0.0))
        )

# ── 4. GRPO config ────────────────────────────────────────────────
grpo_config = GRPOConfig(
    output_dir="smart-calendar-qwen-grpo",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,
    num_generations=4,
    max_completion_length=150,
    logging_steps=1,
    save_steps=25,
    temperature=0.9,
    push_to_hub=True,
    report_to="none",
)

# ── 5. Train ──────────────────────────────────────────────────────
grpo_trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    reward_funcs=[
        reward_valid_json,
        reward_correct_time,
        reward_env_objective,
    ],
    train_dataset=dataset,
    args=grpo_config,
    callbacks=[RewardHistoryCallback()],
)

grpo_trainer.train()
grpo_trainer.push_to_hub()

# Save history
with open("training_history.json", "w") as f:
    json.dump(training_history, f, indent=2)
