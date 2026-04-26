# Smart Calendar Training Plan

Deadline-oriented plan for the OpenEnv hackathon submission.

## Short Answer

Yes, push the latest environment fixes before training/evaluation if your Colab, HF Job, or judges will use the hosted Hugging Face Space.

Required order:

1. Commit and push the current repo changes.
2. Make sure the Hugging Face Space rebuilds successfully.
3. Run a small training job using TRL or Unsloth.
4. Run before/after evaluation and save the reward evidence.
5. Add the Space URL, training link, and results to the README/submission.

The training does not need to be huge. A small run is acceptable if it clearly shows reward improvement.

## What To Show Judges

Minimum evidence to include:

- Base model score before training.
- Fine-tuned model score after training.
- A small reward table or curve.
- One example where the base model fails, and the trained model schedules meetings in dependency order.
- Link to the hosted OpenEnv Space.
- Link to the Colab or HF Job logs/model.

Example table:

| Model | Easy | Medium | Hard | Avg |
|---|---:|---:|---:|---:|
| Base Qwen small | 0.30 | 0.00 | 0.00 | 0.10 |
| Trained model | 0.99 | 0.66 | 0.57 | 0.74 |

It is okay if the numbers are not perfect. The important part is that the trained model improves.

## Fastest Training Strategy

Use TRL or Unsloth for supervised fine-tuning on successful environment trajectories.

This is faster and safer than trying to complete a full GRPO port before the deadline. The reward improvement still comes from the environment: train on high-reward actions, then evaluate against the OpenEnv environment before and after training.

Use a small model:

- `Qwen/Qwen2.5-0.5B-Instruct`
- or `Qwen/Qwen2.5-1.5B-Instruct`
- or `Qwen/Qwen3-0.6B`

Do not use a 7B+ model unless you already know the GPU can handle it.

## Colab Cells

### 1. Install

```python
!pip install -U -q "trl>=0.16.0" "transformers>=4.51.0" "datasets" "accelerate" "peft" "bitsandbytes" "huggingface_hub" "openenv-core[core]>=0.2.2"
```

Optional if using Unsloth:

```python
!pip install -U -q unsloth
```

### 2. Login

```python
from huggingface_hub import notebook_login
notebook_login()
```

### 3. Clone Your Repo

Replace the URL with your GitHub or Hugging Face repo URL.

```python
!git clone https://github.com/YOUR_USERNAME/SmartCalendarRLDemo.git
%cd SmartCalendarRLDemo
!pip install -e .
```

If you use the Hugging Face Space repo:

```python
!git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
%cd YOUR_SPACE_NAME
!pip install -e .
```

### 4. Generate Expert Training Data

This creates successful demonstrations directly from the environment.

```python
import json
from pathlib import Path
from datetime import datetime

from server.smart_calendar_agent_environment import CalendarEnv
from task_definitions import TaskDifficulty
from models import MyCalendarAction, ExpectedAction, PerformedAction, Slot
from inference import build_prompt, fallback_action, parse_time

def make_action(action_json):
    start_iso = parse_time(action_json["start_time"]).isoformat()
    end_iso = parse_time(action_json["end_time"]).isoformat()
    slot = Slot(start_time=start_iso, end_time=end_iso)
    event_id = str(action_json["event_id"])
    return MyCalendarAction(
        expected_action=ExpectedAction(
            command=action_json.get("command", "add_event"),
            slot=slot,
            event_id=event_id,
            day=action_json.get("day"),
        ),
        performed_action=PerformedAction(success=True, slot=slot, event_id=event_id),
    )

def collect_examples(repeats=30):
    rows = []
    for _ in range(repeats):
        for difficulty in [TaskDifficulty.EASY, TaskDifficulty.MEDIUM, TaskDifficulty.HARD]:
            env = CalendarEnv()
            env.reset(difficulty)
            for _step in range(12):
                state = env.state
                prompt = build_prompt(state)
                action_json = fallback_action(state.scheduled_meetings, state)
                obs = env.step(make_action(action_json))
                meta = obs.metadata or {}
                reward = float(meta.get("reward_objective_progress", 0.0))
                if meta.get("action_valid"):
                    rows.append({
                        "prompt": prompt,
                        "completion": json.dumps(action_json),
                        "reward": reward,
                        "difficulty": difficulty.value,
                    })
                if obs.done:
                    break
    return rows

rows = collect_examples(repeats=40)
Path("training_data").mkdir(exist_ok=True)
with open("training_data/smart_calendar_expert.jsonl", "w") as f:
    for row in rows:
        f.write(json.dumps(row) + "\n")

len(rows), rows[0]
```

### 5. Train With TRL SFTTrainer

```python
import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
output_dir = "smart-calendar-qwen-0.5b-sft"

records = []
with open("training_data/smart_calendar_expert.jsonl") as f:
    for line in f:
        row = json.loads(line)
        records.append({
            "text": (
                "You are a smart calendar scheduling agent.\n"
                "Return only valid JSON.\n\n"
                f"USER:\n{row['prompt']}\n\n"
                f"ASSISTANT:\n{row['completion']}"
            )
        })

dataset = Dataset.from_list(records)

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_4bit=True,
)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

args = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=1,
    save_steps=20,
    max_seq_length=2048,
    packing=False,
    push_to_hub=True,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=peft_config,
)

trainer.train()
trainer.push_to_hub()
```

### 6. Evaluate Before And After

Run the base model and trained model against the same environment.

```python
import json, re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from server.smart_calendar_agent_environment import CalendarEnv
from task_definitions import TaskDifficulty
from models import MyCalendarAction, ExpectedAction, PerformedAction, Slot
from inference import build_prompt, parse_time

def parse_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except Exception:
        return None

def model_action(model, tokenizer, state):
    prompt = build_prompt(state)
    full_prompt = (
        "You are a smart calendar scheduling agent. Return only valid JSON.\n\n"
        f"USER:\n{prompt}\n\nASSISTANT:\n"
    )
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return parse_json(text), text

def make_action(action_json, fallback_event_id="kickoff"):
    if not action_json:
        action_json = {
            "command": "add_event",
            "event_id": fallback_event_id,
            "day": "monday",
            "start_time": "10:00",
            "end_time": "11:00",
        }
    start_iso = parse_time(action_json.get("start_time", "10:00")).isoformat()
    end_iso = parse_time(action_json.get("end_time", "11:00")).isoformat()
    slot = Slot(start_time=start_iso, end_time=end_iso)
    event_id = str(action_json.get("event_id", fallback_event_id))
    return MyCalendarAction(
        expected_action=ExpectedAction(
            command=action_json.get("command", "add_event"),
            slot=slot,
            event_id=event_id,
            day=action_json.get("day", "monday"),
        ),
        performed_action=PerformedAction(success=True, slot=slot, event_id=event_id),
    )

def evaluate_model(model_name, max_steps=8):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
    )
    results = {}
    for difficulty in [TaskDifficulty.EASY, TaskDifficulty.MEDIUM, TaskDifficulty.HARD]:
        env = CalendarEnv()
        env.reset(difficulty)
        rewards = []
        errors = []
        for _ in range(max_steps):
            state = env.state
            action_json, raw = model_action(model, tokenizer, state)
            obs = env.step(make_action(action_json))
            meta = obs.metadata or {}
            rewards.append(float(meta.get("reward_objective_progress", 0.0)))
            errors.append(meta.get("rejection_reason"))
            if obs.done:
                break
        results[difficulty.value] = {
            "score": env.state.objective_progress,
            "rewards": rewards,
            "errors": errors,
        }
    return results

base_results = evaluate_model("Qwen/Qwen2.5-0.5B-Instruct")
trained_results = evaluate_model("YOUR_USERNAME/smart-calendar-qwen-0.5b-sft")

print("BASE", json.dumps(base_results, indent=2))
print("TRAINED", json.dumps(trained_results, indent=2))

with open("training_results.json", "w") as f:
    json.dump({"base": base_results, "trained": trained_results}, f, indent=2)
```

Download or push `training_results.json`. This is your reward improvement evidence.

## Optional HF Jobs

Hugging Face Jobs can run training workloads on paid compute. Official docs say Jobs can be run through the `hf jobs` CLI or Python client, and can use `hf jobs uv run` with a hardware flavor.

Install/login locally:

```bash
pip install -U "huggingface_hub[cli]"
hf auth login
```

Example shape:

```bash
hf jobs uv run --flavor t4-small --with trl --with transformers --with datasets --with accelerate --with peft --with bitsandbytes train_smart_calendar.py
```

Use this only if Colab is slow or unavailable. For the deadline, Colab is simpler because you can see failures immediately.

## Final Submission Checklist

- Current repo changes pushed.
- Hugging Face Space rebuilt and `/reset` works.
- Training notebook or script linked.
- Trained model pushed to Hugging Face Hub.
- `training_results.json` or reward table added.
- README corrected to describe current five reward signals.
- Blog/video explains:
  - problem: real calendar scheduling is constrained and stateful
  - environment: dependencies, conflicts, timezones, spacing
  - rewards: five independent signals
  - improvement: before/after reward table

## Sources

- Hugging Face Jobs docs: https://huggingface.co/docs/hub/jobs
- Hugging Face Jobs configuration docs: https://huggingface.co/docs/hub/jobs-configuration
- Hugging Face Hub Jobs guide: https://huggingface.co/docs/huggingface_hub/guides/jobs
