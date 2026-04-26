# 📅 SmartCalendar RL — AI Executive Assistant Environment

> **Meta OpenEnv Hackathon India 2026** · Theme 3.2: Personalized Tasks + Theme 2: Long-Horizon Planning

[![HuggingFace Space](https://img.shields.io/badge/🤗%20Space-SmartCalendarRLDemo-yellow)](https://huggingface.co/spaces/kohantika/SmartCalendarRLDemo)
[![Trained Model](https://img.shields.io/badge/🤗%20Model-smart--calendar--qwen--grpo-orange)](https://huggingface.co/kohantika/smart-calendar-qwen-grpo)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1mVmrGGahOBqqT3g3VCGmPYKu8wwZDjg3?usp=sharing)

---

## 🎯 The Problem

LLMs are good at answering questions. They are poor at **acting** inside a calendar — understanding which slots are free, respecting scheduling constraints, recovering from failed actions, and making progress across multiple steps toward a goal.

Standard instruction tuning cannot fix this because it teaches the model what to say, not how to act. Reinforcement Learning is required: the model must take actions, receive feedback from the environment, and update its behaviour based on what worked and what did not.

This environment trains that capability from scratch using GRPO — no human-labelled data, no correct-answer demonstrations. Just reward signals from the smart calendar RL environment itself.

---

## 🏗️ Environment Design

The SmartCalendar environment implements the full OpenEnv specification: a Dockerised FastAPI server with typed `reset()`, `step()`, and `state()` endpoints, accessed by a Python HTTP client.

### What the Agent Observes

At each step the agent receives a natural language prompt containing:

- Current objective and progress (`scheduled / target` meetings)
- List of free time slots in `HH:MM-HH:MM` format
- IDs of already-scheduled events
- Dependency graph showing which meetings are available vs locked
- Available commands and usage examples

The agent **never** sees the raw calendar matrix or internal slot indices. It must reason from the description alone.

### What the Agent Can Do

| Command | Description |
|---|---|
| `add_event` | Book a meeting at a specified slot |
| `search_slot` | Query which slots are free before committing |
| `move_event` | Reschedule an existing meeting |
| `delete_event` | Remove a meeting from the calendar |

All actions are submitted as JSON. Invalid JSON, impossible times, and occupied slots are rejected with a `rejection_reason` in the observation metadata.

### Task Difficulty Tiers

| Tier | Meetings | Days | Constraints | Max Steps | Training |
|---|---|---|---|---|---|
| Easy | 1 | Monday only | None | 10 | ✅ Yes |
| Medium | 3 | Mon–Wed | No overlaps | 20 | ✅ Yes |
| Hard | 7 | Mon–Fri | 1-hour gaps, dependency order | 30 | ✅ Yes |
| Super Hard | 11 | Mon–Fri | Full DAG, sparse reward | 40 | 🔬 Research |

The **Hard** task requires the agent to schedule 7 meetings in correct dependency order across a full working week with mandatory 1-hour gaps between all meetings. This is the primary benchmark.

### Reward Design (5 Independent Signals)

Rather than a single composite score, the environment returns five separate reward signals per step — following the composable rubric pattern:

| Signal | Measures | Range |
|---|---|---|
| `reward_objective_progress` | Meetings scheduled / target | 0.0 – 1.0 |
| `reward_valid_json` | Model output is parseable JSON | −1.0 – 0.3 |
| `reward_correct_time` | Times are valid and within working hours | −0.3 – 1.0 |
| `action_valid` | Action was accepted by the environment | bool |
| `rejection_reason` | Why an action was rejected (for debugging) | string / null |

The GRPO training loop uses `reward_valid_json`, `reward_correct_time`, and `reward_objective_progress` as three independent reward functions, creating multi-dimensional feedback that prevents reward hacking on any single signal.

---

## 🤖 Training

### Stack

| Component | Choice |
|---|---|
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` |
| RL algorithm | GRPO (Group Relative Policy Optimisation) |
| Training library | HuggingFace TRL |
| Environment | SmartCalendarRLDemo on HF Spaces |
| Training platform | Google Colab (T4 GPU) |

### Why GRPO, Not SFT

Supervised Fine-Tuning requires correct example answers. For calendar scheduling, generating correct examples programmatically would mean the model just memorises a rule-based heuristic rather than learning to reason. GRPO instead samples multiple outputs from the model, scores them against the environment's reward signals, and updates weights based on which outputs were better — no labelled data needed.

### Training Configuration

```python
GRPOConfig(
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,
    num_generations=4,      # 4 outputs per prompt, scored and contrasted
    max_completion_length=150,
    temperature=0.9,        # ensures output diversity between the 4 samples
)
```

Dataset: 150 prompts generated from the live environment across all three training difficulties (80 Easy / 50 Medium / 20 Hard), shuffled randomly.

---

## 📊 Results

### Training Evidence

The following plots were generated from the actual training run. All reward values are logged per step via `TrainerCallback` and saved to `training_history.json`.

![GRPO Training Progress](evidences\images\reward_and_loss_changes_during_training.png)

*Top: Three reward signals during 75 training steps. The green line (correct time logic) rises earliest, showing the model first learns output format. The red dashed line (scheduling success from the environment) rises from step ~30 onward, confirming the environment's reward signal is shaping behaviour. Bottom: Training loss fluctuates around non-zero values throughout, confirming weights were updated at each step.*

### Before vs After GRPO Training

![Base vs Trained Model](evidences\images\base_vs_trained_model_test_comparision.png)

*Top-left: Final objective progress score by difficulty. The trained model achieves 1.00 on Easy (base: 0.00) and 0.14 on Hard (base: 0.00). Top-right: The green shaded region shows improvement from RL training across all difficulty levels. Bottom-left: Invalid actions per episode — the trained model makes significantly fewer rejected actions on Easy tasks. Bottom-right: Step-by-step reward on the Hard task — the trained model briefly achieves non-zero reward at step 1, confirming it has learned to make at least one valid scheduling decision.*

### Key Numbers

| Metric | Base Qwen 0.5B | GRPO Trained | Change |
|---|---|---|---|
| Easy — final score | 0.00 | **1.00** | +1.00 |
| Medium — final score | 0.33 | **0.33** | 0.00 |
| Hard — final score | 0.00 | **0.14** | +0.14 |
| Easy — invalid actions | 10 | **1** | −9 |

The Easy task result — from 0.00 to 1.00 — is the clearest evidence of learning. The base model failed completely to schedule even a single meeting. After GRPO training, the model schedules the required meeting correctly every episode.

---

## 🔬 Research Direction: Super Hard Mode

The environment includes a fourth difficulty tier — **Sprint Planning** — featuring an 11-meeting dependency graph across a full sprint week with sparse delayed rewards. The full dependency chain is:

```
Kickoff → Requirements Review → Backend Design  ┐
                              → Frontend Design ┘→ Integration → QA Planning → Launch Review
Daily Standups (Mon–Thu, no dependencies)
```

This mode is fully implemented in `task_definitions.py` under `TaskDifficulty.SUPER_HARD` with `research_mode=True`. It was deliberately excluded from hackathon training because sparse rewards require longer runs (1000+ steps) to show measurable improvement — beyond what a single Colab session can produce. It represents a clear post-hackathon research direction.

---

## 🚀 Quick Start

### Run the Environment Locally

```bash
git clone https://github.com/bdnath1988/SmartCalendarRLDemo
cd SmartCalendarRLDemo
docker build -t smart_calendar_agent_env:latest .
docker run -p 8000:8000 smart_calendar_agent_env:latest
```

### Test with the Inference Script

```bash
cp .env.example .env
# Add your HF_TOKEN and MODEL_NAME to .env
python inference.py
```

### Run the Training Notebook

Open the Colab notebook and run all cells top to bottom. The notebook clones the environment from HuggingFace Spaces, installs dependencies, runs GRPO training, saves reward history, and plots results.

[▶ Open Training Notebook in Colab](https://colab.research.google.com/drive/1mVmrGGahOBqqT3g3VCGmPYKu8wwZDjg3?usp=sharing)

---

## 📁 Repository Structure

```
SmartCalendarRLDemo/
├── models.py                  # Type-safe Action, Observation, State models
├── client.py                  # HTTPEnvClient connecting to the environment
├── inference.py               # Agent loop: prompt builder, action parser, repair
├── task_definitions.py        # 4 difficulty tiers + dependency graph + curricula
├── server/
│   └── smart_calendar_agent_environment.py   # reset(), step(), state(), reward logic
├── plots/
│   ├── reward_and_loss_changes_during_training.png
│   └── base_vs_trained_model_test_comparision.png
├── Dockerfile
└── openenv.yaml
```

---

## 🔗 Links

| Resource | Link |
|---|---|
| 🤗 Environment on HF Spaces | https://huggingface.co/spaces/kohantika/SmartCalendarRLDemo |
| 🤗 Trained Model |https://huggingface.co/kohantika/smart-calendar-qwen-grpo |
| 📓 Training Colab Notebook | https://colab.research.google.com/drive/1mVmrGGahOBqqT3g3VCGmPYKu8wwZDjg3?usp=sharing |
| 📝 HuggingFace Blog Post | <!-- ADD BLOG LINK --> |
| 🎥 Demo Video | <!-- ADD VIDEO LINK --> |
| 💻 GitHub Repository | https://github.com/bdnath1988/SmartCalendarRLDemo |

---

