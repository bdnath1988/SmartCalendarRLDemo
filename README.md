---
title: SmartCalendar RL Environment Server
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---
# SmartCalendar RL - AI Executive Assistant Environment

Meta OpenEnv Hackathon India 2026

SmartCalendarRLDemo is an OpenEnv-compatible reinforcement learning environment for training an agent to schedule meetings efficiently. The environment exposes a Dockerized FastAPI server with typed `reset()`, `step()`, and `state()` endpoints, plus a Python HTTP client.

## Links

| Resource | Link |
| --- | --- |
| Environment on HF Spaces | https://huggingface.co/spaces/kohantika/SmartCalendarRLDemo |
| Trained Model | https://huggingface.co/kohantika/smart-calendar-qwen-grpo |
| Training Colab Notebook | https://colab.research.google.com/drive/1mVmrGGahOBqqT3g3VCGmPYKu8wwZDjg3?usp=sharing |
| Demo Video | https://youtu.be/bdGxckCcu10 |
| GitHub Repository | https://github.com/bdnath1988/SmartCalendarRLDemo |

## Problem

LLMs are good at answering questions, but they are weak at acting inside a calendar: understanding free slots, respecting scheduling constraints, recovering from failed actions, and making progress across multiple steps.

This environment trains that capability using GRPO. The model takes actions, receives reward feedback from the calendar environment, and improves from those signals without human-labelled demonstrations.

## What the Agent Observes

At each step the agent receives a natural language prompt containing:

- Current objective and progress (`scheduled / target` meetings)
- Free time slots in `HH:MM-HH:MM` format
- IDs of already scheduled events
- Dependency graph state
- Available commands and usage examples

The agent never sees the raw calendar matrix or internal slot indices.

## Actions

| Command | Description |
| --- | --- |
| `add_event` | Book a meeting at a specified slot |
| `search_slot` | Query free slots before committing |
| `move_event` | Reschedule an existing meeting |
| `delete_event` | Remove a meeting from the calendar |

All actions are submitted as JSON. Invalid JSON, impossible times, and occupied slots are rejected with a `rejection_reason`.

## Task Difficulty Tiers

| Tier | Meetings | Days | Constraints | Max Steps | Training |
| --- | --- | --- | --- | --- | --- |
| Easy | 1 | Monday only | None | 10 | Yes |
| Medium | 3 | Mon-Wed | No overlaps | 20 | Yes |
| Hard | 7 | Mon-Fri | 1-hour gaps, dependency order | 30 | Yes |
| Super Hard | 11 | Mon-Fri | Full DAG, sparse reward | 40 | Research |

The Hard task is the primary benchmark. It requires the agent to schedule 7 meetings in dependency order across a full working week with mandatory 1-hour gaps.

## Reward Design

The environment returns five reward-related signals per step:

| Signal | Measures | Range |
| --- | --- | --- |
| `reward_objective_progress` | Meetings scheduled / target | 0.0 to 1.0 |
| `reward_valid_json` | Model output is parseable JSON | -1.0 to 0.3 |
| `reward_correct_time` | Times are valid and within working hours | -0.3 to 1.0 |
| `action_valid` | Action accepted by the environment | bool |
| `rejection_reason` | Why an action was rejected | string / null |

The GRPO training loop uses `reward_valid_json`, `reward_correct_time`, and `reward_objective_progress` as independent reward functions.

## Training

| Component | Choice |
| --- | --- |
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` |
| RL algorithm | GRPO |
| Training library | HuggingFace TRL |
| Environment | SmartCalendarRLDemo on HF Spaces |
| Training platform | Google Colab T4 GPU |

Example configuration:

```python
GRPOConfig(
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,
    num_generations=4,
    max_completion_length=150,
    temperature=0.9,
)
```

Dataset: 150 prompts generated from the live environment across Easy, Medium, and Hard difficulties.

## Results

### Training Evidence

The following plots were generated from the training run and are included in this repository.

![GRPO training reward and loss curves](https://huggingface.co/spaces/kohantika/SmartCalendarRLDemo/resolve/main/evidences/images/reward_and_loss_changes_during_training.png)

![Base model vs GRPO trained model comparison](https://huggingface.co/spaces/kohantika/SmartCalendarRLDemo/resolve/main/evidences/images/base_vs_trained_model_test_comparision.png)

| Metric | Base Qwen 0.5B | GRPO Trained | Change |
| --- | --- | --- | --- |
| Easy final score | 0.00 | 1.00 | +1.00 |
| Medium final score | 0.33 | 0.33 | 0.00 |
| Hard final score | 0.00 | 0.14 | +0.14 |
| Easy invalid actions | 10 | 1 | -9 |

The Easy task result, from 0.00 to 1.00, is the clearest evidence of learning. The base model failed to schedule a single meeting, while the GRPO-trained model schedules the required meeting correctly.

## Quick Start

```bash
git clone https://github.com/bdnath1988/SmartCalendarRLDemo
cd SmartCalendarRLDemo
docker build -t smart_calendar_agent_env:latest .
docker run -p 8000:8000 smart_calendar_agent_env:latest
```

Run inference:

```bash
cp .env.example .env
# Add HF_TOKEN and MODEL_NAME to .env
python inference.py
```

Run tests:

```bash
python -m pytest
```

## Repository Structure

```text
SmartCalendarRLDemo/
|-- models.py
|-- client.py
|-- inference.py
|-- task_definitions.py
|-- server/
|   |-- smart_calendar_agent_environment.py
|   |-- app.py
|-- tests/
|-- Dockerfile
|-- openenv.yaml
```
