---
title: Smart Calendar Agent Environment
emoji: 📅
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Smart Calendar Agent Environment

A real-world task simulation environment designed to evaluate an agent's ability to schedule meetings, manage time slots, and maneuver calendar events efficiently and reliably.

## Motivation
Tool-use capabilities for LLMs must be tested on authentic, everyday tasks rather than generic toy datasets. Calendar triage and meeting scheduling are among the most standard workflow automation bottlenecks. This environment simulates these constraints by introducing state tracking, time availability parsing, collision detection, and reward shaping to encourage strategic searching prior to committing to an action.

## Tasks and Difficulty
The environment defines 3 standard tasks, assigned dynamically by difficulty (graders evaluate performance between `0.0` and `1.0`):

1. **Easy:** Schedule exactly 1 meeting. (Target: 1)
2. **Medium:** Schedule 3 meetings efficiently without overlapping any existing events. (Target: 3)
3. **Hard:** Schedule 5 meetings while mandating a minimum of 1-hour gaps between all meetings. (Target: 5)

## Action & Observation Spaces

### Action Space (MyCalendarAction)
Actions are structured Pydantic models with the following key fields under `ExpectedAction`:
- `command` (Literal): `add_event`, `move_event`, `delete_event`, `search_slot`.
- `slot` (Slot, Optional): Includes `start_time` and `end_time`.
- `event_id` (str, Optional): Unique identifier for the event.

### Observation Space (MyCalendarObservation)
The environment returns structured feedback tracking real-time status:
- `message` (str): Textual feedback representing success, failure, or warnings.
- `reward` (float): Current action's scalar reward.
- `done` (bool): Termination condition switch.
- `metadata` (dict): Additional internal structures (such as score mapping).

## Scoring & Rewards
- **Rewards**: Dynamic reward shaping based on trajectory. Finding free slots grants base rewards (e.g., `+0.2`), successfully adding a valid meeting yields proportional goal progress bonuses (e.g., `+1.0` to `+1.5`), and attempting to force a meeting into an occupied slot triggers an immediate stiff penalty (`-1.0`).
- **Score (0.0 to 1.0)**: Final grading metric based directly on task completion progress and conditional efficiency (like spacing gaps for Hard mode).

## Setup instructions

### 1. Build the Docker Image
The environment server fully executes inside a containerized sandbox. Recompiling the Dockerfile applies configuration updates locally:
```bash
docker build -t smart_calendar_agent_env:latest .
```

### 2. Environment Variables
Add your corresponding LLM API keys via the standard `.env` configuration mapping in your root directory:
```bash
MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
API_BASE_URL="https://router.huggingface.co/v1"
OPENAI_API_KEY="..." # Or HF_TOKEN="..."
```

### 3. Run Inference
Use the included OpenEnv baseline script to test the agent using the OpenAI SDK natively:
```bash
python inference.py
```

## Baseline Scores
Testing the baseline with `Qwen/Qwen2.5-72B-Instruct` natively completes the task deterministically, correctly recognizing collision edge cases due to the updated `search_slot` instructions, and efficiently achieves a perfectly reproducible optimal score of `1.000` consistently across Easy, Medium, and Hard workloads.