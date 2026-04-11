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

## Formal Specifications

### 1. State Space ($\mathcal{S}$)
The true environment state $\mathcal{S}$ is formally defined as:
$s_t = \langle \mathcal{C}, M_{target}, M_{current}, \tau, \Phi \rangle$
- $\mathcal{C}$: The calendar array containing 24 chronological slots, each $c_i \in \{0, 1\}$.
- $M_{target}$: Target number of meetings required by the task $T \in \{1, 3, 5\}$.
- $M_{current}$: Meetings currently successfully scheduled.
- $\tau$: Timestep count $\tau \in [0, 10]$.
- $\Phi$: Task-specific rules (e.g., minimum 1-hour spacing $\phi_{gap} \ge 1$).

### 2. Action Space ($\mathcal{A}$)
The agent selects an action $a_t \in \mathcal{A}$ parameterized as a tuple:
$a_t = \langle C, e, t_{start}, t_{end} \rangle$
Where the command $C \in \{add\_event, move\_event, delete\_event, search\_slot\}$, $e$ is an event ID, and $t$ represents timestamps.

### 3. Reward Function ($\mathcal{R}$)
$r_t = \mathcal{R}(s_t, a_t) \rightarrow [0.0, 1.0]$
- The reward is a composite of the immediate action success (50%) and total objective progress (50%) as measured by the deterministic Grader.
- **Valid operation** (Successful add/move/delete): Incremental reward up to $0.5$.
- **Objective Progress**: Progressive bonus up to $0.5$ based on the total task score.
- **Invalid/Redundant operations**: $0.0$ (Zero reward).

---

## 🛑 Reward Leakage & Privileged Information
To maintain evaluation integrity and prevent **Reward Leakage**, there is a strict separation between what the Agent observes and what the Grader knows.

**Agent Observables (Passed in prompt):**
- $M_{target}$ (Objective count) and current $M_{current}$ (Count of scheduled meetings).
- A textual list of currently `free_slots` (e.g., "09:00-10:00").
- IDs of existing scheduled `events`.
- The explicit task goal and objective strings.
- **Filtering:** The agent **never** sees the raw calendar matrix or the internal timestamp strings of occupied slots unless it has successfully scheduled them itself.

**Privileged Grader Information (Hidden from Agent):**
The external `Grader` class has direct access to the raw Python `Calendar` object matrix and the `task_definitions.py` metrics. It independently computes:
- **Exact Overlap Matrix:** Checks for sub-minute overlaps that might be obscured in the prompt's hourly labels.
- **Spacing Bound Violations:** Mathematically verifies that gaps (e.g., 1.0 hour) are strictly maintained.
- **Deterministic Score:** Computes $\Sigma \in [0,1]$ which is completely decoupled from the stepwise RL reward $r_t$. The agent does not see its current "Score" during the episode, only the "Reward" for the immediate action.

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