# SmartCalendar-v2 — Project Conventions

Project: SmartCalendar Executive Assistant RL Environment
Hackathon: Meta OpenEnv India 2026
Stack: Python 3.11, FastAPI, Uvicorn, OpenEnv, TRL, Unsloth
Model: Qwen/Qwen3-1.7B
Training: GRPO via HuggingFace TRL
Themes: 3.2 Personalized Tasks + 2 Long-Horizon Planning

## Rules (always follow)
- Type hints on every function signature
- Google docstrings on every class and method
- logging not print()
- Black formatter, mypy strict compatible
- Never import server internals from client.py
- Never combine reward signals into one composite score
- Agent observation never contains raw calendar matrix
- Store all times in UTC internally
- Use zoneinfo for timezone conversion (no extra packages)

## Attendee Personas (fixed)
Alice: CEO, priority=1, IST (Asia/Kolkata), prefers 10am-4pm local
Bob:   Director, priority=2, GMT (Europe/London), prefers 9am-5pm local
Carol: Manager, priority=3, PDT (America/Los_Angeles), prefers 9am-6pm local

## Task Difficulties
EASY:       1 meeting, 1 attendee, today, 10 steps max
MEDIUM:     3 meetings, 2 attendees, this week, 20 steps max
HARD:       7 meetings, dep graph, all attendees, 30 steps max
SUPER_HARD: 12 meetings, full sprint DAG, sparse reward — research only

## Curriculum for hackathon training
HACKATHON_CURRICULUM = [EASY, MEDIUM, HARD]
SUPER_HARD is built but NOT used in hackathon training

## File ownership
models.py    → extend only, never delete existing fields
client.py    → NEVER touch
Dockerfile   → NEVER touch unless asked
CLAUDE.md    → update if conventions change