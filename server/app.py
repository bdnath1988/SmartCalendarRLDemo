# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Smart Calendar Agent Environment.

This module creates an HTTP server that exposes the SmartCalendarAgentEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""
from fastapi import FastAPI, HTTPException

try:
    from .smart_calendar_agent_environment import CalendarEnv
except ImportError:
    from smart_calendar_agent_environment import CalendarEnv

try:
    from ..models import Action, Observation, State
except ImportError:
    from models import Action, Observation, State

app = FastAPI(title="OpenEnv Calendar API")
env = CalendarEnv()

@app.post("/reset", response_model=Observation)
def reset(task_id: int = 0):
    if task_id not in [0, 1, 2]:
        raise HTTPException(status_code=400, detail="Invalid Task ID")
    return env.reset(task_id)

@app.post("/step")
def step(action: Action):
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info
    }

@app.get("/state", response_model=State)
def get_state():
    return env.get_internal_state()