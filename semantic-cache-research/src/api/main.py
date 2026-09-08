"""
src/api/main.py

FastAPI application that exposes the experiment runner over HTTP so the
browser UI (ui/index.html) can control and observe runs without a
WebSocket or SSE — it polls /status instead.

All routes live in this one file (no router split) because the surface
area is small and a solo-researcher project at this scale doesn't
benefit from the indirection.

Routes
------
POST /run
    Start a new experiment run.
    Body: { "config": "<filename inside configs/>", "run_id": "<optional>" }
    Returns: { "run_id": str, "status": "started" }

GET /status/{run_id}
    Poll for the current status of a run.
    Returns: { "run_id": str, "status": "running"|"done"|"error",
               "progress": int, "total": int, "error": str|null }

GET /runs
    List all known run IDs and their status.
    Returns: [ { "run_id": str, "config": str, "status": str }, ... ]

GET /compare?run_a=<id>&run_b=<id>
    Load summary metrics for two runs and return them side-by-side.
    Returns: { "run_a": { metrics… }, "run_b": { metrics… } }

Usage
-----
    uvicorn src.api.main:app --reload --port 8000
"""

import json
import uuid
import threading
from pathlib import Path
from typing import Optional

# TODO: pip install fastapi uvicorn
# from fastapi import FastAPI, HTTPException, BackgroundTasks
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel

# ---------------------------------------------------------------------------
# App bootstrap
# ---------------------------------------------------------------------------

# app = FastAPI(title="Semantic Cache Research API", version="0.1.0")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],   # local dev only — tighten for production
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# ---------------------------------------------------------------------------
# In-memory run registry (sufficient for local research use)
# ---------------------------------------------------------------------------

_runs: dict[str, dict] = {}
# Schema per entry:
# {
#   "run_id":         str,
#   "config":         str,            # filename from configs/
#   "match_strategy": str,            # strategy used for this run
#   "status":         "running" | "done" | "error",
#   "progress":       int,            # queries processed so far
#   "total":          int,            # total queries in this run
#   "error":          str | None,
# }

RESULTS_DIR = Path("results")

# ---------------------------------------------------------------------------
# Request / response models (Pydantic)
# ---------------------------------------------------------------------------

# class StartRunRequest(BaseModel):
#     config: str                                   # e.g. "threshold_sweep.yaml"
#     match_strategy: str = "cosine_threshold"      # one of STRATEGIES keys
#     run_id: Optional[str] = None

# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

def _run_experiment_worker(run_id: str, config_path: str, match_strategy: str) -> None:
    """
    Runs in a background thread spawned by POST /run.

    Imports experiment_runner inline to avoid circular imports at module
    load time and to make the worker self-contained for future
    process-based parallelism.
    """
    try:
        _runs[run_id]["status"] = "running"
        # TODO:
        # from src.experiment_runner import load_config, run_experiment
        # config = load_config(config_path)
        # run_experiment(
        #     config,
        #     match_strategy=match_strategy,
        #     progress_callback=lambda done, total: _runs[run_id].update(
        #         {"progress": done, "total": total}
        #     ),
        # )
        _runs[run_id]["status"] = "done"
    except Exception as exc:
        _runs[run_id]["status"] = "error"
        _runs[run_id]["error"] = str(exc)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

# @app.post("/run")
# def start_run(req: StartRunRequest, background_tasks: BackgroundTasks):
#     """Start a new experiment run in a background thread."""
#     run_id = req.run_id or str(uuid.uuid4())[:8]
#     if run_id in _runs:
#         raise HTTPException(409, detail=f"run_id '{run_id}' already exists")
#
#     config_path = Path("configs") / req.config
#     if not config_path.exists():
#         raise HTTPException(404, detail=f"Config '{req.config}' not found in configs/")
#
#     _runs[run_id] = {
#         "run_id":         run_id,
#         "config":         req.config,
#         "match_strategy": req.match_strategy,
#         "status":         "queued",
#         "progress":       0,
#         "total":          0,
#         "error":          None,
#     }
#     background_tasks.add_task(
#         _run_experiment_worker, run_id, str(config_path), req.match_strategy
#     )
#     return {"run_id": run_id, "status": "started"}


# @app.get("/status/{run_id}")
# def get_status(run_id: str):
#     """Poll for progress and status of a specific run."""
#     if run_id not in _runs:
#         raise HTTPException(404, detail=f"No run with id '{run_id}'")
#     return _runs[run_id]


# @app.get("/runs")
# def list_runs():
#     """Return all known runs."""
#     return list(_runs.values())


# @app.get("/compare")
# def compare_runs(run_a: str, run_b: str):
#     """
#     Load the summary metrics JSON for two completed runs and return
#     them merged into a single response for the Compare tab.
#     """
#     def _load_summary(run_id: str) -> dict:
#         # Convention: results/<config_name>/<run_id>/summary.json
#         if run_id not in _runs:
#             raise HTTPException(404, detail=f"No run with id '{run_id}'")
#         run_meta = _runs[run_id]
#         config_name = run_meta["config"].replace(".yaml", "")
#         summary_path = RESULTS_DIR / config_name / run_id / "summary.json"
#         if not summary_path.exists():
#             raise HTTPException(404, detail=f"Summary not found for run '{run_id}'")
#         return json.loads(summary_path.read_text())
#
#     return {
#         "run_a": _load_summary(run_a),
#         "run_b": _load_summary(run_b),
#     }


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Run with: python -m src.api.main
    # Or:       uvicorn src.api.main:app --reload --port 8000
    pass
    # import uvicorn
    # uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
