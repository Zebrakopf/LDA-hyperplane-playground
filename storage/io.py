"""Persistence: configs as JSON, models as joblib, trial tables as parquet.

Responsibility
--------------
Content-hash configs into stable identifiers, and read/write artefacts so a run
can be reproduced from its own metadata.

Explicitly NOT this module's job
--------------------------------
Any science. Nothing here decides anything about the experiment.

Determinism note (CLAUDE.md §7)
-------------------------------
`provenance` carries a wall-clock timestamp, so the JSON sidecar is NOT
byte-stable across runs. The trial parquet and every diagnostic ARE. The
determinism test therefore compares trial tables, not sidecars.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from pydantic import BaseModel

from core.config import DataConfig, RunConfig, SamplingConfig
from core.evaluation import ExperimentResult

HASH_LENGTH = 12          # hex chars kept from each sha256; collision risk negligible
REPO_ROOT = Path(__file__).resolve().parent.parent


def config_hash(config: BaseModel) -> str:
    """Stable content hash of a pydantic config.

    Uses the JSON dump with sorted keys so field order in the model definition
    cannot change the hash.
    """
    payload = json.dumps(config.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:HASH_LENGTH]


def _hash_parts(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:HASH_LENGTH]


def run_identifiers(cfg: RunConfig) -> dict[str, str]:
    """The three identifiers that key every artefact (plan.md §7.1).

    `run_id` is deterministic, so re-running the same experiment overwrites
    rather than accumulating near-duplicate result files.
    """
    data_id = config_hash(cfg.data)
    sampling_id = config_hash(cfg.sampling)
    training_id = config_hash(cfg.training)
    model_id = _hash_parts(data_id, sampling_id, training_id, str(cfg.master_seed))
    return {
        "data_config_id": data_id,
        "sampling_config_id": sampling_id,
        "model_id": model_id,
        "run_id": _hash_parts(model_id, config_hash(cfg.evaluation),
                              str(cfg.master_seed)),
    }


def provenance() -> dict[str, str]:
    """Everything needed to know which code produced a result."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            capture_output=True, text=True, timeout=5, check=False,
        ).stdout.strip() or "not-a-git-repo"
    except (OSError, subprocess.SubprocessError):
        commit = "unknown"
    return {
        "git_commit": commit,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }


def save_config(config: BaseModel, directory: Path, name: str | None = None) -> Path:
    """Write a config as pretty JSON. Returns the path written."""
    directory.mkdir(parents=True, exist_ok=True)
    stem = name or getattr(config, "name", None) or config_hash(config)
    path = directory / f"{stem}.json"
    path.write_text(json.dumps(config.model_dump(mode="json"), indent=2) + "\n",
                    encoding="utf-8")
    return path


def load_run_config(path: Path) -> RunConfig:
    """Read a RunConfig from JSON, with full pydantic validation."""
    return RunConfig.model_validate_json(Path(path).read_text(encoding="utf-8"))


def load_data_config(path: Path) -> DataConfig:
    return DataConfig.model_validate_json(Path(path).read_text(encoding="utf-8"))


def load_sampling_config(path: Path) -> SamplingConfig:
    return SamplingConfig.model_validate_json(Path(path).read_text(encoding="utf-8"))


def save_model(result: ExperimentResult, directory: Path) -> Path:
    """Persist the fitted model under its `model_id`."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{result.model_id}.joblib"
    joblib.dump(
        {
            "model": result.model,
            "config": result.config.model_dump(mode="json"),
            "provenance": result.provenance,
        },
        path,
    )
    return path


def load_model(path: Path) -> dict[str, Any]:
    """Load a persisted model bundle: keys `model`, `config`, `provenance`."""
    return joblib.load(path)


def save_result(result: ExperimentResult, directory: Path) -> dict[str, Path]:
    """Write the trial table, the diagnostics summary and the sidecar.

    Returns
    -------
    dict of the paths written, keyed `trials`, `summary`, `sidecar`.
    """
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{result.config.data.name}__{result.run_id}"

    trials_path = directory / f"{stem}__trials.parquet"
    result.trials.to_parquet(trials_path, index=False)

    summary = {
        "run_id": result.run_id,
        "model_id": result.model_id,
        "data_config": result.config.data.name,
        "sampling_config": result.config.sampling.name,
        "master_seed": result.config.master_seed,
        **result.report.summary_row(),
    }
    if result.oracle_report is not None:
        summary.update({f"oracle_{k}": v
                        for k, v in result.oracle_report.summary_row().items()})
    summary_path = directory / f"{stem}__summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n",
                            encoding="utf-8")

    sidecar = {
        "run_id": result.run_id,
        "model_id": result.model_id,
        "config": result.config.model_dump(mode="json"),
        "provenance": result.provenance,
        "training_diagnostics": vars(result.model.diagnostics),
        "jitter_columns": [c for c in result.trials.columns if c.startswith("jit_")],
        "n_features": result.model.diagnostics.n_features,
        "unique_pixels_sampled": result.composition.attrs.get("unique_pixels"),
        "sampled_area_composition": result.composition.to_dict(orient="records"),
        "weight_attribution": {
            "abs_correlation": result.attribution.abs_correlation,
            "sign_agreement": result.attribution.sign_agreement,
            "by_kind": result.attribution.mean_signed_weight_by_kind
                       .to_dict(orient="records"),
        },
    }
    sidecar_path = directory / f"{stem}__meta.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2, default=str) + "\n",
                            encoding="utf-8")
    return {"trials": trials_path, "summary": summary_path, "sidecar": sidecar_path}
