"""Smoke tests for scripts/ and the committed configs.

Nothing imported scripts/ before, so a broken CLI or a config drifting away from
its generator would only surface when someone ran the careful battery.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core.config import RunConfig

REPO = Path(__file__).resolve().parent.parent


def test_committed_configs_match_their_generator() -> None:
    """configs/*.json must be exactly what make_configs produces."""
    from scripts.make_configs import build_configs

    generated = build_configs()
    on_disk = {p.stem for p in (REPO / "configs").glob("*.json")}
    assert set(generated) <= on_disk
    for stem, cfg in generated.items():
        committed = json.loads((REPO / "configs" / f"{stem}.json").read_text("utf-8"))
        assert committed == cfg.model_dump(mode="json"), stem


def test_every_committed_config_validates() -> None:
    for path in (REPO / "configs").glob("*.json"):
        RunConfig.model_validate_json(path.read_text("utf-8"))


def test_heat_and_strong_scenarios_share_one_coupling_ratio() -> None:
    """The pair is only a saturation control if the ratios match (D14)."""
    load = lambda n: RunConfig.model_validate_json(
        (REPO / "configs" / f"{n}.json").read_text("utf-8")).data
    heat, strong = load("scenario_heat_dead"), load("scenario_strong_dead")
    heat_ratio = next(z.gain for z in heat.zones if z.kind == "heat") / heat.background_gain
    strong_ratio = (next(z.gain for z in strong.zones if z.kind == "strong")
                    / strong.background_gain)
    assert heat_ratio == strong_ratio == 2.0
    assert heat.clipping_is_reachable() and not strong.clipping_is_reachable()


def test_run_matrix_dry_run_lists_no_duplicate_cells(capsys) -> None:
    from scripts.run_matrix import build_cells, main
    from storage.io import load_run_config

    assert main(["--dry-run"]) == 0
    base = load_run_config(REPO / "configs" / "scenario_heat_dead.json")
    cells = build_cells(base, {"pixel_noise", "jitter", "features", "solver"})
    signatures = [c.config.model_copy(update={"data": c.config.data.model_copy(
        update={"name": ""})}).model_dump_json() for c in cells]
    assert len(signatures) == len(set(signatures))


def test_run_experiment_cli_end_to_end(tmp_path: Path) -> None:
    """A tiny config through the real CLI, including writing every artefact."""
    from scripts.run_experiment import main

    cfg = RunConfig.model_validate_json(
        (REPO / "configs" / "control_simple.json").read_text("utf-8"))
    cfg.data.plane.height = cfg.data.plane.width = 20
    cfg.sampling.grid_step = 5
    cfg.training.n_per_class = 30
    cfg.training.cv_folds = 0
    cfg.evaluation.contrast_grid = np.linspace(0, 1, 5).tolist()
    cfg.evaluation.n_per_contrast = 20
    path = tmp_path / "tiny.json"
    path.write_text(cfg.model_dump_json(), encoding="utf-8")

    assert main([str(path), "--results-dir", str(tmp_path / "r"),
                 "--models-dir", str(tmp_path / "m")]) == 0
    assert list((tmp_path / "r").glob("*__trials.parquet"))
    assert list((tmp_path / "m").glob("*.joblib"))
    # A second invocation must add to, not overwrite, the summary table.
    cfg.master_seed = 1
    path.write_text(cfg.model_dump_json(), encoding="utf-8")
    main([str(path), "--results-dir", str(tmp_path / "r"),
          "--models-dir", str(tmp_path / "m")])
    import pandas as pd
    assert len(pd.read_csv(tmp_path / "r" / "summary_table.csv")) == 2
