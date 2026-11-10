"""Report checks using explicitly synthetic fixtures in pytest temporary paths."""
import csv
import importlib
import json
from pathlib import Path

import numpy as np
import pytest


def save_json(path, value):
    path.write_text(json.dumps(value))


def save_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def evidence(tmp_path):
    """Synthetic fixture only: never presented as measured assignment results."""
    from dataclasses import asdict
    from dqn import defaults
    from train import summarize
    from validate import paired_comparison, summarize_evaluation
    root = tmp_path / "synthetic_results"
    for task in ("cartpole", "mspacman"):
        folder = root / task
        folder.mkdir(parents=True)
        config = {**asdict(defaults(task)), "test_fixture": True,
                  "requested_episodes": 120, "eval_every": 50,
                  "development_episodes": 10, "final_episodes": 500,
                  "evaluation_seed": 100007, "evaluation_epsilon": 0,
                  "runtime": {"hostname": "synthetic-fixture", "gpu": "test GPU",
                              "device": "cuda:0", "cuda_visible_devices": "2",
                              "python": "3.12", "torch": "test", "gymnasium": "test",
                              "ale_py": "test", "numpy": "test"}}
        save_json(folder / "config.json", config)
        training = [{"episode": i + 1, "reward": 20 + i, "max_q": i / 10,
                     "steps": (i+1)*100, "length": 100, "epsilon": 0.8,
                     "loss": 0.1, "td_abs": 0.2} for i in range(120)]
        save_csv(folder / "training.csv", training)
        def rows(n, offset, seed=80000):
            return [{"episode": i+1, "seed": seed+i, "reward": offset + i % 5,
                     "length": 100, "terminated": True, "truncated": False,
                     "max_q": 12} for i in range(n)]
        evaluated = rows(500, 100, 100007)
        save_csv(folder / "evaluation.csv", evaluated)
        save_csv(folder / "development.csv", [{"training_episode": 50, "mean": 80},
                                                {"training_episode": 100, "mean": 100}])
        save_json(folder / "metrics.json", {
            "training_episodes": 120, "environment_steps": 12000,
            "optimizer_steps": 3000, "wall_seconds": 1200,
            "selected_checkpoint": {"episode": 100, "development_mean": 100},
            "evaluation": summarize(evaluated), "success_rate": 0})
        final, initial, random = rows(10, 100), rows(10, 20), rows(10, 30)
        for label, outcomes in [("final", final), ("initial", initial), ("random", random)]:
            save_csv(folder / f"validation_{label}.csv", outcomes)
        summary = {
            "task": task,
            "sampling": {"episodes_per_condition": 10, "first_seed": 80000},
            "final_policy": summarize_evaluation(final),
            "initial_policy": summarize_evaluation(initial),
            "random_policy": summarize_evaluation(random),
            "comparison": {"final_vs_random": paired_comparison(random, final),
                           "final_vs_initial": paired_comparison(initial, final)},
            "target_audit": {"transitions_checked": 10, "all_checked_targets_match": True,
                             "terminal_status": "not_observed", "truncation_status": "not_observed"},
            "boundaries": {"observed_terminated": 10, "observed_truncated": 0,
                           "all_episodes_report_an_end": True},
            "limitations": ["Synthetic test fixture."]}
        if task == "cartpole":
            grid = [{"angle": a, "angular_velocity": v, "repeat": 0,
                     "success": v == 0, "length": 500 if v == 0 else 10}
                    for a in [-0.1, 0.1] for v in [-1, 0]]
            save_csv(folder / "validation_grid.csv", grid)
            summary["perturbation"] = {"grid_size": 2, "repeats_per_cell": 1,
                "grid_evaluations": 4, "overall_success_fraction": 0.5, "mean_length": 255}
        else:
            blackout = rows(10, 40)
            save_csv(folder / "validation_blackout.csv", blackout)
            summary["perturbation"] = {"normal_minus_blackout": paired_comparison(blackout, final),
                                        "blackout_policy": summarize_evaluation(blackout)}
        save_json(folder / "validation.json", summary)
        save_json(folder / "validation_first10_steps.json", [{
            "step": i+1, "action": 0, "reward": 0, "train_reward": 0,
            "next_max_q": 10, "target": config["gamma"] * 10,
            "terminated": False, "truncated": False, "q_values": [1, 2],
            "state_summary": [0, 0, 0, 0]} for i in range(10)])
    return root


def test_ma_requires_full_window():
    report = importlib.import_module("build_report")
    actual = report.moving_average(np.arange(102), 100)
    assert np.isnan(actual[:99]).all()
    assert actual[99:].tolist() == pytest.approx([49.5, 50.5, 51.5])
    with pytest.raises(ValueError):
        report.moving_average([1, 2], 0)


def test_missing_evidence_has_clear_error(tmp_path):
    report = importlib.import_module("build_report")
    with pytest.raises(ValueError, match="Missing required result"):
        report.load_run(tmp_path, "cartpole")


def test_load_validates_required_500_episodes(evidence):
    report = importlib.import_module("build_report")
    run = report.load_run(evidence, "cartpole")
    assert len(run.evaluation) == 500
    path = evidence / "cartpole" / "evaluation.csv"
    path.write_text("episode,seed,reward,length,terminated,truncated\n1,1,1,1,True,False\n")
    with pytest.raises(ValueError, match="500"):
        report.load_run(evidence, "cartpole")


def test_learning_language_does_not_invent_success():
    report = importlib.import_module("build_report")
    assert "does not establish" in report.learning_statement({"ci95_low": -3, "ci95_high": 4})
    assert "positive" in report.learning_statement({"ci95_low": 2, "ci95_high": 4})


def test_reproduction_commands_use_recorded_parameters(evidence):
    from dataclasses import replace
    report = importlib.import_module("build_report")
    run = report.load_run(evidence, "cartpole")
    modified = replace(run, config={**run.config, "seed": 99, "learning_rate": 0.0007,
                                    "epsilon_decay_steps": 321, "early_stop": False})
    command = report.training_command(modified)
    for flag in ("--seed 99", "--learning-rate 0.0007", "--epsilon-decay-steps 321", "--no-early-stop"):
        assert flag in command


def test_reproduction_uses_new_run_for_training_validation_and_report(evidence):
    from reportlab.platypus import Preformatted
    report = importlib.import_module("build_report")
    runs = [report.load_run(evidence, task) for task in ("cartpole", "mspacman")]
    commands = "\n".join("\n".join(item.lines) for item in report.reproducibility(*runs)
                          if isinstance(item, Preformatted))
    assert "python3.12 -m venv .venv" in commands
    assert "source .venv/bin/activate" in commands
    assert "torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126" in commands
    for task in ("cartpole", "mspacman"):
        assert f"--output results/rerun/{task}" in commands
        assert f"--run results/rerun/{task}" in commands
        assert f"--run results/{task} " not in commands
    assert "--results results/rerun" in commands
    assert "--output hw2_report_rerun.pdf" in commands


def test_report_builds_complete_pdf_only_from_supplied_evidence(evidence, tmp_path):
    report = importlib.import_module("build_report")
    result = report.build_report(evidence, tmp_path / "synthetic_report.pdf",
                                 "https://github.com/ZikunZheng1129/mlds490-hw2")
    assert result.exists() and result.stat().st_size > 20000
    # PDF parser is available from the bundled authoring runtime.
    from pypdf import PdfReader
    reader = PdfReader(result)
    assert 6 <= len(reader.pages) <= 7
    text = "\n".join(page.extract_text() for page in reader.pages)
    for phrase in ["SYNTHETIC TEST FIXTURE", "500", "three months", "ZikunZheng1129",
                   "Maximum Q", "CartPole", "MsPacman", "not_observed"]:
        assert phrase in text
    assert len(list(evidence.glob("*/figures/*.png"))) == 8
