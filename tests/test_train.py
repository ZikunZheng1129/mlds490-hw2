"""Integration and command-line checks for the complete DQN workflow."""
import importlib
import json
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest


def test_statistics_and_serialization(tmp_path):
    train = importlib.import_module("train")
    rows = [{"episode": 1, "reward": 2.0}, {"episode": 2, "reward": 4.0}]
    summary = train.summarize(rows)
    assert summary["mean"] == 3.0
    assert summary["std"] == pytest.approx(np.sqrt(2.0))
    train.write_csv(tmp_path / "nested" / "scores.csv", rows)
    train.write_json(tmp_path / "stats.json", summary)
    assert json.loads((tmp_path / "stats.json").read_text())["n"] == 2
    with pytest.raises(ValueError):
        train.summarize([])


def test_summary_distinguishes_failure_from_time_limit():
    train = importlib.import_module("train")
    rows = [{"reward": 500, "length": 500, "terminated": True, "truncated": True},
            {"reward": 500, "length": 500, "terminated": False, "truncated": True}]
    stats = train.summarize(rows)
    assert stats["terminated_episodes"] == 1
    assert stats["truncated_episodes"] == 2


def test_frozen_evaluation_and_reproducible_trace():
    train = importlib.import_module("train")
    from dqn import DQNAgent, defaults
    import torch
    torch.manual_seed(42)
    agent = DQNAgent((4,), 2, defaults("cartpole"), "cpu")
    before = {key: value.clone() for key, value in agent.online.state_dict().items()}
    rows, trace = train.evaluate_model(agent, 2, 123, trace_steps=10)
    repeat, _ = train.evaluate_model(agent, 2, 123)
    assert rows == repeat
    assert len(trace) == 10
    assert len(rows) == 2
    assert all(row["length"] <= 500 for row in rows)
    assert all(torch.equal(before[key], value) for key, value in agent.online.state_dict().items())
    assert all("target" in step and "terminated" in step for step in trace)


def test_trace_audits_the_same_target_function_used_for_training(monkeypatch):
    train = importlib.import_module("train")
    import dqn
    import torch
    agent = dqn.DQNAgent((4,), 2, dqn.defaults("cartpole"), "cpu")
    monkeypatch.setattr(dqn, "bellman_targets", lambda *args: torch.tensor([123.0]))
    row = train.trace_record(agent, np.zeros(4), np.ones(4), np.array([1., 2.]),
                             1, 1.0, False, False, 1)
    assert row["target"] == 123.0


def test_tiny_training_to_checkpoint_to_evaluation(tmp_path):
    train = importlib.import_module("train")
    from dqn import DQNAgent, defaults
    config = replace(defaults("cartpole", seed=17), learning_starts=4,
                     batch_size=4, buffer_size=64, target_interval=2)
    result = train.run_training(config, episodes=2, output=tmp_path / "run",
                                device="cpu", eval_every=1, eval_episodes=2,
                                final_episodes=3, early_stop=False)
    assert result["training_episodes"] == 2
    assert result["evaluation"]["n"] == 3
    assert result["optimizer_steps"] > 0
    agent, metadata = DQNAgent.load(tmp_path / "run" / "policy.pt", "cpu")
    assert metadata["episode"] in (1, 2)
    assert len(train.evaluate_model(agent, 1, 999)[0]) == 1
    for name in ["training.csv", "evaluation.csv", "config.json", "metrics.json",
                 "initial.pt", "latest.pt", "policy.pt", "development.csv"]:
        assert (tmp_path / "run" / name).exists()
    with pytest.raises(FileExistsError):
        train.run_training(config, 1, tmp_path / "run", device="cpu")


def test_cli_rejects_invalid_episode_count(tmp_path):
    script = __import__("pathlib").Path(__file__).parents[1] / "train.py"
    result = subprocess.run([sys.executable, str(script), "--task", "cartpole",
                             "--episodes", "0", "--output", str(tmp_path / "run")],
                            text=True, capture_output=True)
    assert result.returncode != 0
    assert "positive" in result.stderr.lower()


def test_evaluation_cli_preserves_saved_results(tmp_path):
    script = __import__("pathlib").Path(__file__).parents[1] / "train.py"
    original = '{"training_evidence": true}\n'
    (tmp_path / "metrics.json").write_text(original)
    result = subprocess.run([sys.executable, str(script), "--evaluate", "missing.pt",
                             "--output", str(tmp_path)], text=True, capture_output=True)
    assert result.returncode != 0
    assert "fresh output directory" in result.stderr
    assert (tmp_path / "metrics.json").read_text() == original
