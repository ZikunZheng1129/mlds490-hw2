"""Behavioral checks for the saved-model validation workflow."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import validate


def rows(rewards, seeds=None):
    seeds = list(range(80000, 80000 + len(rewards))) if seeds is None else seeds
    return [
        {
            "episode": index + 1,
            "seed": seed,
            "reward": float(reward),
            "length": 500 if reward == 500 else 20,
            "terminated": reward != 500,
            "truncated": reward == 500,
            "max_q": 4.0,
        }
        for index, (seed, reward) in enumerate(zip(seeds, rewards, strict=True))
    ]


def test_paired_comparison_matches_seeds_instead_of_row_order():
    reference = rows([10, 20, 30])
    candidate = rows([34, 12, 26], [80002, 80000, 80001])
    result = validate.paired_comparison(reference, candidate, bootstrap_seed=17)
    assert result["mean_difference"] == pytest.approx(4.0)
    assert result["candidate_mean"] == pytest.approx(24.0)
    assert result["reference_mean"] == pytest.approx(20.0)
    assert result["n_pairs"] == 3
    assert result["ci95_low"] <= 4 <= result["ci95_high"]
    assert result == validate.paired_comparison(reference, candidate, bootstrap_seed=17)


@pytest.mark.parametrize("reference,candidate", [
    ([], []),
    (rows([1]), rows([2], [999])),
    (rows([1, 2], [1, 1]), rows([2, 3], [1, 1])),
    (rows([1]), rows([float("nan")])),
])
def test_comparison_rejects_evidence_that_cannot_support_pairing(reference, candidate):
    with pytest.raises(ValueError):
        validate.paired_comparison(reference, candidate)


def test_single_pair_does_not_invent_uncertainty():
    result = validate.paired_comparison(rows([8]), rows([3]))
    assert result["mean_difference"] == -5
    assert result["ci95_low"] == result["ci95_high"] == -5
    assert result["inference_note"]


def test_summary_counts_success_and_boundary_conditions():
    result = validate.summarize_evaluation(rows([500, 500, 20]))
    assert result["episodes"] == 3
    assert result["mean_reward"] == pytest.approx(340)
    assert result["std_reward"] == pytest.approx(np.std([500, 500, 20], ddof=1))
    assert result["terminated_count"] == 1
    assert result["truncated_count"] == 2
    assert result["length_500_fraction"] == pytest.approx(2 / 3)


def test_blackout_preserves_original_input_and_dtype():
    observation = np.arange(48, dtype=np.uint8).reshape(4, 3, 4)
    before = observation.copy()
    transformed = validate.blackout_observation(observation)
    np.testing.assert_array_equal(observation, before)
    assert transformed.shape == observation.shape
    assert transformed.dtype == np.uint8
    assert not np.shares_memory(transformed, observation)
    assert np.count_nonzero(transformed) == 0


def test_cartpole_grid_is_reproducible_and_covers_both_directions():
    first = validate.cartpole_initial_states(grid_size=3, repeats=2, seed=90)
    second = validate.cartpole_initial_states(grid_size=3, repeats=2, seed=90)
    assert first == second
    assert len(first) == 18
    assert {row["angle"] for row in first} == {-0.18, 0.0, 0.18}
    assert {row["angular_velocity"] for row in first} == {-1.5, 0.0, 1.5}
    assert len({row["seed"] for row in first}) == 18
    assert all(len(row["initial_state"]) == 4 for row in first)
    assert all(abs(row["initial_state"][2]) < 0.2 for row in first)
    assert first[0]["initial_state"] != first[1]["initial_state"]


def test_terminal_audit_does_not_call_unobserved_boundary_a_pass():
    audit = validate.audit_targets([
        {"reward": 1.0, "train_reward": 1.0, "next_max_q": 2.0,
         "target": 2.9, "terminated": False, "truncated": False}
    ], gamma=0.95)
    assert audit["all_checked_targets_match"]
    assert audit["terminal_status"] == "not_observed"
    assert audit["truncation_status"] == "not_observed"


def test_terminal_and_time_limit_targets_are_distinguished():
    audit = validate.audit_targets([
        {"train_reward": 1.0, "next_max_q": 100, "target": 1.0,
         "terminated": True, "truncated": False},
        {"train_reward": 1.0, "next_max_q": 2.0, "target": 2.9,
         "terminated": False, "truncated": True},
    ], gamma=0.95)
    assert audit["terminal_status"] == "passed"
    assert audit["truncation_status"] == "passed"
    assert audit["all_checked_targets_match"]
    broken = validate.audit_targets([
        {"train_reward": 1.0, "next_max_q": 2.0, "target": 1.0,
         "terminated": False, "truncated": True},
    ], gamma=0.95)
    assert broken["truncation_status"] == "failed"
    assert not broken["all_checked_targets_match"]


@pytest.mark.parametrize("task", ["cartpole", "mspacman"])
def test_saved_validation_workflow_records_real_evaluator_outputs(tmp_path, monkeypatch, task):
    evaluation_trace = [{"seed": 100007, "purpose": "Independent final evaluation evidence"}]
    evaluation_path = tmp_path / "first10_steps.json"
    evaluation_path.write_text(json.dumps(evaluation_trace), encoding="utf-8")
    original_trace_bytes = evaluation_path.read_bytes()
    selected = SimpleNamespace(config=SimpleNamespace(task=task, gamma=0.95 if task == "cartpole" else 0.99))
    initial = SimpleNamespace(config=selected.config)
    calls = []

    def load(path, device):
        return (initial if Path(path).name == "initial.pt" else selected), {"step": 42}

    def evaluate(agent, episodes, seed, epsilon=0.0, observation_transform=None,
                 initial_state=None, trace_steps=0):
        calls.append((agent, episodes, seed, epsilon, observation_transform, initial_state))
        reward = 500 if agent is selected else 20
        if epsilon == 1.0 or observation_transform is not None:
            reward = 10
        result = rows([reward] * episodes, list(range(seed, seed + episodes)))
        trace = [{"train_reward": 1.0, "next_max_q": 2.0,
                  "target": 1 + 2 * selected.config.gamma,
                  "terminated": False, "truncated": False}] if trace_steps else []
        return result, trace

    monkeypatch.setattr(validate, "_load_agent", load)
    monkeypatch.setattr(validate, "_evaluate", evaluate)
    result = validate.run_validation(task, tmp_path, episodes=3, grid_size=3, grid_repeats=1)
    saved = json.loads((tmp_path / "validation.json").read_text())
    assert saved == result
    assert saved["comparison"]["final_vs_random"]["mean_difference"] == 490
    assert saved["comparison"]["final_vs_initial"]["mean_difference"] == 480
    assert saved["final_policy"]["episodes"] == 3
    assert evaluation_path.read_bytes() == original_trace_bytes
    assert (tmp_path / "validation_first10_steps.json").exists()
    assert (tmp_path / "validation_final.csv").exists()
    assert (tmp_path / "validation_initial.csv").exists()
    assert (tmp_path / "validation_random.csv").exists()
    if task == "cartpole":
        assert (tmp_path / "validation_grid.csv").exists()
        assert saved["perturbation"]["grid_evaluations"] == 9
        assert len([call for call in calls if call[-1] is not None]) == 9
    else:
        assert (tmp_path / "validation_blackout.csv").exists()
        assert saved["perturbation"]["normal_minus_blackout"]["mean_difference"] == 490
        assert calls[-1][4] is validate.blackout_observation


@pytest.mark.parametrize("task,episodes,grid_size,repeats", [
    ("unknown", 3, 3, 1), ("cartpole", 0, 3, 1),
    ("cartpole", 3, 1, 1), ("cartpole", 3, 3, 0),
])
def test_invalid_validation_request_fails_before_loading(task, episodes, grid_size, repeats, tmp_path):
    with pytest.raises(ValueError):
        validate.run_validation(task, tmp_path, episodes, grid_size=grid_size, grid_repeats=repeats)


def test_real_cartpole_validation_preserves_frozen_checkpoint(tmp_path):
    from dqn import DQNAgent, defaults

    agent = DQNAgent((4,), 2, defaults("cartpole"), "cpu")
    agent.save(tmp_path / "policy.pt", {"purpose": "test only: untrained checkpoint"})
    agent.save(tmp_path / "initial.pt", {"purpose": "test only: untrained checkpoint"})
    before = hashlib.sha256((tmp_path / "policy.pt").read_bytes()).hexdigest()
    result = validate.run_validation("cartpole", tmp_path, episodes=2, grid_size=2, grid_repeats=1)
    assert result["comparison"]["final_vs_initial"]["mean_difference"] == 0
    assert result["perturbation"]["grid_evaluations"] == 4
    assert result["target_audit"]["all_checked_targets_match"]
    assert result["boundaries"]["all_episodes_report_an_end"]
    assert before == hashlib.sha256((tmp_path / "policy.pt").read_bytes()).hexdigest()
