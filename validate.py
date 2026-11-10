"""Reproducible, task-specific experiments on frozen DQN checkpoints.

These experiments supplement software tests. They measure learned behavior on
held-out seeds and deliberately changed task conditions without retraining.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _reward_by_seed(rows):
    if not rows:
        raise ValueError("Evaluation evidence must contain at least one episode.")
    seeds = [int(row["seed"]) for row in rows]
    rewards = [float(row["reward"]) for row in rows]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Paired comparisons require unique episode seeds.")
    if not np.all(np.isfinite(rewards)):
        raise ValueError("Evaluation rewards must be finite.")
    return dict(zip(seeds, rewards, strict=True))


def paired_comparison(reference_rows, candidate_rows, bootstrap_seed=191,
                      bootstrap_samples=2000):
    """Bootstrap paired score differences; pair by seed, never by row order."""
    reference = _reward_by_seed(reference_rows)
    candidate = _reward_by_seed(candidate_rows)
    if reference.keys() != candidate.keys():
        raise ValueError("The two policies must use exactly the same episode seeds.")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive.")
    seeds = sorted(reference)
    differences = np.array([candidate[s] - reference[s] for s in seeds])
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(0, len(seeds), size=(bootstrap_samples, len(seeds)))
    low, high = np.percentile(differences[indices].mean(axis=1), [2.5, 97.5])
    return {
        "n_pairs": len(seeds),
        "reference_mean": float(np.mean(list(reference.values()))),
        "candidate_mean": float(np.mean(list(candidate.values()))),
        "mean_difference": float(differences.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "positive_difference_supported": bool(len(seeds) > 1 and low > 0),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "inference_note": (
            "One pair cannot estimate uncertainty across evaluation seeds."
            if len(seeds) == 1 else
            "Percentile interval over paired evaluation seeds; it does not "
            "measure variability across independent training runs."
        ),
    }


def summarize_evaluation(rows):
    _reward_by_seed(rows)
    rewards = np.array([float(row["reward"]) for row in rows])
    lengths = np.array([int(row["length"]) for row in rows])
    return {
        "episodes": len(rows),
        "mean_reward": float(rewards.mean()),
        "std_reward": float(rewards.std(ddof=1)) if len(rewards) > 1 else 0.0,
        "min_reward": float(rewards.min()),
        "max_reward": float(rewards.max()),
        "mean_length": float(lengths.mean()),
        "length_500_fraction": float(np.mean(lengths == 500)),
        "terminated_count": sum(bool(row["terminated"]) for row in rows),
        "truncated_count": sum(bool(row["truncated"]) for row in rows),
        "unterminated_count": sum(
            not (row["terminated"] or row["truncated"]) for row in rows
        ),
        "std_convention": "Sample standard deviation (ddof=1); reported as 0 for one episode, where uncertainty is not estimable.",
    }


def blackout_observation(observation):
    """Remove visual information while leaving the environment untouched."""
    return np.zeros_like(observation)


def cartpole_initial_states(grid_size=7, repeats=5, seed=91000):
    """Return a fixed angle/velocity grid with reproducible small jitters."""
    if grid_size < 2 or repeats < 1:
        raise ValueError("Use at least two grid points and one repeat.")
    rng = np.random.default_rng(seed)
    angles = np.linspace(-0.18, 0.18, grid_size)
    velocities = np.linspace(-1.5, 1.5, grid_size)
    points = [
        (float(angle), float(velocity), repeat)
        for angle in angles for velocity in velocities for repeat in range(repeats)
    ]
    jitters = rng.uniform(-1.0, 1.0, size=(len(points), 4))
    scales = np.array([0.01, 0.02, 0.002, 0.01])
    return [
        {
            "angle": angle,
            "angular_velocity": velocity,
            "repeat": repeat,
            "seed": seed + index,
            "initial_state": (
                np.array([0.0, 0.0, angle, velocity]) + jitters[index] * scales
            ).tolist(),
        }
        for index, (angle, velocity, repeat) in enumerate(points)
    ]


def audit_targets(trace, gamma):
    """Check observed targets without treating an unobserved edge as passed."""
    checks = [
        {
            "terminated": bool(row["terminated"]),
            "truncated": bool(row["truncated"]),
            "matches": bool(np.isclose(
                float(row["target"]),
                float(row["train_reward"]) + (
                    0.0 if row["terminated"] else gamma * float(row["next_max_q"])
                ),
                rtol=1e-5, atol=1e-6,
            )),
        }
        for row in trace
    ]

    def status(condition):
        selected = [row for row in checks if condition(row)]
        return ("not_observed" if not selected else
                "passed" if all(row["matches"] for row in selected) else "failed")

    return {
        "transitions_checked": len(checks),
        "all_checked_targets_match": all(row["matches"] for row in checks) if checks else None,
        "terminal_status": status(lambda row: row["terminated"]),
        "truncation_status": status(lambda row: row["truncated"] and not row["terminated"]),
        "nonterminal_status": status(lambda row: not row["terminated"]),
        "expected_rule": "Reward only at true termination; bootstrap across time truncation.",
        "scope": "Checks only recorded transitions, not every replay-buffer update.",
    }


def _load_agent(path, device):
    from dqn import DQNAgent
    return DQNAgent.load(path, device)


def _evaluate(*args, **kwargs):
    from train import evaluate_model
    return evaluate_model(*args, **kwargs)


def _write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot save an empty validation table.")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _cartpole_perturbation(agent, run_dir, grid_size, grid_repeats):
    points = cartpole_initial_states(grid_size, grid_repeats)
    outcomes = []
    for point in points:
        result, _ = _evaluate(
            agent, episodes=1, seed=point["seed"], epsilon=0.0,
            initial_state=np.array(point["initial_state"], dtype=np.float64),
        )
        row = result[0]
        outcomes.append({
            **{key: value for key, value in point.items() if key != "initial_state"},
            "initial_x": point["initial_state"][0],
            "initial_x_velocity": point["initial_state"][1],
            "initial_angle": point["initial_state"][2],
            "initial_angular_velocity": point["initial_state"][3],
            "reward": row["reward"], "length": row["length"],
            "success": bool(row["length"] == 500 and not row["terminated"]),
            "terminated": row["terminated"], "truncated": row["truncated"],
        })
    _write_csv(run_dir / "validation_grid.csv", outcomes)
    return {
        "experiment": "Initial pole angle and angular velocity stress test.",
        "expectation": "Recovery may deteriorate near the stability boundary; no monotonicity is assumed.",
        "grid_size": grid_size, "repeats_per_cell": grid_repeats,
        "grid_evaluations": len(outcomes),
        "overall_success_fraction": float(np.mean([row["success"] for row in outcomes])),
        "mean_length": float(np.mean([row["length"] for row in outcomes])),
        "jitter": "Uniform independent perturbations: x +/-0.01, xdot +/-0.02, angle +/-0.002, angular velocity +/-0.01.",
        "interpretation": "This deliberately changes the reset distribution; it is not the standard 500-episode evaluation.",
        "nonstandard_plot_data": "validation_grid.csv",
    }


def _mspacman_perturbation(agent, normal_rows, run_dir, episodes):
    blackout_rows, _ = _evaluate(
        agent, episodes=episodes, seed=80000, epsilon=0.0,
        observation_transform=blackout_observation,
    )
    _write_csv(run_dir / "validation_blackout.csv", blackout_rows)
    comparison = paired_comparison(blackout_rows, normal_rows)
    return {
        "experiment": "Normal observations versus all-zero image input with matched reset seeds.",
        "expectation": "A visually informed policy may outperform its blackout version; the measured difference decides.",
        "blackout_policy": summarize_evaluation(blackout_rows),
        "normal_minus_blackout": comparison,
        "interpretation": "Only network inputs change; environment dynamics and raw game scoring stay unchanged. Paired seeds do not force identical trajectories after actions diverge.",
        "nonstandard_plot_data": "validation_final.csv and validation_blackout.csv",
    }


def _comparison_experiments(agent, initial, run_dir, episodes):
    final_rows, trace = _evaluate(
        agent, episodes=episodes, seed=80000, epsilon=0.0, trace_steps=10,
    )
    initial_rows, _ = _evaluate(initial, episodes=episodes, seed=80000, epsilon=0.0)
    random_rows, _ = _evaluate(agent, episodes=episodes, seed=80000, epsilon=1.0)
    for name, rows in [("final", final_rows), ("initial", initial_rows), ("random", random_rows)]:
        _write_csv(run_dir / f"validation_{name}.csv", rows)
    _write_json(run_dir / "validation_first10_steps.json", trace)
    return final_rows, initial_rows, random_rows, trace


def run_validation(task, run_dir, episodes=50, device="cpu", grid_size=7, grid_repeats=5):
    if task not in {"cartpole", "mspacman"}:
        raise ValueError("task must be cartpole or mspacman.")
    if episodes < 1 or grid_size < 2 or grid_repeats < 1:
        raise ValueError("episodes/repeats must be positive and grid_size at least two.")
    run_dir = Path(run_dir)
    agent, _ = _load_agent(run_dir / "policy.pt", device)
    initial, _ = _load_agent(run_dir / "initial.pt", device)
    if agent.config.task != task or initial.config.task != task:
        raise ValueError("Both checkpoints must match the requested task.")
    final_rows, initial_rows, random_rows, trace = _comparison_experiments(
        agent, initial, run_dir, episodes,
    )
    perturbation = (
        _cartpole_perturbation(agent, run_dir, grid_size, grid_repeats)
        if task == "cartpole" else
        _mspacman_perturbation(agent, final_rows, run_dir, episodes)
    )
    summary = _validation_summary(task, agent, episodes, final_rows, initial_rows,
                                  random_rows, trace, perturbation)
    _write_json(run_dir / "validation.json", summary)
    return summary


def _validation_summary(task, agent, episodes, final_rows, initial_rows, random_rows,
                        trace, perturbation):
    final_summary = summarize_evaluation(final_rows)
    return {
        "task": task,
        "sampling": {"episodes_per_condition": episodes, "first_seed": 80000,
                     "final_epsilon": 0.0, "initial_epsilon": 0.0, "random_epsilon": 1.0,
                     "checkpoint": "policy.pt", "initial_checkpoint": "initial.pt",
                     "note": "Held-out seed comparison; separate from the required 500-episode evaluation."},
        "final_policy": final_summary,
        "initial_policy": summarize_evaluation(initial_rows),
        "random_policy": summarize_evaluation(random_rows),
        "comparison": {
            "final_vs_random": paired_comparison(random_rows, final_rows),
            "final_vs_initial": paired_comparison(initial_rows, final_rows),
            "expectation": "A learned frozen policy should outperform untrained and random policies under fixed exploration settings.",
        },
        "perturbation": perturbation,
        "target_audit": audit_targets(trace, agent.config.gamma),
        "boundaries": {
            "observed_terminated": final_summary["terminated_count"],
            "observed_truncated": final_summary["truncated_count"],
            "all_episodes_report_an_end": final_summary["unterminated_count"] == 0,
            "time_limit_rule": "Time truncation ends rollout, but does not zero the Bellman bootstrap.",
            "life_loss_note": ("Validation does not impose life-loss endings; scores are full-environment episode rewards."
                               if task == "mspacman" else "Not applicable."),
        },
        "limitations": [
            "One selected training seed cannot establish robustness across training seeds.",
            "Confidence intervals quantify seed variation within this evaluation protocol only.",
            "Perturbation performance is not a replacement for standard-environment evaluation.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["cartpole", "mspacman"], required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--grid-size", type=int, default=7)
    parser.add_argument("--grid-repeats", type=int, default=5)
    args = parser.parse_args()
    summary = run_validation(args.task, args.run, args.episodes, args.device,
                             args.grid_size, args.grid_repeats)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
