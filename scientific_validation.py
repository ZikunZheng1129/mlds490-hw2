"""Controlled learning, value calibration and image-information experiments.

This supplements game scores and software tests. Checkpoints stay frozen during
rollouts, and failed scientific expectations remain recorded as failures.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from dqn import Batch, DQNAgent, bellman_targets, defaults, make_env
from train import evaluate_model, q_values, write_csv, write_json
from validate import paired_comparison, summarize_evaluation


def known_solution(gamma, reward_scale):
    if not 0 <= gamma < 1 or not np.isfinite(reward_scale) or reward_scale < 0:
        raise ValueError("Use gamma in [0, 1) and a finite nonnegative reward scale")
    # Action 0 continues to state 1, then pays c; action 1 quits for zero.
    return np.array([[gamma * reward_scale, 0.0], [reward_scale, 0.0]])


def _fingerprint(agent):
    digest = hashlib.sha256()
    for network in (agent.online, agent.target):
        for name, value in network.state_dict().items():
            digest.update(name.encode())
            digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def analytic_case(gamma, reward_scale, steps=1500, learning_rate=0.001, tolerance=0.02):
    expected = known_solution(gamma, reward_scale)
    if steps < 1 or not np.isfinite(learning_rate) or learning_rate < 0 or tolerance <= 0:
        raise ValueError("Use positive steps/tolerance and a nonnegative learning rate")
    config = replace(defaults("cartpole", seed=123), gamma=gamma,
                     learning_rate=learning_rate if learning_rate > 0 else 0.001,
                     batch_size=4)
    agent = DQNAgent((2,), 2, config, "cpu")
    # Zero-rate negative control uses the same update, including backward/step.
    agent.optimizer = torch.optim.Adam(agent.online.parameters(), lr=learning_rate)
    states = torch.eye(2)
    batch = Batch(states[[0, 0, 1, 1]], torch.tensor([0, 1, 0, 1]),
                  torch.tensor([0.0, 0.0, reward_scale, 0.0], dtype=torch.float32),
                  states[[1, 0, 0, 0]], torch.tensor([False, True, True, True]),
                  torch.zeros(4, dtype=torch.bool))
    before = _fingerprint(agent)
    for step in range(steps):
        metrics = agent.update(batch)
        if (step + 1) % 25 == 0:
            agent.sync_target()
    with torch.inference_mode():
        learned = agent.online(states).numpy()
    error = float(np.max(np.abs(learned - expected)))
    return {"gamma": gamma, "reward_scale": reward_scale, "updates": steps,
            "learning_rate": learning_rate, "expected": expected.tolist(),
            "learned": learned.tolist(), "max_absolute_error": error,
            "tolerance": tolerance, "passed": bool(error < tolerance),
            "final_loss": metrics["loss"], "weights_unchanged": before == _fingerprint(agent)}


def analytic_experiments(steps=1500):
    cases = [analytic_case(gamma, scale, steps) for gamma, scale in
             ((0.95, 1.0), (0.99, 1.0), (0.0, 1.0), (0.95, 3.0), (0.95, 0.0))]
    control = analytic_case(0.95, 3.0, min(steps, 200), learning_rate=0.0)
    return {"description": "Two-state MDP with known optimal action values; actual DQNAgent.update.",
            "cases": cases, "all_known_solutions_passed": all(c["passed"] for c in cases),
            "zero_learning_rate_control": control,
            "control_detected_no_learning": control["weights_unchanged"] and not control["passed"],
            "limitation": "Tests Bellman learning and scale/discount controls, not pixel-game competence."}


def discounted_returns(rewards, gamma):
    values = np.asarray(rewards, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or not 0 <= gamma < 1:
        raise ValueError("Use nonempty finite rewards and gamma in [0, 1)")
    result = np.zeros_like(values)
    following = 0.0
    for index in range(len(values) - 1, -1, -1):
        following = float(values[index]) + gamma * following
        result[index] = following
    return result


def discounted_return(rewards, gamma):
    return float(discounted_returns(rewards, gamma)[0])


def _tail_bound(gamma, remaining, terminated, truncated):
    return float(gamma ** remaining / (1 - gamma)) if truncated and not terminated else 0.0


def sample_calibration_states(seed, predictions, rewards, gamma, terminated, truncated,
                              max_states=50, minimum_truncated_horizon=100):
    returns = discounted_returns(rewards, gamma)
    if len(predictions) != len(returns) or max_states < 1:
        raise ValueError("Predictions/rewards must align and max_states must be positive")
    count = len(returns)
    usable = count if not truncated or terminated else max(0, count - minimum_truncated_horizon + 1)
    if not usable:
        return []
    indices = np.unique(np.linspace(0, usable - 1, min(max_states, usable), dtype=int))
    return [{"seed": int(seed), "step": int(i), "remaining_horizon": count - int(i),
             "predicted_q": float(predictions[i]), "discounted_return": float(returns[i]),
             "error": float(predictions[i] - returns[i]),
             "terminated": bool(terminated), "truncated": bool(truncated),
             "tail_upper_bound": _tail_bound(gamma, count - int(i), terminated, truncated)}
            for i in indices]


def summarize_calibration(rows):
    if not rows:
        raise ValueError("Calibration needs at least one observation")
    predicted = np.array([r["predicted_q"] for r in rows], dtype=float)
    realized = np.array([r["discounted_return"] for r in rows], dtype=float)
    if not np.isfinite(predicted).all() or not np.isfinite(realized).all():
        raise ValueError("Calibration values must be finite")
    errors = predicted - realized
    variable = len(rows) > 1 and predicted.std() > 1e-12 and realized.std() > 1e-12
    censored = {r["seed"] for r in rows if r["truncated"] and not r["terminated"]}
    return {"episodes": len({r["seed"] for r in rows}), "observations": len(rows),
            "mean_predicted_q": float(predicted.mean()), "mean_discounted_return": float(realized.mean()),
            "mean_signed_error": float(errors.mean()), "mae": float(np.abs(errors).mean()),
            "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "pearson_correlation": float(np.corrcoef(predicted, realized)[0, 1]) if variable else None,
            "censored_episodes": len(censored),
            "maximum_tail_upper_bound": float(max(r["tail_upper_bound"] for r in rows)),
            "correlation_note": "Undefined for a constant variable or fewer than two observations.",
            "sampling_note": "Within-game states are dependent; observation count is not independent sample size."}


def _ending_target(agent, seed, reward, next_state, terminated, truncated):
    learning_reward = float(np.sign(reward)) if agent.config.reward_clip else float(reward)
    next_max = float(q_values(agent, next_state, target=True).max())
    actual = bellman_targets(torch.tensor([learning_reward]), torch.tensor([terminated]),
                             torch.tensor([next_max]), agent.config.gamma).item()
    expected = learning_reward + (0.0 if terminated else agent.config.gamma * next_max)
    return {"seed": int(seed), "reward": float(reward), "train_reward": learning_reward,
            "next_max_q": next_max, "target": float(actual), "expected": expected,
            "terminated": bool(terminated), "truncated": bool(truncated),
            "matches_expected": bool(np.isclose(actual, expected, rtol=1e-5, atol=1e-6))}


def _calibration_episode(agent, env, seed):
    state, _ = env.reset(seed=seed)
    raw_rewards, learning_rewards, predictions = [], [], []
    while True:
        values = q_values(agent, state)
        action = int(values.argmax())
        next_state, reward, terminated, truncated, _ = env.step(action)
        raw_rewards.append(float(reward))
        learning_rewards.append(float(np.sign(reward)) if agent.config.reward_clip else float(reward))
        predictions.append(float(values[action]))
        if terminated or truncated:
            ending = _ending_target(agent, seed, reward, next_state, terminated, truncated)
            break
        state = next_state
    realized = discounted_return(learning_rewards, agent.config.gamma)
    row = {"seed": int(seed), "reward": float(sum(raw_rewards)), "length": len(raw_rewards),
           "terminated": bool(terminated), "truncated": bool(truncated),
           "max_q": max(predictions), "predicted_q": predictions[0], "discounted_return": realized,
           "error": predictions[0] - realized,
           "tail_upper_bound": _tail_bound(agent.config.gamma, len(raw_rewards), terminated, truncated)}
    states = sample_calibration_states(seed, predictions, learning_rewards, agent.config.gamma,
                                      terminated, truncated)
    return row, ending, states


def calibration_rollouts(agent, episodes=50, seed=300000):
    if episodes < 1 or seed < 0:
        raise ValueError("Use positive episode count and nonnegative seed")
    env = make_env(agent.config.task, frame_stack=agent.config.frame_stack)
    rows, endings, states = [], [], []
    try:
        for index in range(episodes):
            row, ending, selected = _calibration_episode(agent, env, seed + index)
            rows.append({"episode": index + 1, **row})
            endings.append(ending)
            states.extend(selected)
    finally:
        env.close()
    return rows, endings, states


def summarize_endings(rows):
    def summary(selected):
        return {"count": len(selected), "status": ("not_observed" if not selected else
                "passed" if all(r["matches_expected"] for r in selected) else "failed")}
    return {"transitions_checked": len(rows),
            "terminal": summary([r for r in rows if r["terminated"]]),
            "truncation": summary([r for r in rows if r["truncated"] and not r["terminated"]]),
            "all_observed_targets_match": all(r["matches_expected"] for r in rows),
            "rule": "Actual dqn.bellman_targets, checked against reward-only terminal / bootstrapped truncation targets.",
            "scope": "Observed episode endings only; unobserved branches are not counted as passes."}


def repeat_latest_frame(observation):
    state = np.asarray(observation)
    if state.ndim != 3 or state.dtype != np.uint8:
        raise ValueError("Expected a uint8 CHW frame stack")
    return np.repeat(state[-1:], state.shape[0], axis=0)


def constant_midgray(observation):
    return np.full_like(observation, 128)


def _pixel_interventions(agent, normal_rows, folder, episodes, seed):
    conditions = {"normal": {"summary": summarize_evaluation(normal_rows)}}
    for name, transform in (("repeat_latest", repeat_latest_frame), ("constant128", constant_midgray)):
        rows, _ = evaluate_model(agent, episodes, seed, epsilon=0.0, observation_transform=transform)
        write_csv(folder / f"scientific_{name}.csv", rows)
        conditions[name] = {"summary": summarize_evaluation(rows),
                            "normal_minus_condition": paired_comparison(rows, normal_rows)}
    return {"seed_start": seed, "episodes_per_condition": episodes, "conditions": conditions,
            "interpretation": "Repeated latest frame removes temporal history; constant 128 normalizes to zero and removes visual information. Both introduce distribution shifts; neither retrains the agent."}


def run_scientific_validation(run_dir, episodes=50, device="cpu", seed=300000, analytic_steps=1500):
    if min(episodes, analytic_steps) < 1 or seed < 0:
        raise ValueError("Use positive episode/update counts and nonnegative seed")
    torch.set_num_threads(1)
    folder = Path(run_dir)
    checkpoint = folder / "policy.pt"
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    agent, metadata = DQNAgent.load(checkpoint, device)
    agent.online.eval()
    before = _fingerprint(agent)
    analytic = analytic_experiments(analytic_steps)
    rows, endings, states = calibration_rollouts(agent, episodes, seed)
    write_csv(folder / "calibration.csv", rows)
    write_csv(folder / "ending_targets.csv", endings)
    if states:
        write_csv(folder / "calibration_states.csv", states)
    interventions = (_pixel_interventions(agent, rows, folder, episodes, seed)
                     if agent.config.task == "mspacman" else None)
    unchanged = before == _fingerprint(agent) and checkpoint_hash == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if not unchanged:
        raise RuntimeError("Scientific evaluation changed the frozen checkpoint")
    result = {"task": agent.config.task, "checkpoint_sha256": checkpoint_hash,
              "selected_checkpoint": metadata, "weights_unchanged": unchanged,
              "seed_start": seed, "episodes": episodes,
              "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                for name in ("dqn.py", "train.py", "validate.py", "scientific_validation.py")},
              "analytic_learning": analytic, "calibration": summarize_calibration(rows),
              "sampled_calibration": summarize_calibration(states) if states else None,
              "normal_policy": summarize_evaluation(rows),
              "ending_target_audit": summarize_endings(endings), "pixel_interventions": interventions,
              "limitations": ["One training seed does not establish robustness across retraining.",
                  "Calibration uses raw CartPole rewards and sign-clipped MsPacman rewards, with full-game termination.",
                  "At truncation the observed return omits a nonnegative tail bounded by gamma**remaining/(1-gamma).",
                  "Within-game calibration selects at most 50 evenly spaced states; truncated episodes require at least 100 remaining steps.",
                  "Initial Atari inputs may be identical across seeds; constant-variable correlations are reported as null.",
                  "Image interventions change the input distribution, not the environment or game scoring."]}
    write_json(folder / "scientific_validation.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=300000)
    parser.add_argument("--analytic-steps", type=int, default=1500)
    args = parser.parse_args()
    result = run_scientific_validation(args.run, args.episodes, args.device, args.seed, args.analytic_steps)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
