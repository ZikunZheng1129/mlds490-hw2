"""Train standard DQN and evaluate frozen checkpoints on complete game episodes."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import socket
import time

import numpy as np
import torch

import dqn
from dqn import Config, DQNAgent, ReplayBuffer, defaults, make_env


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot write an empty result table")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    if not rows:
        raise ValueError("At least one episode is required")
    rewards = np.array([row["reward"] for row in rows], dtype=float)
    return {"n": len(rows), "mean": float(rewards.mean()),
            "std": float(rewards.std(ddof=1)) if len(rows) > 1 else 0.0,
            "median": float(np.median(rewards)), "min": float(rewards.min()),
            "max": float(rewards.max()),
            "mean_length": float(np.mean([r.get("length", 0) for r in rows])),
            "terminated_episodes": sum(bool(r.get("terminated", False)) for r in rows),
            "truncated_episodes": sum(bool(r.get("truncated", False)) for r in rows)}


def q_values(agent, state, target=False):
    with torch.inference_mode():
        inputs = torch.as_tensor(np.asarray(state), device=agent.device).unsqueeze(0)
        network = agent.target if target else agent.online
        return network(inputs)[0].detach().cpu().numpy()


def epsilon_at(config, step):
    fraction = min(max(step / config.epsilon_decay_steps, 0.0), 1.0)
    return config.epsilon_start + fraction * (config.epsilon_end - config.epsilon_start)


def state_summary(state):
    values = np.asarray(state)
    if values.ndim == 1:
        return values.tolist()
    return {"shape": list(values.shape), "mean": float(values.mean()),
            "std": float(values.std()), "sha256": hashlib.sha256(values.tobytes()).hexdigest()}


def trace_record(agent, state, next_state, values, action, reward, terminated, truncated, step):
    train_reward = float(np.sign(reward)) if agent.config.reward_clip else float(reward)
    next_max = float(q_values(agent, next_state, target=True).max())
    target = dqn.bellman_targets(torch.tensor([train_reward]), torch.tensor([terminated]),
                                 torch.tensor([next_max]), agent.config.gamma).item()
    return {"step": step, "state_summary": state_summary(state),
            "q_values": values.tolist(), "action": int(action), "reward": float(reward),
            "train_reward": train_reward, "next_max_q": next_max,
            "target": float(target),
            "terminated": bool(terminated), "truncated": bool(truncated)}


def _evaluate_episode(agent, env, seed, epsilon, transform, initial_state, trace_steps):
    state, _ = env.reset(seed=int(seed))
    if initial_state is not None:
        if agent.config.task != "cartpole":
            raise ValueError("Initial-state intervention is only supported for CartPole")
        state = np.array(initial_state, dtype=np.float32, copy=True)
        if state.shape != (4,) or not np.isfinite(state).all():
            raise ValueError("CartPole initial_state must contain four finite values")
        env.unwrapped.state = state.astype(np.float64, copy=True)
    rng = np.random.default_rng(seed + 1_000_000)
    total, length, maximum, trace = 0.0, 0, -float("inf"), ()
    while True:
        observed = transform(state) if transform is not None else state
        values = q_values(agent, observed)
        action = int(rng.integers(len(values))) if rng.random() < epsilon else int(values.argmax())
        next_state, reward, terminated, truncated, _ = env.step(action)
        length, total, maximum = length + 1, total + float(reward), max(maximum, float(values.max()))
        if length <= trace_steps:
            trace += (trace_record(agent, state, next_state, values, action, reward,
                                   terminated, truncated, length),)
        state = next_state
        if terminated or truncated:
            return {"seed": int(seed), "reward": total, "length": length,
                    "terminated": bool(terminated), "truncated": bool(truncated),
                    "max_q": maximum}, trace


def evaluate_model(agent, episodes, seed, epsilon=0.0, observation_transform=None,
                   initial_state=None, trace_steps=0, progress=False):
    if episodes <= 0 or not 0 <= epsilon <= 1:
        raise ValueError("Episodes must be positive and epsilon must lie in [0, 1]")
    env = make_env(agent.config.task, frame_stack=agent.config.frame_stack)
    rows, trace = (), ()
    try:
        for episode in range(episodes):
            row, steps = _evaluate_episode(agent, env, seed + episode, epsilon,
                                           observation_transform, initial_state,
                                           trace_steps if episode == 0 else 0)
            rows += ({"episode": episode + 1, **row},)
            trace += steps
            if progress and ((episode + 1) % 50 == 0 or episode + 1 == episodes):
                mean = np.mean([item["reward"] for item in rows])
                print(f"Evaluation {episode + 1}/{episodes}: mean={mean:.2f}", flush=True)
    finally:
        env.close()
    return list(rows), list(trace)


def runtime_info(device):
    import ale_py
    import gymnasium
    return {"hostname": socket.gethostname(), "python": platform.python_version(),
            "torch": torch.__version__, "gymnasium": gymnasium.__version__,
            "ale_py": ale_py.__version__, "numpy": np.__version__, "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if str(device).startswith("cuda") else None,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                              for name in ("dqn.py", "train.py")}}


def collect_episode(agent, env, replay, rng, episode, steps, updates):
    config = agent.config
    state, _ = env.reset(seed=config.seed + episode)
    total, length, maximum, losses, td_errors = 0.0, 0, -float("inf"), (), ()
    while True:
        epsilon = epsilon_at(config, steps)
        values = q_values(agent, state)
        action = int(rng.integers(len(values))) if rng.random() < epsilon else int(values.argmax())
        next_state, reward, terminated, truncated, _ = env.step(action)
        learn_reward = float(np.sign(reward)) if config.reward_clip else float(reward)
        replay.add(state, action, learn_reward, next_state, terminated, truncated)
        steps, length, total = steps + 1, length + 1, total + float(reward)
        maximum = max(maximum, float(values.max()))
        if (steps >= config.learning_starts and len(replay) >= config.batch_size
                and steps % config.train_frequency == 0):
            metrics = agent.update(replay.sample(config.batch_size, agent.device))
            updates += 1
            losses += (metrics["loss"],)
            td_errors += (metrics["td_abs"],)
            if updates % config.target_interval == 0:
                agent.sync_target()
        state = next_state
        if terminated or truncated:
            row = {"episode": episode, "steps": steps, "reward": total, "length": length,
                   "max_q": maximum, "epsilon": epsilon, "updates": updates,
                   "loss": float(np.mean(losses)) if losses else 0.0,
                   "td_abs": float(np.mean(td_errors)) if td_errors else 0.0,
                   "terminated": bool(terminated), "truncated": bool(truncated)}
            return row, steps, updates


def _development(agent, episode, steps, updates, episodes, output, best_score, history,
                 seed=40_000):
    rows, _ = evaluate_model(agent, episodes, seed=seed, epsilon=0.0)
    stats = summarize(rows)
    new_history = history + ({"training_episode": episode, **stats},)
    write_csv(output / "development.csv", new_history)
    metadata = {"episode": episode, "steps": steps, "optimizer_steps": updates,
                "development_mean": stats["mean"], "development_seed": seed}
    agent.save(output / "latest.pt", metadata)
    if stats["mean"] > best_score:
        agent.save(output / "policy.pt", metadata)
        best_score = stats["mean"]
    print(f"Development at episode {episode}: mean={stats['mean']:.2f}, "
          f"best={best_score:.2f}", flush=True)
    return best_score, new_history, stats


def run_training(config: Config, episodes, output, device="cpu", eval_every=50,
                 eval_episodes=10, final_episodes=500, early_stop=True,
                 evaluation_seed=None, development_seed=40_000):
    if min(episodes, eval_every, eval_episodes, final_episodes) <= 0:
        raise ValueError("All episode counts and evaluation intervals must be positive")
    evaluation_seed = 100_000 + config.seed if evaluation_seed is None else evaluation_seed
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0
           for value in (evaluation_seed, development_seed)):
        raise ValueError("Development and evaluation seeds must be nonnegative integers")
    if max(development_seed, evaluation_seed) < min(development_seed + eval_episodes,
                                                   evaluation_seed + final_episodes):
        raise ValueError("Development and final evaluation seed ranges must not overlap")
    output = Path(output)
    if (output / "config.json").exists():
        raise FileExistsError(f"Run already exists: {output}; choose a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(1)
    env = make_env(config.task, frame_stack=config.frame_stack)
    agent = DQNAgent(env.observation_space.shape, env.action_space.n, config, device)
    replay, rng = ReplayBuffer(config.buffer_size, config.seed), np.random.default_rng(config.seed)
    write_json(output / "config.json", {**asdict(config), "requested_episodes": episodes,
               "eval_every": eval_every, "development_episodes": eval_episodes,
               "final_episodes": final_episodes, "early_stop": early_stop,
               "runtime": runtime_info(device), "max_q_definition": "max over visited states and all actions in each training episode",
               "evaluation_epsilon": 0.0, "evaluation_seed": evaluation_seed,
               "development_seed": development_seed})
    agent.save(output / "initial.pt", {"episode": 0})
    history, development, steps, updates, solved_streak = (), (), 0, 0, 0
    best_score, started = -float("inf"), time.monotonic()
    try:
        for episode in range(1, episodes + 1):
            row, steps, updates = collect_episode(agent, env, replay, rng, episode, steps, updates)
            history += ({**row, "elapsed_seconds": round(time.monotonic() - started, 3)},)
            write_csv(output / "training.csv", history)
            if episode % 10 == 0 or episode == 1:
                mean = np.mean([r["reward"] for r in history[-100:]])
                print(f"{config.task} episode={episode} steps={steps} reward={row['reward']:.0f} "
                      f"last100={mean:.1f} eps={row['epsilon']:.3f} "
                      f"steps/s={steps / max(time.monotonic()-started, 1):.1f}", flush=True)
            if episode % eval_every == 0 or episode == episodes:
                best_score, development, stats = _development(agent, episode, steps, updates,
                    eval_episodes, output, best_score, development, seed=development_seed)
                solved = stats["min"] == 500 and stats["terminated_episodes"] == 0
                solved_streak = solved_streak + 1 if solved else 0
                if early_stop and config.task == "cartpole" and solved_streak >= 2:
                    break
    finally:
        env.close()
    # The independent final evaluation never selects the checkpoint.
    selected, selected_metadata = DQNAgent.load(output / "policy.pt", device)
    scores, trace = evaluate_model(selected, final_episodes, evaluation_seed,
                                   trace_steps=10, progress=True)
    write_csv(output / "evaluation.csv", scores)
    write_json(output / "first10_steps.json", trace)
    result = {"training_episodes": len(history), "environment_steps": steps,
              "optimizer_steps": updates, "selected_checkpoint": selected_metadata,
              "evaluation": summarize(scores), "wall_seconds": time.monotonic()-started}
    if config.task == "cartpole":
        result = {**result, "success_rate": float(np.mean([
            r["length"] == 500 and not r["terminated"] for r in scores]))}
    write_json(output / "metrics.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def positive_integer(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def nonnegative_integer(value):
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("cartpole", "mspacman"))
    parser.add_argument("--episodes", type=positive_integer, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=nonnegative_integer, default=7)
    parser.add_argument("--profile", choices=("baseline", "extended"), default="baseline")
    parser.add_argument("--development-seed", type=nonnegative_integer, default=40_000)
    parser.add_argument("--evaluation-seed", type=nonnegative_integer,
                        help="First final evaluation seed; default is 100000 + training seed")
    parser.add_argument("--eval-every", type=positive_integer, default=50)
    parser.add_argument("--eval-episodes", type=positive_integer, default=10)
    parser.add_argument("--final-episodes", type=positive_integer, default=500)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--epsilon-decay-steps", type=positive_integer)
    parser.add_argument("--no-early-stop", action="store_true")
    parser.add_argument("--evaluate", type=Path, help="Evaluate an existing checkpoint without training")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.evaluate:
        if any((args.output / name).exists() for name in ("config.json", "metrics.json", "evaluation.csv")):
            parser.error("Evaluation results already exist; choose a fresh output directory")
        agent, _ = DQNAgent.load(args.evaluate, args.device)
        seed = 100_000 + args.seed if args.evaluation_seed is None else args.evaluation_seed
        rows, trace = evaluate_model(agent, args.episodes, seed, trace_steps=10, progress=True)
        write_csv(args.output / "evaluation.csv", rows)
        write_json(args.output / "metrics.json", summarize(rows))
        write_json(args.output / "first10_steps.json", trace)
        return
    if args.task is None:
        parser.error("--task is required for training")
    if args.profile == "extended" and args.task != "mspacman":
        parser.error("--profile extended is only supported for mspacman")
    config = defaults(args.task, args.seed, profile=args.profile)
    overrides = {key: getattr(args, key) for key in ("learning_rate", "epsilon_decay_steps")
                 if getattr(args, key) is not None}
    config = replace(config, **overrides)
    run_training(config, args.episodes, args.output, args.device, args.eval_every,
                 args.eval_episodes, args.final_episodes, not args.no_early_stop,
                 evaluation_seed=args.evaluation_seed, development_seed=args.development_seed)


if __name__ == "__main__":
    main()
