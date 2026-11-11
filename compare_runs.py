"""Freeze development-only selection, then compare policies on paired fresh seeds."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import torch

from dqn import DQNAgent
from train import evaluate_model, write_csv, write_json
from validate import paired_comparison, summarize_evaluation


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path):
    return json.loads(Path(path).read_text())


def _csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def _flag(value):
    if value not in (True, False, "True", "False", "1", "0"):
        raise ValueError("Episode ending flags must be boolean.")
    return value in (True, "True", "1")


def _evaluation(rows, episodes, seed):
    if episodes <= 0 or seed < 0:
        raise ValueError("Episodes must be positive and seeds nonnegative.")
    normalized = [{**r, "seed": int(r["seed"]), "reward": float(r["reward"]),
                   "length": int(r["length"]), "terminated": _flag(r["terminated"]),
                   "truncated": _flag(r["truncated"])} for r in rows]
    if len(normalized) != episodes or {r["seed"] for r in normalized} != set(range(seed, seed+episodes)):
        raise ValueError("Evaluation must contain exactly the planned unique seed set.")
    if any(not math.isfinite(r["reward"]) or r["length"] <= 0 or
           not (r["terminated"] or r["truncated"]) for r in normalized):
        raise ValueError("Evaluation requires finite rewards and complete episodes.")
    return normalized


def _evidence_paths(baseline_run, new_run, output, plan_path=None, baseline_development=None):
    baseline, candidate, output = Path(baseline_run), Path(new_run), Path(output)
    reference = Path(baseline_development) if baseline_development else output/"baseline_development20.json"
    return {
        "baseline_checkpoint": baseline/"policy.pt", "candidate_checkpoint": candidate/"policy.pt",
        "plan": Path(plan_path) if plan_path else output/"experiment_plan.json",
        "baseline_development": reference, "baseline_development_rows": reference.with_suffix(".csv"),
        "candidate_development": candidate/"development.csv", "candidate_config": candidate/"config.json",
    }


def _development_selection(paths):
    plan, reference = _json(paths["plan"]), _json(paths["baseline_development"])
    protocol, config = plan["mspacman"], _json(paths["candidate_config"])
    expected = {"requested_episodes": protocol["episodes"],
                "development_seed": protocol["development_seed"],
                "development_episodes": protocol["development_episodes"],
                "evaluation_seed": protocol["final_evaluation_seed"],
                "final_episodes": protocol["final_evaluation_episodes"]}
    if any(config.get(k) != value for k, value in expected.items()):
        raise ValueError("Candidate configuration does not match the declared protocol.")
    n, seed = protocol["development_episodes"], protocol["development_seed"]
    baseline_rows = _evaluation(_csv(paths["baseline_development_rows"]), n, seed)
    baseline_mean = sum(r["reward"] for r in baseline_rows)/n
    if (reference["checkpoint_sha256"] != _sha(paths["baseline_checkpoint"]) or
            reference["seed_start"] != seed or reference["development"]["n"] != n or
            not math.isclose(baseline_mean, reference["development"]["mean"]) or
            not math.isclose(baseline_mean, plan["old_model_development_mean"])):
        raise ValueError("Baseline development evidence does not match the fixed reference.")
    development = _csv(paths["candidate_development"])
    episodes = [int(r["training_episode"]) for r in development]
    if not episodes or episodes != sorted(set(episodes)) or episodes[-1] != protocol["episodes"]:
        raise ValueError("Training and development evaluations must be complete before freezing.")
    if any(int(r["n"]) != n or not math.isfinite(float(r["mean"])) for r in development):
        raise ValueError("Every development evaluation must use the planned episode count.")
    best = max(development, key=lambda r: float(r["mean"]))  # first maximum retains earliest tie
    metadata = torch.load(paths["candidate_checkpoint"], map_location="cpu", weights_only=True)["metadata"]
    if (metadata["episode"] != int(best["training_episode"]) or
            metadata["development_seed"] != seed or
            not math.isclose(metadata["development_mean"], float(best["mean"]))):
        raise ValueError("Selected checkpoint must match the earliest best development result.")
    return {"baseline_development_mean": baseline_mean,
            "candidate_development_mean": float(best["mean"]),
            "candidate_selected": float(best["mean"]) > baseline_mean,
            "candidate_selected_episode": int(best["training_episode"])}


def freeze_selection(baseline_run, new_run, output, plan_path=None, baseline_development=None):
    """Must run after the final development pass and before final CSV exists."""
    if (Path(new_run)/"evaluation.csv").exists():
        raise FileExistsError("Freeze selection before candidate evaluation.csv exists.")
    destination = Path(output)/"selection_decision.json"
    if destination.exists():
        raise FileExistsError("A frozen selection decision already exists.")
    paths = _evidence_paths(baseline_run, new_run, output, plan_path, baseline_development)
    selection = _development_selection(paths)
    plan = _json(paths["plan"])
    n, seed = plan["mspacman"]["final_evaluation_episodes"], plan["mspacman"]["final_evaluation_seed"]
    decision = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(), "selection": selection,
                "selection_rule": plan["selection_rule"],
                "evaluation": {"episodes": n, "seed_start": seed, "seed_end": seed+n-1},
                "evidence_sha256": {key: _sha(path) for key, path in paths.items()}}
    write_json(destination, decision)
    return decision


def _fingerprint(agent):
    digest = hashlib.sha256()
    for model in (agent.online, agent.target):
        for name, value in model.state_dict().items():
            digest.update(name.encode())
            digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def compare_runs(baseline_run, new_run, output, episodes=500, seed=200007, device="cpu",
                 plan_path=None, baseline_development=None):
    """Evaluate frozen baseline/random policies; final scores cannot alter selection."""
    paths = _evidence_paths(baseline_run, new_run, output, plan_path, baseline_development)
    frozen = _json(Path(output)/"selection_decision.json")
    if any(_sha(path) != frozen["evidence_sha256"][key] for key, path in paths.items()):
        raise ValueError("Checkpoint or development evidence changed after selection was frozen.")
    protocol = {"episodes": episodes, "seed_start": seed, "seed_end": seed+episodes-1}
    if protocol != frozen["evaluation"]:
        raise ValueError("Evaluation seeds and count must match the frozen protocol.")
    candidate_rows = _evaluation(_csv(Path(new_run)/"evaluation.csv"), episodes, seed)
    torch.set_num_threads(1)
    agent, _ = DQNAgent.load(paths["baseline_checkpoint"], device)
    candidate, _ = DQNAgent.load(paths["candidate_checkpoint"], device)
    if (agent.config.task, agent.config.frame_stack, agent.n_actions) != (
            candidate.config.task, candidate.config.frame_stack, candidate.n_actions):
        raise ValueError("Policies must use the same task, frame stack and action space.")
    agent.online.eval()
    before = _fingerprint(agent)
    baseline_rows, _ = evaluate_model(agent, episodes, seed, epsilon=0.0, progress=True)
    random_rows, _ = evaluate_model(agent, episodes, seed, epsilon=1.0, progress=True)
    baseline_rows = _evaluation(baseline_rows, episodes, seed)
    random_rows = _evaluation(random_rows, episodes, seed)
    unchanged = before == _fingerprint(agent) and all(
        _sha(path) == frozen["evidence_sha256"][key] for key, path in paths.items())
    if not unchanged:
        raise ValueError("Frozen checkpoint weights or evidence changed during evaluation.")
    result = {"baseline": summarize_evaluation(baseline_rows),
              "candidate": summarize_evaluation(candidate_rows), "random": summarize_evaluation(random_rows),
              "comparisons": {"candidate_minus_baseline": paired_comparison(baseline_rows, candidate_rows),
                              "candidate_minus_random": paired_comparison(random_rows, candidate_rows)},
              "selection": frozen["selection"], "evaluation": protocol, "weights_unchanged": unchanged,
              "evidence_sha256": frozen["evidence_sha256"], "selection_frozen_at_utc": frozen["frozen_at_utc"],
              "source_sha256": {name: _sha(Path(__file__).with_name(name))
                                for name in ("compare_runs.py", "dqn.py", "train.py", "validate.py")},
              "sampling_protocol": "Same fresh seed set; complete games; greedy learned policies; uniform random actions. Final results are descriptive and never select the adopted policy.",
              "limitations": ["Training configuration and budget changed together; this is not an isolated ablation.",
                              "One training seed per Ms. Pac-Man configuration; paired intervals measure evaluation-seed variation only."]}
    write_csv(Path(output)/"baseline_evaluation.csv", baseline_rows)
    write_csv(Path(output)/"random_evaluation.csv", random_rows)
    write_json(Path(output)/"comparison.json", result)
    return result


def summarize_cartpole_controls(root):
    """Describe predeclared seeds without replacing the original seed-7 policy."""
    root = Path(root)
    runs = []
    for seed in (7, 8, 9):
        run = root.parent/"cartpole" if seed == 7 else root/f"seed{seed}"
        config, metrics = _json(run/"config.json"), _json(run/"metrics.json")
        if config["seed"] != seed or config["task"] != "cartpole":
            raise ValueError("CartPole controls must identify their declared training seeds.")
        if seed != 7 and (config["evaluation_seed"] != 200007 or config["final_episodes"] != 500):
            raise ValueError("CartPole control evaluation must use the 500 fresh matched seeds.")
        table = root/"seed7_evaluation"/"evaluation.csv" if seed == 7 else run/"evaluation.csv"
        rows = _evaluation(_csv(table), 500, 200007)
        summary = summarize_evaluation(rows)
        runs = [*runs, {"seed": seed, "training_episodes": metrics["training_episodes"],
                       "selected_episode": metrics["selected_checkpoint"]["episode"],
                       "mean_reward": summary["mean_reward"], "std_reward": summary["std_reward"],
                       "successes": sum(r["length"] == 500 and not r["terminated"] for r in rows),
                       "episodes": len(rows)}]
    result = {"primary_seed": 7, "runs": runs,
              "note": "The primary seed-7 checkpoint is unchanged. Training-seed controls are descriptive; final test scores do not select the primary policy."}
    write_json(root/"summary.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline-run", "new-run", "output", "plan", "baseline-development", "cartpole-controls"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=200007)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args(argv)
    if args.cartpole_controls:
        result = summarize_cartpole_controls(args.cartpole_controls)
    else:
        if not all((args.baseline_run, args.new_run, args.output)):
            parser.error("--baseline-run, --new-run and --output are required")
        common = (args.baseline_run, args.new_run, args.output)
        options = {"plan_path": args.plan, "baseline_development": args.baseline_development}
        result = (freeze_selection(*common, **options) if args.freeze_only else
                  compare_runs(*common, episodes=args.episodes, seed=args.seed, device=args.device, **options))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
