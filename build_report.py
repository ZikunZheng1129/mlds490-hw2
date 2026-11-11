"""Build the assignment PDF and eight figures from completed, measured results."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from urllib.parse import urlparse
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Image, PageBreak, Paragraph, Preformatted,
                               SimpleDocTemplate, Spacer, Table, TableStyle)


@dataclass(frozen=True)
class Run:
    task: str
    folder: Path
    config: dict
    training: list
    evaluation: list
    development: list
    metrics: dict
    validation: dict
    trace: list
    conditions: dict


def moving_average(values, window=100):
    if not isinstance(window, int) or window < 1:
        raise ValueError("Moving-average window must be a positive integer")
    data = np.asarray(values, dtype=float)
    if data.ndim != 1 or not np.isfinite(data).all():
        raise ValueError("Moving-average input must be a finite vector")
    result = np.full(len(data), np.nan)
    if len(data) >= window:
        result[window-1:] = np.convolve(data, np.ones(window)/window, mode="valid")
    return result


def _read(path):
    if not path.is_file():
        raise ValueError(f"Missing required result file: {path}")
    with path.open() as handle:
        return json.load(handle) if path.suffix == ".json" else list(csv.DictReader(handle))


def _numbers(rows, key):
    try:
        values = np.array([float(row[key]) for row in rows])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Missing or invalid numeric evidence column: {key}") from error
    if not len(values) or not np.isfinite(values).all():
        raise ValueError(f"Evidence column {key} must contain finite values")
    return values


def _true(value):
    return str(value).lower() in {"true", "1"}


def load_run(results_dir, task):
    folder = Path(results_dir) / task
    names = ("config", "metrics", "validation", "validation_first10_steps")
    config, metrics, validation, trace = [_read(folder / f"{name}.json") for name in names]
    training, evaluation, development = [_read(folder / f"{name}.csv")
                                          for name in ("training", "evaluation", "development")]
    if config["task"] != task or validation["task"] != task:
        raise ValueError(f"Task mismatch in {folder}")
    if len(evaluation) != 500:
        raise ValueError(f"{task}: required final evaluation is exactly 500 episodes; found {len(evaluation)}")
    rewards = _numbers(evaluation, "reward")
    if len(set(_numbers(evaluation, "seed"))) != 500:
        raise ValueError(f"{task}: evaluation seeds must be unique")
    for column in ("episode", "reward", "max_q", "steps", "length", "epsilon"):
        _numbers(training, column)
    if not np.array_equal(_numbers(training, "episode"), np.arange(1, len(training)+1)):
        raise ValueError(f"{task}: training episode sequence must be complete")
    if metrics["evaluation"]["n"] != 500 or not np.isclose(metrics["evaluation"]["mean"], rewards.mean()):
        raise ValueError(f"{task}: metrics.json disagrees with evaluation.csv")
    if len(trace) != 10:
        raise ValueError(f"{task}: ten recorded validation transitions are required")
    labels = ["final", "initial", "random", "grid" if task == "cartpole" else "blackout"]
    conditions = {label: _read(folder / f"validation_{label}.csv") for label in labels}
    expected = validation["sampling"]["episodes_per_condition"]
    for label in ("final", "initial", "random"):
        if len(conditions[label]) != expected:
            raise ValueError(f"{task}: incomplete {label} validation evidence")
        _numbers(conditions[label], "reward")
    return Run(task, folder, config, training, evaluation, development,
               metrics, validation, trace, conditions)


def learning_statement(comparison):
    if comparison["ci95_low"] > 0:
        return "The paired interval supports a positive improvement under this evaluation protocol."
    if comparison["ci95_high"] < 0:
        return "The paired interval indicates lower performance under this evaluation protocol."
    return "The paired interval includes zero and does not establish a positive improvement."


def load_supplemental(results_dir):
    root = Path(results_dir)
    paths = {task: root / task / "scientific_validation.json"
             for task in ("cartpole", "mspacman")}
    present = [path.is_file() for path in paths.values()]
    if any(present) and not all(present):
        raise ValueError("Scientific validation must be supplied for both tasks")
    science = {task: _read(path) for task, path in paths.items()} if all(present) else {}
    for task, result in science.items():
        if result["task"] != task or len(result["analytic_learning"]["cases"]) != 5:
            raise ValueError(f"{task}: scientific evidence must match the task and contain five known-MDP cases")
    optional = {"comparison": root / "upgrade_comparison" / "comparison.json",
                "retraining": root / "cartpole_retraining" / "summary.json"}
    return {"science": science,
            **{key: _read(path) if path.is_file() else None for key, path in optional.items()}}


def _save_figure(fig, path):
    fig.tight_layout(pad=1.0)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _diagnostic_plot(run, path):
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    if run.task == "cartpole":
        rows = run.conditions["grid"]
        angles = np.unique(_numbers(rows, "angle"))
        velocities = np.unique(_numbers(rows, "angular_velocity"))
        cells = [[[_true(r["success"]) for r in rows
                   if float(r["angle"]) == angle and float(r["angular_velocity"]) == velocity]
                  for angle in angles] for velocity in velocities]
        rates = np.array([[np.mean(cell) for cell in row] for row in cells])
        im = ax.imshow(rates, origin="lower", vmin=0, vmax=1, cmap="Greys", aspect="auto")
        ax.set_xticks(range(len(angles)), [f"{v:.2f}" for v in angles])
        ax.set_yticks(range(len(velocities)), [f"{v:.2f}" for v in velocities])
        for y, row in enumerate(cells):
            for x, cell in enumerate(row):
                ax.text(x, y, f"{sum(cell)}/{len(cell)}", ha="center", va="center",
                        color="white" if rates[y, x] > 0.6 else "black", fontsize=8)
        ax.set(xlabel="Initial pole angle (rad)", ylabel="Initial angular velocity (rad/s)",
               title=f"Recovery map: 500-step survival ({len(rows)} perturbed episodes)")
        fig.colorbar(im, ax=ax, label="Fraction reaching 500 steps without termination")
    else:
        normal = {int(r["seed"]): float(r["reward"]) for r in run.conditions["final"]}
        blackout = {int(r["seed"]): float(r["reward"]) for r in run.conditions["blackout"]}
        if normal.keys() != blackout.keys():
            raise ValueError("Normal and blackout plots require matched seeds")
        x = np.array([blackout[k] for k in sorted(normal)])
        y = np.array([normal[k] for k in sorted(normal)])
        limit = max(float(x.max()), float(y.max()), 1) * 1.05
        ax.scatter(x, y, c="black", alpha=0.7, s=24)
        ax.plot([0, limit], [0, limit], "--", color="0.5", label="Equal game score")
        ax.set(xlim=(0, limit), ylim=(0, limit), xlabel="Blackout input: game score (points)",
               ylabel="Normal input: game score (points)",
               title=f"Visual-information intervention ({len(x)} matched reset seeds)")
        ax.legend(frameon=False)
        ax.grid(alpha=0.15)
    _save_figure(fig, path)


def plot_run(run):
    folder = run.folder / "figures"
    folder.mkdir(exist_ok=True)
    paths = {name: folder / f"{name}.png" for name in ("max_q", "rewards", "evaluation", "diagnostic")}
    episodes, rewards = _numbers(run.training, "episode"), _numbers(run.training, "reward")
    task_title = "CartPole-v1" if run.task == "cartpole" else "MsPacman-v5"
    unit = "raw reward" if run.task == "cartpole" else "sign-clipped reward"
    fig, ax = plt.subplots(figsize=(7.2, 2.65))
    ax.plot(episodes, _numbers(run.training, "max_q"), color="black", linewidth=0.8)
    ax.set(xlabel="Training episode", ylabel=f"Maximum Q ({unit} units)", title=f"{task_title}: maximum predicted action value")
    ax.grid(alpha=0.15)
    _save_figure(fig, paths["max_q"])
    fig, ax = plt.subplots(figsize=(7.2, 2.65))
    ax.plot(episodes, rewards, color="0.7", linewidth=0.7, label="Episode reward")
    ax.plot(episodes, moving_average(rewards), color="black", linewidth=1.4, label="Last 100 episodes")
    ax.set(xlabel="Training episode", ylabel="Episode reward (raw points)", title=f"{task_title}: training returns")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(alpha=0.15)
    _save_figure(fig, paths["rewards"])
    scores = _numbers(run.evaluation, "reward")
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    bins = (np.arange(-5, 506, 10) if run.task == "cartpole" else
            25 if np.ptp(scores) > 0 else [scores[0] - 0.5, scores[0] + 0.5])
    ax.hist(scores, bins=bins, color="0.65", edgecolor="white")
    if run.task == "cartpole":
        ax.set_xlim(-5, 505)
    ax.axvline(scores.mean(), color="black", linestyle="--", label=f"Mean = {scores.mean():.2f}")
    ax.set(xlabel="Episode reward (raw points)", ylabel="Number of episodes",
           title=f"{task_title}: 500 frozen-policy evaluation episodes")
    ax.legend(frameon=False)
    _save_figure(fig, paths["evaluation"])
    _diagnostic_plot(run, paths["diagnostic"])
    return paths


STYLES = getSampleStyleSheet()
STYLES.add(ParagraphStyle(name="BodyHW", fontName="Times-Roman", fontSize=10.5,
                         leading=13, spaceAfter=6, textColor=colors.black))
STYLES.add(ParagraphStyle(name="SmallHW", parent=STYLES["BodyHW"], fontSize=9,
                         leading=11, spaceAfter=3))
STYLES.add(ParagraphStyle(name="TitleHW", fontName="Times-Bold", fontSize=14,
                         leading=17, spaceAfter=7))
STYLES.add(ParagraphStyle(name="HeadingHW", fontName="Times-Bold", fontSize=12,
                         leading=15, spaceAfter=7))
STYLES.add(ParagraphStyle(name="CodeHW", fontName="Courier", fontSize=7.3,
                         leading=9, spaceAfter=8))


def p(text, small=False):
    return Paragraph(text, STYLES["SmallHW" if small else "BodyHW"])


def heading(text):
    return Paragraph(text, STYLES["HeadingHW"])


def table(rows, widths):
    data = [[p(f"<b>{escape(str(value))}</b>" if i == 0 else escape(str(value)), small=True)
             for value in row] for i, row in enumerate(rows)]
    result = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    result.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LINEABOVE", (0, 0), (-1, 0), 0.5, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.4, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, colors.black)]))
    return result


def figure(path, height):
    return Image(str(path), width=7.0*inch, height=height*inch, kind="proportional")


def methods(cart, pac, repo_url):
    rows = [["Parameter", "CartPole-v1", "MsPacman-v5"]]
    for key, label in [("gamma", "Discount factor"), ("learning_rate", "Adam learning rate"),
                       ("batch_size", "Batch size"), ("buffer_size", "Replay capacity"),
                       ("learning_starts", "Warm-up decisions"), ("train_frequency", "Decisions per update"),
                       ("target_interval", "Updates per target copy"),
                       ("epsilon_decay_steps", "Epsilon decay decisions"),
                       ("epsilon_end", "Final training epsilon"), ("seed", "Training seed")]:
        rows.append([label, cart.config[key], pac.config[key]])
    return [Paragraph("MLDS 490 - Assignment 2: Deep Q-Network", STYLES["TitleHW"]),
            p("Zikun Zheng | GitHub: ZikunZheng1129 | Repository: mlds490-hw2"),
            p(f'<link href="{escape(repo_url)}" color="black">{escape(repo_url)}</link>'),
            heading("Method"),
            p("Both tasks use standard DQN: a neural Q function, uniform replay, a target network, "
              "and epsilon-greedy exploration. The target is <b>y = r + gamma (1 - terminated) "
              "max Q_target(next state, action)</b>. Adam minimizes mean Huber loss against detached "
              "targets; gradient norm is clipped at 10. Epsilon decreases linearly from 1."),
            table(rows, [240, 132, 132]), Spacer(1, 8),
            p("<b>Networks.</b> CartPole: 4 inputs, two 128-unit ReLU layers, 2 Q outputs. "
              "MsPacman: four 88 x 80 frames; convolution layers 32 (8 x 8, stride 4), "
              f"64 (4 x 4, stride 2), and 64 (3 x 3, stride 1); {pac.config.get('cnn_hidden_dim', 256)} "
              f"ReLU units; 9 Q outputs ({escape(pac.config.get('profile', 'baseline'))} profile). "
              "Outputs are raw action values, with no softmax."),
            p("<b>MsPacman preprocessing.</b> The assignment's crop, grayscale and contrast mask are used. "
              "Replay stores uint8 images; the network computes (pixel - 128)/128. Four frames provide "
              "motion information. Training rewards are sign-clipped; all reported scores are raw game "
              "points. ALE/MsPacman-v5 uses frame skip 4, sticky-action probability 0.25 and 9 actions. "
              "Life loss does not end a full-game episode."),
            p("<b>Evaluation.</b> The highest greedy development mean selects the saved model; ties keep "
              f"the earlier model. Development seeds start at {cart.config.get('development_seed', 40000)} "
              f"(CartPole) and {pac.config.get('development_seed', 40000)} (MsPacman). Each selected model is evaluated "
              "without updates or exploration on 500 complete episodes with independent seeds. These "
              "scores never select the model. Standard deviations below use ddof=1."),
            p("<b>Validation.</b> Random, untrained and selected policies share held-out seeds "
              "starting at 80000. Untrained and selected policies are greedy; the random policy samples "
              "uniformly. Paired 95% intervals use 2,000 bootstrap resamples. These compare learned "
              "behavior beyond software tests. Different actions can produce different trajectories "
              "even with matched seeds. Evaluation episodes do not measure variation across retraining."),
            p("<b>Shared edge cases.</b> Both termination and truncation end rollouts, but only true "
              "termination removes the Bellman bootstrap. Learning waits for warm-up and a full batch. "
              "Replay is bounded and copies observations. Non-finite losses fail visibly. Tests cover "
              "these branches, target-network freezing and unchanged weights during evaluation.")]


def training_page(run, paths):
    name = "CartPole" if run.task == "cartpole" else "MsPacman"
    rewards = _numbers(run.training, "reward")
    window = min(100, len(rewards))
    first, last = rewards[:window].mean(), rewards[-window:].mean()
    story = [heading(f"{name}: training behavior"), figure(paths["max_q"], 2.65),
            p("<b>Maximum Q.</b> Each point is the maximum over all visited states and all action "
              "outputs in that training episode. The online network changes during the episode. "
              "Q values are predictions, not measured game scores; a larger Q alone does not prove learning.", True),
            figure(paths["rewards"], 2.65),
            p(f"The first and final {window} training episodes average <b>{first:.2f}</b> and "
              f"<b>{last:.2f}</b> raw reward, respectively (difference {last-first:+.2f}). "
              "The 100-episode moving average begins only when 100 complete episodes exist. "
              "Exploration changes over training, so the fixed-policy held-out comparisons on the next "
              "page provide a separate check of actual learned behavior."),
            p(f"The run completed <b>{len(run.training):,}</b> training episodes, "
              f"{run.metrics['environment_steps']:,} environment decisions and "
              f"{run.metrics['optimizer_steps']:,} optimizer updates in "
              f"{run.metrics['wall_seconds']/3600:.2f} hours, including final evaluation. "
              f"Frozen evaluation uses the development-selected checkpoint from episode "
              f"<b>{run.metrics['selected_checkpoint']['episode']}</b>. The training curve includes "
              "exploratory actions and subsequent parameter updates; its final moving average therefore "
              "need not match the selected checkpoint's greedy evaluation score.")]
    if run.task == "cartpole":
        peak = _numbers(run.training, "max_q").max()
        bound = 1 / (1 - run.config["gamma"])
        interpretation = ("The excess indicates Q-value overestimation or function-approximation error; "
                          "Q growth is not a correctness certificate." if peak > bound else
                          "Remaining below this bound does not by itself establish accurate Q estimates.")
        story.append(p(f"The largest predicted Q is <b>{peak:.2f}</b>. With reward 1 per step and "
                       f"gamma={run.config['gamma']}, discounted return cannot exceed <b>{bound:.2f}</b>. "
                       + interpretation))
    else:
        recent = _numbers(run.development[-5:], "mean")
        changes = np.diff(recent)
        interpretation = (
            "Late development scores rise and fall; the run does not demonstrate stable or monotonic convergence."
            if np.any(changes > 0) and np.any(changes < 0) else
            "These development checks alone do not establish stable convergence.")
        story.append(p(f"The last {len(recent)} development means range from {recent.min():.1f} "
                       f"to {recent.max():.1f} points, ending at {recent[-1]:.1f}. "
                       + interpretation + " Development scores select a checkpoint; the independent "
                       "500-episode result on the next page is the final performance estimate."))
    return story


def evaluation_page(run, paths):
    name = "CartPole" if run.task == "cartpole" else "MsPacman"
    scores, v = _numbers(run.evaluation, "reward"), run.validation
    rows = [[f"{v['sampling']['episodes_per_condition']}-seed validation policy", "Mean reward", "Sample SD"]]
    for label, key in [("Random", "random_policy"), ("Untrained", "initial_policy"),
                       ("Selected", "final_policy")]:
        info = v[key]
        rows.append([label, f"{info['mean_reward']:.2f}", f"{info['std_reward']:.2f}"])
    story = [heading(f"{name}: evaluation and validation"), figure(paths["evaluation"], 2.0),
             p(f"<b>500 episodes:</b> mean {scores.mean():.2f}, sample SD {scores.std(ddof=1):.2f}, "
               f"range {scores.min():.0f}-{scores.max():.0f}. "
               f"{sum(_true(r['terminated']) for r in run.evaluation)} true terminations; "
               f"{sum(_true(r['truncated']) for r in run.evaluation)} time truncations. "
               f"Evaluation seeds: {run.config['evaluation_seed']}-{run.config['evaluation_seed']+499}."),
             table(rows, [270, 117, 117]), Spacer(1, 5)]
    for label, key in [("Selected minus random", "final_vs_random"),
                       ("Selected minus untrained", "final_vs_initial")]:
        c = v["comparison"][key]
        story.append(p(f"{label}: {c['mean_difference']:.2f}; 95% paired interval "
                       f"[{c['ci95_low']:.2f}, {c['ci95_high']:.2f}].", True))
    story += [p(learning_statement(v["comparison"]["final_vs_random"]), True),
              figure(paths["diagnostic"], 2.1)]
    story += diagnostic_page(run, paths)
    return story


def diagnostic_page(run, paths):
    v, a = run.validation["perturbation"], run.validation["target_audit"]
    if run.task == "cartpole":
        success = sum(float(r["length"]) == 500 and not _true(r["terminated"]) for r in run.evaluation)
        text = (f"<b>Objective and edge cases.</b> {success}/500 standard episodes reached 500 steps. "
                "The nonstandard recovery map tests starting angles and angular velocities beyond "
                "normal resets, with small independent jitters. Each cell shows successes/trials. "
                f"Across {v['grid_evaluations']} trials, {100*v['overall_success_fraction']:.1f}% "
                f"reached 500 steps; mean length was {v['mean_length']:.1f}. "
                "Difficult initial states reveal a recovery limit. Time-limit endings keep the "
                "bootstrap term; physical failure removes it.")
    else:
        c = v["normal_minus_blackout"]
        text = ("<b>Visual-input edge case.</b> The nonstandard plot compares normal observations with "
                "all-zero images on matched seeds; points above the diagonal favor normal vision. "
                f"Blackout mean: {v['blackout_policy']['mean_reward']:.2f}. Normal minus blackout: "
                f"{c['mean_difference']:.2f}, 95% interval [{c['ci95_low']:.2f}, {c['ci95_high']:.2f}]. "
                "Only network inputs change; full-game endings and raw scoring remain intact. "
                "Blackout also introduces distribution shift, so it is not a pure test of perception.")
    return [p(text), p(f"Ten-step target audit: {a['transitions_checked']} recorded transitions, "
             f"all match: {a['all_checked_targets_match']}. Terminal: {a['terminal_status']}; "
             f"truncation: {a['truncation_status']}. Unobserved cases are not passes. "
             "The full states/image summaries, Q values, actions and rewards are retained in "
             "validation_first10_steps.json.", True)]


def _analytic_section(science):
    cart, pac = [science[task]["analytic_learning"] for task in ("cartpole", "mspacman")]
    rows = [["Known MDP: gamma / reward scale", "CartPole check: error / result", "MsPacman check: error / result"]]
    for a, b in zip(cart["cases"], pac["cases"]):
        if (a["gamma"], a["reward_scale"]) != (b["gamma"], b["reward_scale"]):
            raise ValueError("Known-MDP cases must match between tasks")
        rows.append([f"{a['gamma']:g} / {a['reward_scale']:g}",
                     *[f"{case['max_absolute_error']:.2g} / {'pass' if case['passed'] else 'FAIL'}"
                       for case in (a, b)]])
    controls = []
    for label, result in (("CartPole", cart), ("MsPacman", pac)):
        control = result["zero_learning_rate_control"]
        controls.append(f"{label}: error {control['max_absolute_error']:.3g}, "
                        f"weights unchanged={control['weights_unchanged']}, "
                        f"no-learning detected={result['control_detected_no_learning']}")
    return [p("<b>Learning a known MDP.</b> Actual DQN updates learn a two-state problem with exact "
              "Q* = [[gamma x scale, 0], [scale, 0]]. Maximum absolute errors and recorded "
              f"pass/fail checks use tolerance {cart['cases'][0]['tolerance']:g} after "
              f"{cart['cases'][0]['updates']} updates; this tests Bellman learning, not pixel-game competence.", True),
            table(rows, [180, 162, 162]),
            p("The <b>zero learning rate</b> negative control should remain unlearned. " + "; ".join(controls) + ".", True)]


def _calibration_section(science):
    rows = [["Within-game states", "Mean Q", "Mean G", "Q - G", "MAE", "Pearson r"]]
    notes = []
    for task, result in science.items():
        name = "CartPole" if task == "cartpole" else "MsPacman"
        sample = result.get("sampled_calibration")
        if sample is None:
            notes.append(f"{name}: within-game calibration not available")
            continue
        correlation = sample["pearson_correlation"]
        rows.append([f"{name} ({sample['observations']})",
                     *[f"{sample[key]:.3f}" for key in ("mean_predicted_q", "mean_discounted_return",
                                                       "mean_signed_error", "mae")],
                     "undefined" if correlation is None else f"{correlation:.3f}"])
        notes.append(f"{name}: {sample['censored_episodes']}/{sample['episodes']} games truncated, "
                     f"maximum omitted tail {sample['maximum_tail_upper_bound']:.4g}")
    endings = []
    for task, result in science.items():
        audit = result["ending_target_audit"]
        branches = "; ".join(f"{key} {audit[key]['count']} ({audit[key]['status']})"
                             for key in ("terminal", "truncation"))
        endings.append(f"{task}: {branches}; all observed targets match={audit['all_observed_targets_match']}")
    return [p("<b>Value calibration.</b> Frozen greedy Q(s,a) is compared with the observed discounted "
              "return G from the same state and trajectory: raw CartPole / sign-clipped MsPacman rewards.", True),
            table(rows, [154, 70, 70, 70, 70, 70]),
            p("; ".join(notes) + ". Truncation omits a nonnegative tail: the true Q-minus-return error "
              "can be smaller by at most this bound. States within a game are dependent. Nearly constant "
              "returns make correlation hard to interpret; good scores do not imply accurate Q values.", True),
            p("<b>Actual ending targets.</b> " + " | ".join(endings) + ". Unobserved branches are not passes.", True)]


def _intervention_section(science):
    intervention = science["mspacman"].get("pixel_interventions")
    if intervention is None:
        return [p("MsPacman image interventions: not available.", True)]
    conditions = intervention["conditions"]
    rows = [["MsPacman input", "Mean game score", "Sample SD", "Normal minus input: paired 95% CI"]]
    for label in ("normal", "repeat_latest", "constant128"):
        condition = conditions[label]
        summary = condition["summary"]
        c = condition.get("normal_minus_condition")
        difference = ("reference" if c is None else
                      f"{c['mean_difference']:.2f} [{c['ci95_low']:.2f}, {c['ci95_high']:.2f}]")
        rows.append([label, f"{summary['mean_reward']:.2f}", f"{summary['std_reward']:.2f}", difference])
    return [p(f"<b>History and image interventions.</b> Same {intervention['episodes_per_condition']} seeds "
              f"starting at {intervention['seed_start']}; repeat_latest repeats the newest frame in all "
              "four channels, constant128 becomes zero after normalization.", True),
            table(rows, [116, 90, 70, 228]),
            p("Only policy inputs change; both interventions introduce distribution shift and do not "
              "isolate causality. An interval including zero gives no clear positive effect. "
              "Selected network/target weights and checkpoint files unchanged: "
              + ", ".join(f"{task}={result['weights_unchanged']}" for task, result in science.items()) + ".", True)]


def _comparison_section(comparison):
    if comparison is None:
        return []
    selection, evaluation = comparison["selection"], comparison["evaluation"]
    means = "; ".join(f"{key} {comparison[key]['mean_reward']:.2f} (SD {comparison[key]['std_reward']:.2f})"
                      for key in ("baseline", "candidate", "random"))
    intervals = []
    for label in ("baseline", "random"):
        c = comparison["comparisons"][f"candidate_minus_{label}"]
        conclusion = ("positive" if c["ci95_low"] > 0 else
                      "negative" if c["ci95_high"] < 0 else "includes zero")
        intervals.append(f"candidate minus {label}: {c['mean_difference']:.2f} "
                         f"[{c['ci95_low']:.2f}, {c['ci95_high']:.2f}] ({conclusion})")
    return [p("<b>MsPacman upgrade comparison.</b> Development means: baseline "
              f"{selection['baseline_development_mean']:.2f}, candidate {selection['candidate_development_mean']:.2f}; "
              f"candidate selected={selection['candidate_selected']}. Promotion uses development scores only. "
              f"All three policies then use the same {evaluation['episodes']} fresh final seeds "
              f"{evaluation['seed_start']}-{evaluation['seed_end']}; these scores never select a model. "
              + means + ". Paired 95% intervals: " + "; ".join(intervals) + ". "
              "The upgrade changes architecture and hyperparameters as well as training budget; "
              "it does not isolate the effect of extra episodes. Each MsPacman configuration has "
              "one training seed; intervals measure evaluation-seed variation only.", True)]


def supplemental_page(supplemental):
    story = [heading("Additional measured checks")]
    science = supplemental["science"]
    if science:
        story += _analytic_section(science) + _calibration_section(science) + _intervention_section(science)
    story += _comparison_section(supplemental["comparison"])
    retraining = supplemental["retraining"]
    if retraining is not None:
        runs = []
        for run in retraining["runs"]:
            runs.append(f"seed {run['seed']}: {run['mean_reward']:.2f} (SD {run['std_reward']:.2f}), "
                        f"{run['successes']}/{run['episodes']} successes, trained {run['training_episodes']} "
                        f"episodes / selected {run['selected_episode']}")
        story.append(p("<b>CartPole retraining controls.</b> " + "; ".join(runs) + ". "
                       f"The primary seed {retraining['primary_seed']} is retained; controls are not selected "
                       "by test scores. A few runs give limited evidence about training-seed robustness.", True))
    return story


def training_command(run):
    c = run.config
    stop = "" if c.get("early_stop", True) else " --no-early-stop"
    return (f"python train.py --task {run.task} --episodes {c['requested_episodes']} "
            f"--seed {c['seed']} --learning-rate {c['learning_rate']} \\\n"
            f"  --profile {c.get('profile', 'baseline')} --development-seed {c.get('development_seed', 40000)} "
            f"--evaluation-seed {c['evaluation_seed']} \\\n"
            f"  --eval-every {c['eval_every']} --eval-episodes {c['development_episodes']} "
            f"--epsilon-decay-steps {c['epsilon_decay_steps']} \\\n"
            f"  --device cuda:0 --output results/rerun/{run.task}{stop}\n")


def reproducibility(cart, pac):
    hardware = "; ".join(f"{name}: {escape(str(run.config['runtime']['gpu']))} on "
                         f"{escape(str(run.config['runtime']['hostname']))}, visible GPU "
                         f"{escape(str(run.config['runtime']['cuda_visible_devices']))}"
                         for name, run in (("CartPole", cart), ("MsPacman", pac)))
    return [heading("Running the code and validation"),
            p(hardware + ". "
              "Use Python 3.12 and run from the repository root. Select one available GPU on deepdish. "
              "The fresh folders preserve supplied records. For the baseline comparison, replace the MsPacman "
              "training command below with README's background/freeze procedure before starting."),
            Preformatted("python3.12 -m venv .venv\n"
              "source .venv/bin/activate\n"
              "python -m pip install --upgrade pip\n"
              "python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126\n"
              "python -m pip install -r requirements.txt\n"
              "export CUDA_VISIBLE_DEVICES=2\n" +
              training_command(cart) + training_command(pac) +
              f"python validate.py --task cartpole --run results/rerun/cartpole --episodes {cart.validation['sampling']['episodes_per_condition']} \\\n"
              f"  --grid-size {cart.validation['perturbation']['grid_size']} --grid-repeats {cart.validation['perturbation']['repeats_per_cell']} --device cuda:0\n"
              f"python validate.py --task mspacman --run results/rerun/mspacman --episodes {pac.validation['sampling']['episodes_per_condition']} \\\n"
              "  --device cuda:0\n"
              "python scientific_validation.py --run results/rerun/cartpole --device cuda:0\n"
              "python scientific_validation.py --run results/rerun/mspacman --device cuda:0\n"
              "python -m pytest --cov=dqn --cov=train --cov=validate \\\n"
              "  --cov=scientific_validation --cov=compare_runs --cov=build_report\n"
              "python build_report.py --results results/rerun --output hw2_report_rerun.pdf \\\n"
              "  --repo-url https://github.com/ZikunZheng1129/mlds490-hw2", STYLES["CodeHW"]),
            p("Training automatically performs the 500-episode evaluation. Validation compares "
              "random/untrained/selected policies, runs the CartPole recovery grid or MsPacman "
              "blackout experiment, and saves the ten-step target audit. To validate the supplied "
              "models, replace results/rerun/&lt;task&gt; with results/&lt;task&gt;. "
              "Scientific validation adds exact-MDP learning, value calibration, actual ending audits "
              "and frame-history controls. README also gives independent retraining commands. "
              "The JSON/CSV files contain configurations, source hashes, raw outcomes and intervals. "
              "Checkpoints omit replay and random-generator state; reproduction starts a fresh run.")]


def future_experiments():
    return [heading("Given an additional three months"),
            p("<b>Month 1:</b> Repeat each setup with at least 10 training seeds and equal interaction "
              "budgets. Report confidence intervals across training runs, not only evaluation episodes. "
              "Keep development and final test seeds separate."),
            p("<b>Month 2:</b> Change one factor at a time: remove replay, vary target-copy intervals, "
              "compare Huber and squared loss, and vary exploration. For MsPacman, compare one versus "
              "four frames and clipped versus raw training rewards. Compare score, learning speed "
              "and TD error at equal budgets."),
            p("<b>Month 3:</b> Expand CartPole's initial-state grid and test changes in mass, pole length "
              "and observation noise. Test MsPacman occlusion, delayed frames and sticky-action rates. "
              "Inspect failures and confirm improvements on untouched test seeds; retain standard "
              "environments as the main benchmark.")]


def build_report(results_dir, output, repo_url):
    parsed = urlparse(repo_url)
    if parsed.scheme != "https" or parsed.netloc != "github.com" or len(parsed.path.strip("/").split("/")) != 2:
        raise ValueError("--repo-url must be an HTTPS GitHub repository URL")
    cart, pac = load_run(results_dir, "cartpole"), load_run(results_dir, "mspacman")
    supplemental = load_supplemental(results_dir)
    figures = {r.task: plot_run(r) for r in (cart, pac)}
    pages = [methods(cart, pac, repo_url)]
    for run in (cart, pac):
        paths = figures[run.task]
        pages += [training_page(run, paths), evaluation_page(run, paths)]
    if any(supplemental.values()):
        pages += [supplemental_page(supplemental)]
    pages += [reproducibility(cart, pac) + future_experiments()]
    story = []
    for index, page in enumerate(pages):
        story.extend(([PageBreak()] if index else []) + page)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    synthetic = cart.config.get("test_fixture", False) or pac.config.get("test_fixture", False)
    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont("Times-Roman", 9)
        canvas.setFillColor(colors.black)
        canvas.drawCentredString(306, 28, str(document.page))
        if synthetic:
            canvas.setFillColor(colors.red)
            canvas.drawCentredString(306, 770, "SYNTHETIC TEST FIXTURE - NOT ASSIGNMENT RESULTS")
        canvas.restoreState()
    document = SimpleDocTemplate(str(output), pagesize=(612, 792), leftMargin=54,
                                 rightMargin=54, topMargin=48, bottomMargin=48,
                                 title="MLDS 490 Assignment 2 - Deep Q-Networks", author="Zikun Zheng")
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("hw2_report.pdf"))
    parser.add_argument("--repo-url", required=True)
    args = parser.parse_args()
    print(build_report(args.results, args.output, args.repo_url))


if __name__ == "__main__":
    main()
