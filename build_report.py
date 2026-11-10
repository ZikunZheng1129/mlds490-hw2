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
        im = ax.imshow(rates, origin="lower", vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
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
        ax.scatter(x, y, c="#155E75", alpha=0.7, s=28, edgecolors="white", linewidth=0.4)
        ax.plot([0, limit], [0, limit], "--", color="#9A3412", label="Equal game score")
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
    ax.plot(episodes, _numbers(run.training, "max_q"), color="#155E75", linewidth=1)
    ax.set(xlabel="Training episode", ylabel=f"Maximum Q ({unit} units)", title=f"{task_title}: maximum predicted action value")
    ax.grid(alpha=0.15)
    _save_figure(fig, paths["max_q"])
    fig, ax = plt.subplots(figsize=(7.2, 2.65))
    ax.plot(episodes, rewards, color="#94A3B8", alpha=0.6, linewidth=0.7, label="Episode reward")
    ax.plot(episodes, moving_average(rewards), color="#C2410C", linewidth=1.6, label="Last 100 episodes")
    ax.set(xlabel="Training episode", ylabel="Episode reward (raw points)", title=f"{task_title}: training returns")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(alpha=0.15)
    _save_figure(fig, paths["rewards"])
    scores = _numbers(run.evaluation, "reward")
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    bins = (np.arange(-5, 506, 10) if run.task == "cartpole" else
            25 if np.ptp(scores) > 0 else [scores[0] - 0.5, scores[0] + 0.5])
    ax.hist(scores, bins=bins, color="#155E75", alpha=0.85, edgecolor="white")
    if run.task == "cartpole":
        ax.set_xlim(-5, 505)
    ax.axvline(scores.mean(), color="#C2410C", linestyle="--", label=f"Mean = {scores.mean():.2f}")
    ax.set(xlabel="Episode reward (raw points)", ylabel="Number of episodes",
           title=f"{task_title}: 500 frozen-policy evaluation episodes")
    ax.legend(frameon=False)
    _save_figure(fig, paths["evaluation"])
    _diagnostic_plot(run, paths["diagnostic"])
    return paths


STYLES = getSampleStyleSheet()
STYLES.add(ParagraphStyle(name="BodyHW", fontName="Helvetica", fontSize=9.5, leading=13.4,
                         spaceAfter=8, textColor=colors.HexColor("#1E293B")))
STYLES.add(ParagraphStyle(name="SmallHW", parent=STYLES["BodyHW"], fontSize=8, leading=10.5, spaceAfter=5))
STYLES.add(ParagraphStyle(name="TitleHW", fontName="Helvetica-Bold", fontSize=25, leading=30,
                         textColor=colors.HexColor("#12354A"), spaceAfter=15))
STYLES.add(ParagraphStyle(name="HeadingHW", parent=STYLES["Heading2"], fontSize=15,
                         leading=19, textColor=colors.HexColor("#12354A"), spaceAfter=11))
STYLES.add(ParagraphStyle(name="CodeHW", fontName="Courier", fontSize=7.3, leading=10,
                         backColor=colors.HexColor("#F1F5F9"), borderPadding=7, spaceAfter=12))


def p(text, small=False):
    return Paragraph(text, STYLES["SmallHW" if small else "BodyHW"])


def heading(text):
    return Paragraph(text, STYLES["HeadingHW"])


def table(rows, widths):
    data = [[p(escape(str(value)), small=True) for value in row] for row in rows]
    result = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    result.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor("#94A3B8"))]))
    return result


def figure(path, height):
    return Image(str(path), width=7.0*inch, height=height*inch, kind="proportional")


def comparison_text(label, result):
    return (f"{label}: mean paired difference {result['mean_difference']:.2f} points; "
            f"95% paired-bootstrap interval [{result['ci95_low']:.2f}, {result['ci95_high']:.2f}] "
            f"(n={result['n_pairs']} pairs). {learning_statement(result)}")


def overview(cart, pac, repo_url):
    story = [Paragraph("Deep Q-Networks", STYLES["TitleHW"]),
             p("<b>MLDS 490 - Assignment 2</b><br/>Zikun Zheng<br/>November 2026"),
             p(f"GitHub username: <b>ZikunZheng1129</b><br/>Repository: <b>mlds490-hw2</b><br/>"
               f'<link href="{escape(repo_url)}" color="#155E75">{escape(repo_url)}</link>'),
             Spacer(1, 12), heading("Scope and measured results"),
             p("This assignment implements standard DQN separately for CartPole-v1 and "
               "ALE/MsPacman-v5. Both agents use a neural action-value function, uniform replay, "
               "a periodically copied target network and epsilon-greedy exploration. "
               "All figures and statistics below are generated from the accompanying saved records.")]
    rows = [["Measure", "CartPole-v1", "MsPacman-v5"]]
    for label, fun in [("Training episodes", lambda r: str(len(r.training))),
                       ("Environment decisions", lambda r: f"{r.metrics['environment_steps']:,}"),
                       ("Selected checkpoint episode", lambda r: r.metrics["selected_checkpoint"]["episode"]),
                       ("Frozen evaluation episodes", lambda r: len(r.evaluation)),
                       ("Mean raw reward", lambda r: f"{_numbers(r.evaluation, 'reward').mean():.2f}"),
                       ("Sample standard deviation", lambda r: f"{_numbers(r.evaluation, 'reward').std(ddof=1):.2f}")]:
        rows.append([label, fun(cart), fun(pac)])
    story += [table(rows, [220, 142, 142]), Spacer(1, 13),
              p("<b>Interpretation.</b> CartPole is judged by survival to its 500-step time limit. "
                "MsPacman has no score threshold in the assignment; improvement is assessed using "
                "training curves and frozen-policy comparisons to untrained and random policies."),
              p("<b>Evidence provided.</b> Each task includes maximum-Q and reward curves, a 500-episode "
                "reward histogram, a task-specific nonstandard visualization, controlled validation "
                "experiments, edge-case discussion and an auditable ten-step trace. The final pages "
                "document how to reproduce the work and the experiments proposed for three additional months."),
              p("<b>Scope of claims.</b> Evaluation episodes measure variation across initial seeds and "
                "environment stochasticity. A single training seed per selected run does not establish "
                "robustness across independent retraining runs.")]
    return story


def methods(cart, pac):
    a, b = cart.config, pac.config
    rows = [["Setting", "CartPole-v1", "MsPacman-v5"]]
    labels = [("gamma", "Discount factor"), ("learning_rate", "Adam learning rate"),
              ("batch_size", "Batch size"), ("buffer_size", "Replay capacity"),
              ("learning_starts", "Warm-up decisions"), ("train_frequency", "Decisions per update"),
              ("target_interval", "Optimizer updates per target copy"),
              ("epsilon_decay_steps", "Linear exploration-decay decisions"),
              ("epsilon_end", "Final training epsilon"), ("seed", "Training seed")]
    rows += [[label, a[key], b[key]] for key, label in labels]
    story = [heading("Method, environments and hyperparameters"),
             p("For a sampled transition, the regression target is <b>y = r + gamma (1 - terminated) "
               "max Q_target(next state, action)</b>. The online network minimizes mean Huber loss "
               "against this detached target. Adam updates only the online parameters; gradient norm "
               "is clipped at 10. Target parameters are copied after the specified number of optimizer updates."),
             table(rows, [230, 137, 137]), Spacer(1, 10),
             p("<b>Networks.</b> CartPole uses 4 inputs, two 128-unit ReLU hidden layers and 2 raw Q outputs. "
               "MsPacman uses four 88 x 80 grayscale frames; convolutions are 32 channels (8 x 8, stride 4), "
               "64 (4 x 4, stride 2), 64 (3 x 3, stride 1), followed by 256 ReLU units and 9 raw Q outputs. "
               "There is no policy softmax."),
             p("<b>Image and reward handling.</b> The assignment's crop, channel averaging and contrast mask "
               "are used. Frames are stored as uint8; the network computes (pixel - 128)/128. Frame stacking "
               "provides short motion history. MsPacman learning uses sign-clipped rewards; all reported "
               "game scores and evaluation rewards are unmodified. CartPole rewards are unmodified."),
             p("<b>Environment fidelity.</b> MsPacman is ALE/MsPacman-v5 with nine actions, frame skip 4 and "
               "sticky-action probability 0.25. A life loss does not end a reported game. CartPole retains "
               "the standard 500-step limit. Both termination and truncation stop a rollout, but only true "
               "termination suppresses the next-state Bellman term. [1-4]"),
             p("<b>Checkpoint selection.</b> Development evaluations use a fixed seed range starting at 40000 "
               "and greedy actions. The highest development mean selects policy.pt. The independent 500 "
               "episodes start at 100000 + training seed and never select the checkpoint. Initial and latest "
               "checkpoints are retained. CartPole may stop after two consecutive development checks in "
               "which every episode reaches 500 steps.")]
    return story


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
            p(f"Frozen evaluation uses the development-selected checkpoint from episode "
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
    rewards = _numbers(run.evaluation, "reward")
    v = run.validation
    rows = [["Held-out policy", "Episodes", "Mean reward", "Sample SD"]]
    for label, key in [("Uniform random (epsilon=1)", "random_policy"),
                       ("Untrained network (epsilon=0)", "initial_policy"),
                       ("Selected network (epsilon=0)", "final_policy")]:
        info = v[key]
        rows.append([label, info["episodes"], f"{info['mean_reward']:.2f}", f"{info['std_reward']:.2f}"])
    story = [heading(f"{name}: frozen evaluation and learning validation"),
             figure(paths["evaluation"], 2.8),
             p(f"<b>Required 500-episode result:</b> mean {rewards.mean():.2f}; sample standard "
               f"deviation {rewards.std(ddof=1):.2f}; range [{rewards.min():.0f}, {rewards.max():.0f}]. "
               f"Evaluation uses epsilon=0 and seeds {run.config['evaluation_seed']} through "
               f"{run.config['evaluation_seed']+499}; no updates or replay sampling occur. "
               f"Ending flags: {sum(_true(r['terminated']) for r in run.evaluation)} true terminations "
               f"and {sum(_true(r['truncated']) for r in run.evaluation)} time truncations."),
             table(rows, [240, 70, 100, 94]), Spacer(1, 9),
             p(f"<b>Controlled learning check.</b> The three policies use the same {v['sampling']['episodes_per_condition']} "
               "held-out reset seeds starting at 80000, separate from checkpoint selection and the required "
               "500-episode evaluation. Initial and selected networks use identical greedy evaluation "
               "settings; the uniform-random control uses epsilon=1. Matched seeds do not imply identical "
               "trajectories once actions differ."),
             p(comparison_text("Selected minus random", v["comparison"]["final_vs_random"])),
             p(comparison_text("Selected minus initial", v["comparison"]["final_vs_initial"]))]
    if run.task == "cartpole":
        success = sum(float(r["length"]) == 500 and not _true(r["terminated"]) for r in run.evaluation)
        verdict = "Every evaluation episode reached the objective." if success == 500 else "The objective was not reached in every evaluation episode."
        story.append(p(f"<b>500-step objective:</b> {success}/500 episodes ({success/5:.1f}%) survived "
                       f"to the time limit without true termination. {verdict}"))
    return story


def diagnostic_page(run, paths):
    name = "CartPole" if run.task == "cartpole" else "MsPacman"
    perturbation, audit = run.validation["perturbation"], run.validation["target_audit"]
    story = [heading(f"{name}: task-specific diagnostic and edge cases"), figure(paths["diagnostic"], 2.85)]
    if run.task == "cartpole":
        story.append(p(f"<b>Nonstandard recovery map.</b> Each cell reports successes/trials for deliberately "
            f"perturbed starting angle and angular velocity, with small independent reset jitters. "
            f"Across {perturbation['grid_evaluations']} trials, survival is "
            f"{100*perturbation['overall_success_fraction']:.1f}% and mean length is "
            f"{perturbation['mean_length']:.1f} steps. Failure near difficult initial states reveals the "
            "policy's recovery boundary; no monotonic pattern is assumed. These changed resets do not "
            "replace the standard 500-episode evaluation."))
    else:
        story.append(p("<b>Nonstandard visual-information intervention.</b> Each point compares normal "
            "and all-zero network inputs under one matched reset seed. Above the diagonal, normal "
            "observations score higher. The environment, action set and raw score remain unchanged; "
            "the zero tensor removes visual information and also creates out-of-distribution inputs."))
        story.append(p(comparison_text("Normal minus blackout", perturbation["normal_minus_blackout"])))
    story.append(p("<b>Auditable first ten steps.</b> The selected policy's validation rollout at seed 80000 "
                   "records observations (or image summaries/hashes), every action value, selected action, "
                   "raw/training rewards, target and ending flags in validation_first10_steps.json.", True))
    rows = [["Step", "Action", "Raw reward", "Training reward", "Target", "Term / trunc"]]
    rows += [[r["step"], r["action"], f"{r['reward']:.0f}", f"{r['train_reward']:.0f}",
              f"{r['target']:.3f}", f"{int(r['terminated'])} / {int(r['truncated'])}"] for r in run.trace]
    story += [table(rows, [43, 53, 85, 105, 100, 118]), Spacer(1, 7),
              p(f"Target audit: {audit['transitions_checked']} transitions; all recorded targets match: "
                f"{audit['all_checked_targets_match']}. Terminal case: {audit['terminal_status']}; "
                f"time-truncation case: {audit['truncation_status']}. A status of not_observed is not a pass. "
                "Separate software checks exercise both target branches and target-network freezing.", True)]
    return story


def training_command(run):
    c = run.config
    stop = "" if c.get("early_stop", True) else " --no-early-stop"
    return (f"python train.py --task {run.task} --episodes {c['requested_episodes']} "
            f"--seed {c['seed']} --learning-rate {c['learning_rate']} \\\n"
            f"  --eval-every {c['eval_every']} --eval-episodes {c['development_episodes']} "
            f"--epsilon-decay-steps {c['epsilon_decay_steps']} \\\n"
            f"  --device cuda:0 --output results/rerun/{run.task}{stop}\n")


def reproducibility(cart, pac):
    rows = [["Recorded compute", "CartPole", "MsPacman"]]
    for key in ("hostname", "gpu", "device", "cuda_visible_devices", "python", "torch", "gymnasium", "ale_py"):
        rows.append([key, cart.config["runtime"].get(key, "not recorded"), pac.config["runtime"].get(key, "not recorded")])
    story = [heading("Reproducibility, validation and boundary handling"),
             table(rows, [145, 179, 180]), Spacer(1, 10),
             p("Run from the repository root with Python 3.12. The commands install the CUDA 12.6 "
               "PyTorch build used on deepdish before the remaining dependencies. On deepdish, set "
               "CUDA_VISIBLE_DEVICES to one available GPU only, then use "
               "--device cuda:0. The recorded visible-device setting above identifies the actual run. "
               "The saved config.json also includes source hashes, library versions and all hyperparameters."),
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
               "python -m pytest --cov=dqn --cov=train --cov=validate --cov=build_report\n"
               "python build_report.py --results results/rerun --output hw2_report_rerun.pdf \\\n"
               "  --repo-url https://github.com/"
               "ZikunZheng1129/mlds490-hw2", STYLES["CodeHW"]),
             p("<b>What each validation establishes.</b> Random/initial/selected comparisons test learned "
               "control with fixed evaluation settings. The CartPole recovery map tests a broader initial-state "
               "distribution. MsPacman's input blackout tests whether visual observations affect behavior. "
               "The ten-step audit exposes the reward/target/action interaction. Paired bootstrap intervals "
               "use 2,000 resamples of matched episode seeds. These experiments supplement, rather than "
               "replace, unit/integration/end-to-end software tests."),
             p("<b>Boundary handling.</b> Learning waits for a full minibatch and warm-up; replay capacity "
               "is bounded and old records are replaced. Stored observations are copied, so later environment "
               "updates cannot corrupt replay. Invalid inputs and non-finite losses fail visibly. True terminals "
               "have reward-only targets; time limits retain bootstrap. MsPacman preserves full-game endings "
               "across life losses, nine actions and unsigned image storage. Short reward histories have no "
               "100-episode average until the complete window exists. Checkpoints are saved atomically; "
               "evaluation freezes weights. Missing result files or fewer than 500 final episodes prevent "
               "report generation.")]
    return story


def future_and_references():
    return [heading("Given an additional three months"),
            p("<b>Month 1 - Repeatability and statistical precision.</b> Repeat each training setup with at "
              "least 10 independent training seeds and an identical interaction budget. Keep development "
              "and test seeds separate. Report per-run learning curves, score distributions, CartPole "
              "500-step success rates and confidence intervals that include between-training-run variation. "
              "This directly addresses the largest limitation of the present single-run study."),
            p("<b>Month 2 - Controlled algorithm checks.</b> Change one factor at a time: remove replay, "
              "remove or vary target-network lag, change replay capacity, compare Huber to squared loss, "
              "and vary exploration schedules. Match environment decisions and evaluation rules across "
              "conditions. Measure final score, learning speed, TD-error distribution and instability. "
              "For MsPacman, compare one frame versus four and clipped versus raw learning rewards, "
              "while always evaluating raw full-game scores."),
            p("<b>Month 3 - Task-specific robustness and failure analysis.</b> Expand CartPole's starting-state "
              "grid and test labeled changes in pole length, cart mass and small observation noise. For "
              "MsPacman, compare controlled image occlusion, delayed observations and several sticky-action "
              "probabilities, preserving the standard v5 setup as the primary benchmark. Review failure "
              "trajectories and stratify results by state difficulty and game progress. Repeat apparent "
              "improvements on untouched test seeds before drawing conclusions."),
            p("<b>Interpretive limits.</b> A high predicted Q is not a direct correctness certificate. "
              "A positive paired interval is evidence for the tested frozen policies, not a guarantee about "
              "all retraining runs. The blackout intervention combines information loss with distribution "
              "shift, and the CartPole stress grid deliberately changes the reset distribution. Neither "
              "diagnostic replaces the required standard-environment results."),
            Spacer(1, 9), heading("References and accompanying evidence"),
            p("[1] MLDS 490, Assignment 2 - Deep Q-Network; Week 6 DQN lecture; Lab Week 6. "
              "Course-provided requirements and algorithm reference.", True),
            p('[2] PyTorch, <link href="https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html" '
              'color="#155E75">Reinforcement Learning (DQN) Tutorial</link>. Neural action values, replay, '
              'Huber loss and detached targets.', True),
            p('[3] Farama Foundation, <link href="https://ale.farama.org/environments/" color="#155E75">'
              'Arcade Learning Environment documentation</link>. Atari environment versions, frame skipping '
              'and sticky actions.', True),
            p('[4] Farama Foundation, <link href="https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/" '
              'color="#155E75">Handling Time Limits</link>. Termination versus truncation for bootstrapping.', True),
            p("The repository includes dqn.py, train.py, validate.py, build_report.py, tests, requirements "
              "and README instructions. Each results directory preserves configuration, training/development "
              "records, the 500 evaluation scores, validation tables, first-ten-step trace and checkpoint "
              "files. Figures are regenerated from those records; building this PDF does not retrain models.", True)]


def build_report(results_dir, output, repo_url):
    parsed = urlparse(repo_url)
    if parsed.scheme != "https" or parsed.netloc != "github.com" or len(parsed.path.strip("/").split("/")) != 2:
        raise ValueError("--repo-url must be an HTTPS GitHub repository URL")
    cart, pac = load_run(results_dir, "cartpole"), load_run(results_dir, "mspacman")
    figures = {r.task: plot_run(r) for r in (cart, pac)}
    pages = [overview(cart, pac, repo_url), methods(cart, pac)]
    for run in (cart, pac):
        paths = figures[run.task]
        pages += [training_page(run, paths), evaluation_page(run, paths), diagnostic_page(run, paths)]
    pages += [reproducibility(cart, pac), future_and_references()]
    story = []
    for index, page in enumerate(pages):
        story.extend(([PageBreak()] if index else []) + page)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    synthetic = cart.config.get("test_fixture", False) or pac.config.get("test_fixture", False)
    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#64748B"))
        canvas.drawString(54, 28, "MLDS 490 | Assignment 2 | Zikun Zheng")
        canvas.drawRightString(558, 28, str(document.page))
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
