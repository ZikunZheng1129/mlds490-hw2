# MLDS 490 — Assignment 2

Public repository: https://github.com/ZikunZheng1129/mlds490-hw2

## Results

See [the PDF report](hw2_report.pdf) for all figures, validation results and discussion.
The selected frozen policies were each evaluated on 500 independent episodes.

| Environment | Training episodes | Mean evaluation reward | Sample standard deviation |
|---|---:|---:|---:|
| CartPole-v1 | 1,500 | 500.00 | 0.00 |
| ALE/MsPacman-v5 | 6,000 | 2,589.44 | 593.05 |

CartPole survived all 500 evaluation episodes for 500 steps. Independent
CartPole training seeds 8 and 9 also scored 500.00 ± 0.00; all three policies
survived every episode on the same 500 fresh seeds. The original seed-7 policy
remains the primary model.

MsPacman's first and final 100 training episodes averaged 245.5 and 2,415.0 raw
points. The run used 4,737,156 environment decisions and 1,179,290 optimizer
updates; training plus final evaluation took 5.37 hours on deepdish. The saved
checkpoint came from episode 5,600, with a 20-game development mean of 2,736.5.
Final evaluation data were not used to select the checkpoint or adopt the upgrade.
The 500-game result demonstrates stronger performance without establishing
stable convergence or robustness across MsPacman training seeds.

The original 2,000-episode MsPacman run is preserved in
`results/mspacman_baseline_2000`; its earlier 500-game evaluation averaged
1,420.52 (sample SD 511.69) on a different seed set. Re-evaluating both frozen
checkpoints and random actions on the **same 500 fresh seeds, 200007–200506**,
gave:

| MsPacman policy | Mean game score | Sample standard deviation |
|---|---:|---:|
| Original 2,000-episode model | 1,416.28 | 471.99 |
| Extended 6,000-episode model | 2,589.44 | 593.05 |
| Uniform random actions | 262.54 | 111.44 |

The extended model gained **1,173.16 points (+82.83%)** over the original model;
the paired 95% bootstrap interval for the mean difference was
**[1,107.54, 1,236.62]**. Its gain over random actions was **2,326.90 points**,
with interval **[2,276.00, 2,379.86]**. These intervals measure variation across
evaluation seeds, not across independent training runs. The development-only
adoption decision was frozen before these final scores were inspected.

## Files

| File | Purpose |
|---|---|
| `hw2_report.pdf` | Final English report with measured results and all eight figures |
| `dqn.py` | Environments, MLP/CNN, uniform replay, target network, Huber update |
| `train.py` | Training, development-set checkpoint selection, frozen evaluation |
| `validate.py` | Matched-seed comparisons and task-specific robustness experiments |
| `scientific_validation.py` | Known-solution learning, value calibration and image-history controls |
| `compare_runs.py` | Frozen development-only upgrade decision and matched-seed model comparisons |
| `build_report.py` | Figures and English PDF report from recorded results |
| `tests/` | Unit, real-environment integration, and short end-to-end checks |
| `results/` | Configurations, checkpoints, raw episode records, validation and figures |

## Setup

Use Python 3.12. From this directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# On the deepdish GPU host used for this submission:
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

On a computer without NVIDIA CUDA, install the appropriate PyTorch build and use
`--device cpu`. The submitted training runs use one NVIDIA GPU on deepdish.
`ale-py` supplies the Atari environment; `dqn.py` explicitly registers it with
Gymnasium. No display, game controller, or browser is needed.

## Train and evaluate

Choose **one currently available GPU** on deepdish; the examples select GPU 2.
The output directory must be fresh. Do not overwrite the supplied results when
reproducing an experiment.

```bash
export CUDA_VISIBLE_DEVICES=2

python train.py --task cartpole --episodes 1500 --seed 7 --device cuda \
  --eval-every 50 --eval-episodes 20 --final-episodes 500 \
  --output results/rerun/cartpole
```

Use the MsPacman comparison procedure below to train the extended model and
freeze its adoption decision at the correct point. Run independent training jobs
sequentially when sharing one GPU. Use `nohup` or `screen` if disconnecting from
the server. The complete runtime configuration,
host, GPU name, library versions, source hashes and actual episode count are
recorded in each run's `config.json` and `metrics.json`.

CartPole stops early only after two consecutive development checks achieve
500 steps in every development episode. MsPacman runs its requested number of
episodes. `policy.pt` is selected by mean **development** reward, with ties
keeping the earlier checkpoint. `latest.pt` is the last development checkpoint;
`initial.pt` is the untrained network. Development seeds start at 40000.

Training automatically evaluates the selected, frozen `policy.pt` for **500
complete episodes**. CartPole retains the original evaluation seeds 100007-100506.
The MsPacman upgrade uses fresh seeds 200007-200506 because the old final results
have already been inspected. `--evaluation-seed` sets this range explicitly; its
default remains `100000 + training_seed`. Final scores never select checkpoints. To
repeat a frozen evaluation separately:

```bash
python train.py --evaluate results/cartpole/policy.pt --episodes 500 \
  --seed 7 --device cuda --output results/cartpole_reevaluation
python train.py --evaluate results/mspacman/policy.pt --episodes 500 \
  --evaluation-seed 200007 --device cuda --output results/mspacman_reevaluation
```

Checkpoints contain network and optimizer weights, but not replay contents,
environment state or all random-generator states. There is no exact-resume
claim or resume command. A fresh training run is the supported reproduction.

## MsPacman extended run and baseline comparison

The `extended` profile remains ordinary DQN. Its settings are:

| Setting | Value |
|---|---:|
| Training episodes / seed | 6,000 / 7 |
| Discount factor / Adam learning rate | 0.99 / 0.0001 |
| CNN fully connected width | 512 |
| Batch size / replay capacity | 64 / 100,000 |
| Warm-up decisions / decisions per update | 20,000 / 4 |
| Target copy interval | 2,500 optimizer updates |
| Epsilon schedule | 1 to 0.01 over 1,000,000 decisions |
| Development evaluation | 20 games every 100 training episodes |

Four-frame inputs, clipped training rewards and full-game endings are retained.
Several settings change together, so comparison with the baseline cannot isolate
the effect of extra training episodes.

The original 2,000-episode baseline is archived in
`results/mspacman_baseline_2000`. Its source commit is
`ec92296559e7e3f5c1b92f2405842c0be4ecc166`. Preserve that folder and the reference
files in `results/upgrade_comparison`; the commands below create separate rerun
records. The experiment plan was written before the extended run. On the fixed
development seeds **40000–40019**, the old model averaged **1542.0** points.
Promote the candidate only if its best development mean is **strictly greater
than 1542.0**. Checkpoint ties keep the earlier checkpoint; a tie with the baseline
retains the baseline. The completed candidate reached **2736.5** at episode
**5600**, so the frozen development-only decision adopted it before the final
evaluation results were inspected.

Prepare the unchanged plan and baseline evidence, then start training in the
background in the same terminal. The archived baseline checkpoint must exist
before running this reproduction.

```bash
mkdir -p results/rerun/upgrade_comparison
cp results/upgrade_comparison/experiment_plan.json \
   results/upgrade_comparison/baseline_development20.json \
   results/upgrade_comparison/baseline_development20.csv \
   results/rerun/upgrade_comparison/

python -u train.py --task mspacman --profile extended --episodes 6000 --seed 7 \
  --device cuda --eval-every 100 --eval-episodes 20 --final-episodes 500 \
  --development-seed 40000 --evaluation-seed 200007 \
  --output results/rerun/mspacman > results/rerun/mspacman_training.log 2>&1 &
training_pid=$!
```

Wait for `Development at episode 6000:` in the training log. **Before reading
final evaluation progress, and before `results/rerun/mspacman/evaluation.csv`
exists**, run the following in the original terminal. Training automatically
starts its 500-game final evaluation after that development check; freeze the
decision while this evaluation runs, not after waiting for training to exit.

```bash
python compare_runs.py --baseline-run results/mspacman_baseline_2000 \
  --new-run results/rerun/mspacman --output results/rerun/upgrade_comparison \
  --freeze-only

wait "$training_pid"

python compare_runs.py --baseline-run results/mspacman_baseline_2000 \
  --new-run results/rerun/mspacman --output results/rerun/upgrade_comparison \
  --device cuda
```

`--freeze-only` writes `selection_decision.json` using development evidence only.
It refuses incomplete training or an existing candidate final CSV. The subsequent
comparison rejects changed checkpoints, configuration, plan or development files.
It evaluates the old policy and uniform random actions on the candidate's exact
**500 fresh seeds, 200007–200506**, then writes `comparison.json`,
`baseline_evaluation.csv` and `random_evaluation.csv`. Candidate minus baseline
and candidate minus random use paired seed differences and 95% bootstrap
intervals. These final scores describe the frozen policies and never change the
adoption decision. A single MsPacman training seed does not establish robustness
across retraining.

## CartPole training-seed controls

The completed controls all reached 500 steps in **500/500** fresh test episodes,
with mean **500.00** and sample SD **0.00**:

| Training seed | Training episodes | Selected episode | Successes |
|---|---:|---:|---:|
| 7 (primary) | 1,500 | 1,300 | 500/500 |
| 8 | 1,900 | 950 | 500/500 |
| 9 | 3,000 | 1,050 | 500/500 |

The three runs support repeatability for these seeds, but do not establish
robustness across all initializations.

Keep the original seed-7 model as the primary result. Train seeds 8 and 9 as
separate controls, allowing up to 3,000 episodes. Early stopping still requires
two consecutive development checks where all 20 games reach 500 steps. Evaluate
all three frozen models on the same fresh 500 seeds; final scores never select
among the three training seeds.

```bash
for seed in 8 9; do
  python train.py --task cartpole --episodes 3000 --seed "$seed" --device cuda \
    --eval-every 50 --eval-episodes 20 --final-episodes 500 \
    --development-seed 40000 --evaluation-seed 200007 \
    --output "results/cartpole_retraining/seed$seed"
done

python train.py --evaluate results/cartpole/policy.pt --episodes 500 \
  --seed 7 --evaluation-seed 200007 --device cuda \
  --output results/cartpole_retraining/seed7_evaluation

python compare_runs.py --cartpole-controls results/cartpole_retraining
```

Use fresh folders for a repeat: replace the control root with a new directory
beside `results/cartpole`. The summary command reads that sibling primary run,
the two control runs and `seed7_evaluation/evaluation.csv`, checks the 500 matching
seeds, and writes `summary.json`. Raw data are retained; the primary checkpoint is
unchanged. To include this summary in a rerun report, use
`results/rerun/cartpole_retraining` beside its `results/rerun/cartpole` primary
run and point the seed-7 reevaluation at that run's checkpoint.

## Definitions and edge cases

- Discount factors are **0.95** for CartPole and **0.99** for MsPacman.
- MsPacman uses the v5 environment with **9 actions**, frame skip 4 and sticky
  action probability 0.25. A lost life does not end a full-game episode.
- The assignment's crop, grayscale and contrast transformation produces an
  88 × 80 image. Four recent images are stacked; unsigned-byte storage saves
  replay memory, and the network uses `(pixels - 128) / 128`.
- MsPacman's TD targets use sign-clipped rewards. All recorded episode rewards,
  histograms, validation comparisons and performance statistics use **original
  game points**, not clipped rewards. Its Q-values therefore measure discounted
  clipped reward, not game points.
- A genuine terminal state has target `reward`. A time-limit truncation retains
  the next-state bootstrap term. Either condition resets the environment.
- Replay sampling starts only after enough records exist. Stored observations
  are copied, so later environment changes cannot corrupt old transitions.
- `max_q` is the maximum prediction over **all visited states and all actions
  in that training episode**, evaluated with the online network at each step.
- The reward moving average uses a full trailing window of **100 episodes**;
  the first 99 values are left undefined. Standard deviations use `ddof=1`.

## Validation experiments

```bash
python validate.py --task cartpole --run results/rerun/cartpole --episodes 50 --device cuda
python validate.py --task mspacman --run results/rerun/mspacman --episodes 50 --device cuda
python scientific_validation.py --run results/rerun/cartpole --episodes 50 --device cuda
python scientific_validation.py --run results/rerun/mspacman --episodes 50 --device cuda
python build_report.py --results results/rerun --output hw2_report_rerun.pdf \
  --repo-url https://github.com/ZikunZheng1129/mlds490-hw2
```

These commands validate the new runs above. To recheck the supplied runs instead,
use `--run results/cartpole` and `--run results/mspacman`.

Each validation compares a uniform random policy, the untrained checkpoint,
and the trained checkpoint on the **same 50 seeds**, separate from development
and final evaluation. Both learned-network checkpoints use epsilon 0. Paired
bootstrap confidence intervals quantify improvement; a positive training curve
alone is not treated as proof of learning. The first 10 evaluation steps are
saved with actions, Q-values, rewards and Bellman targets for inspection.

CartPole additionally evaluates a grid of initial pole angles and angular
velocities with small seeded perturbations. The resulting success/length map is
its non-standard figure and shows where balancing generalizes or fails.
MsPacman compares normal vision with blacked-out observations on matched seeds;
the game itself and its raw scoring rules are unchanged. The paired-score figure
tests whether learned performance depends on visual information. Actual results,
including inconclusive or contrary outcomes, are reported without replacing them
with expected results.

The additional scientific experiments use the actual DQN update to learn a
small two-state problem with a known solution. Discount and reward-scale changes
have analytic answers, and a zero-learning-rate control must fail to learn.
On 50 new game seeds starting at 300000, the selected frozen network's action
values are compared with subsequent discounted returns at up to 50 evenly spaced
states per game. MsPacman uses clipped rewards for this value comparison; game
scores remain raw. Truncated trajectories omit a tail: retained states have at
least 100 observed future decisions, and the maximum omitted return is recorded.
States from one game are dependent and are not treated as independent trials.

The same 50 MsPacman seeds compare intact observations, repetition of the newest
frame, and constant-128 pixels (zero after normalization). These remove temporal
history or visual information without changing game scoring. They also create
input distribution shifts, so their effects are interpreted cautiously. Actual
terminal and truncated endings are separately checked against Bellman targets;
an unobserved branch is recorded as `not_observed`. Online/target weight
fingerprints and the checkpoint file hash verify that evaluation stays frozen.
Raw records and summaries are saved beside each checkpoint. Negative or
inconclusive scientific findings remain in the report.

Measured scientific checks:

- All five known-solution cases passed for both task checks (maximum absolute
  error below 0.000001; tolerance 0.02). The zero-learning-rate control retained
  unchanged weights and an error of 2.799, correctly exposing failed learning.
- Across 2,500 sampled states in 50 MsPacman games, mean Q was 14.845 versus
  observed discounted return 13.372, with signed bias +1.473, MAE 4.295 and
  correlation 0.828. None of these games was truncated. CartPole had bias +0.521,
  MAE 0.521 and correlation -0.224; its nearly constant returns and omitted
  truncation tail (at most 0.1184 per sampled state) limit interpretation.
- On the same 50 scientific-test seeds, normal MsPacman input averaged 2,622.20
  points, repeated-latest-frame input 1,894.00, and constant-128 input 70.00.
  Normal minus repeated-frame input was 728.20 points, with paired 95% interval
  [486.60, 1,008.60]. Input distribution changes prevent a pure causal claim.
- All 50 observed CartPole truncation targets and all 50 MsPacman terminal
  targets matched the Bellman rule. The opposite ending types were not observed.
  Both selected checkpoints and their online/target weights remained unchanged.

Software tests are a separate correctness layer, not a substitute for these RL
experiments:

```bash
python -m pytest -q --cov=dqn --cov=train --cov=validate --cov=scientific_validation --cov=compare_runs --cov=build_report --cov-report=term-missing
python build_report.py --repo-url https://github.com/ZikunZheng1129/mlds490-hw2
```

The report includes the additional-three-months discussion, limitations, all
required figures, the public repository link, and the instructions above.
