# MLDS 490 — Assignment 2


Public repository: https://github.com/ZikunZheng1129/mlds490-hw2

## Results

See [the PDF report](hw2_report.pdf) for all figures, validation results and discussion.
The selected frozen policies were each evaluated on 500 independent episodes.

| Environment | Training episodes | Mean evaluation reward | Sample standard deviation |
|---|---:|---:|---:|
| CartPole-v1 | 1,500 | 500.00 | 0.00 |
| ALE/MsPacman-v5 | 2,000 | 1,420.52 | 511.69 |

CartPole survived all 500 evaluation episodes for 500 steps. MsPacman's first
and final 100 training episodes averaged 257.0 and 1,377.7 raw points. Its late
development scores fluctuated, so this demonstrates improvement without claiming
stable convergence. Final evaluation data were not used to select checkpoints.

## Files

| File | Purpose |
|---|---|
| `hw2_report.pdf` | Final English report with measured results and all eight figures |
| `dqn.py` | Environments, MLP/CNN, uniform replay, target network, Huber update |
| `train.py` | Training, development-set checkpoint selection, frozen evaluation |
| `validate.py` | Matched-seed comparisons and task-specific robustness experiments |
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

python train.py --task mspacman --episodes 2000 --seed 7 --device cuda \
  --eval-every 100 --eval-episodes 10 --final-episodes 500 \
  --output results/rerun/mspacman
```

Run the two training commands sequentially on the selected GPU. Use `nohup` or
`screen` if disconnecting from the server. The complete runtime configuration,
host, GPU name, library versions, source hashes and actual episode count are
recorded in each run's `config.json` and `metrics.json`.

CartPole stops early only after two consecutive development checks achieve
500 steps in every development episode. MsPacman runs its requested number of
episodes. `policy.pt` is selected by mean **development** reward, with ties
keeping the earlier checkpoint. `latest.pt` is the last development checkpoint;
`initial.pt` is the untrained network. Development seeds start at 40000.

Training automatically evaluates the selected, frozen `policy.pt` for **500
complete episodes** with independent seeds starting at `100000 + training_seed`.
These final evaluation scores are never used for checkpoint selection. To
repeat a frozen evaluation separately:

```bash
python train.py --evaluate results/cartpole/policy.pt --episodes 500 \
  --seed 7 --device cuda --output results/cartpole_reevaluation
python train.py --evaluate results/mspacman/policy.pt --episodes 500 \
  --seed 7 --device cuda --output results/mspacman_reevaluation
```

Checkpoints contain network and optimizer weights, but not replay contents,
environment state or all random-generator states. There is no exact-resume
claim or resume command. A fresh training run is the supported reproduction.

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

Software tests are a separate correctness layer, not a substitute for these RL
experiments:

```bash
python -m pytest -q --cov=dqn --cov=train --cov=validate --cov=build_report --cov-report=term-missing
python build_report.py --repo-url https://github.com/ZikunZheng1129/mlds490-hw2
```

The report includes the additional-three-months discussion, limitations, all
required figures, the public repository link, and the instructions above.

