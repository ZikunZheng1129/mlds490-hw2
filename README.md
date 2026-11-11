# MLDS 490 — Assignment 2

[Report](hw2_report.pdf) · [GitHub](https://github.com/ZikunZheng1129/mlds490-hw2)

## Results

Each selected model was evaluated on 500 complete episodes. SD is the sample standard deviation.

| Environment | Training episodes | Mean reward | SD |
|---|---:|---:|---:|
| CartPole-v1 | 1,500 | 500.00 | 0.00 |
| MsPacman-v5 | 6,000 | 2,589.44 | 593.05 |

The report contains the figures, validation results and discussion. Saved models and raw records are in `results/`.

## Setup

Use Python 3.12 and one available GPU on deepdish; GPU 2 is used below.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
export CUDA_VISIBLE_DEVICES=2
```

## Train and evaluate

Run from the repository root, using fresh output folders. Each command selects a checkpoint using development scores, then evaluates it for 500 episodes.
**For the optional baseline comparison, use the expanded procedure below instead of the MsPacman command here.**

```bash
python train.py --task cartpole --episodes 1500 --seed 7 --device cuda \
  --eval-every 50 --eval-episodes 20 --final-episodes 500 \
  --output results/rerun/cartpole

python train.py --task mspacman --profile extended --episodes 6000 --seed 7 --device cuda \
  --eval-every 100 --eval-episodes 20 --final-episodes 500 \
  --development-seed 40000 --evaluation-seed 200007 --output results/rerun/mspacman
```

## Validate and build the report

`validate.py` compares trained, untrained and random policies, checks ten transitions, and runs the CartPole recovery grid or MsPacman blackout experiment.
`scientific_validation.py` tests known solutions, Q-value calibration, ending targets and frame-history interventions. Details are in the report.

```bash
for task in cartpole mspacman; do
  python validate.py --task "$task" --run "results/rerun/$task" --episodes 50 --device cuda
  python scientific_validation.py --run "results/rerun/$task" --episodes 50 --device cuda
done
python -m pytest -q
python build_report.py --results results/rerun --output hw2_report_rerun.pdf \
  --repo-url https://github.com/ZikunZheng1129/mlds490-hw2
```

To regenerate the supplied report from saved results, run `python build_report.py --repo-url https://github.com/ZikunZheng1129/mlds490-hw2`.

<details>
<summary>Optional: reproduce the baseline comparison and training-seed controls</summary>

### MsPacman comparison

Use this procedure **before starting MsPacman training**, in place of the main command above. It preserves the recorded development-only selection rule and uses the archived baseline.

```bash
mkdir -p results/rerun/upgrade_comparison
cp results/upgrade_comparison/experiment_plan.json \
   results/upgrade_comparison/baseline_development20.json \
   results/upgrade_comparison/baseline_development20.csv results/rerun/upgrade_comparison/
python -u train.py --task mspacman --profile extended --episodes 6000 --seed 7 --device cuda \
  --eval-every 100 --eval-episodes 20 --final-episodes 500 \
  --development-seed 40000 --evaluation-seed 200007 --output results/rerun/mspacman \
  > results/rerun/mspacman_training.log 2>&1 &
training_pid=$!
```

When the log contains `Development at episode 6000:`, run the following in the same terminal **before inspecting final scores and before `evaluation.csv` is written**. Do not wait for training to exit before freezing selection.

```bash
python compare_runs.py --baseline-run results/mspacman_baseline_2000 \
  --new-run results/rerun/mspacman --output results/rerun/upgrade_comparison --freeze-only
wait "$training_pid"
python compare_runs.py --baseline-run results/mspacman_baseline_2000 \
  --new-run results/rerun/mspacman --output results/rerun/upgrade_comparison --device cuda
```

### CartPole training-seed controls

After the main CartPole run, use fresh folders below. Seed 7 remains the primary model; seeds 8 and 9 are additional checks.

```bash
for seed in 8 9; do
  python train.py --task cartpole --episodes 3000 --seed "$seed" --device cuda \
    --eval-every 50 --eval-episodes 20 --final-episodes 500 \
    --development-seed 40000 --evaluation-seed 200007 \
    --output "results/rerun/cartpole_retraining/seed$seed"
done
python train.py --evaluate results/rerun/cartpole/policy.pt --episodes 500 --device cuda \
  --evaluation-seed 200007 --output results/rerun/cartpole_retraining/seed7_evaluation
python compare_runs.py --cartpole-controls results/rerun/cartpole_retraining
```

Run the report command again to include these additional results.

</details>
