# RL Trading in Agent-Based Markets

This repository contains the ABIDES-backed market simulator, RL agents, PPO training loop, and diagnostics used for the research project *Persistent One-Sided Order Books from Learned Trading Behavior*.

The codebase centers on a frozen baseline market ecology and studies what happens as a fraction `phi` of replaceable background traders is swapped out for RL-controlled agents. In practice, the repo supports three main workflows:

1. Run baseline or mixed-agent simulations on top of ABIDES.
2. Train and evaluate a shared PPO policy for the RL agents.
3. Generate diagnostics, ablations, and experiment summaries for market quality and order-book behavior.

## Repository Layout

```text
agents/                        Simple trader interfaces and scripted agents
tests/                         Unit and integration tests
external/abides-jpmc-public/   Vendored ABIDES source tree

market.py                      ABIDES-backed simulator wrapper
config.py                      Market profiles and top-level experiment config
env.py                         RL state/reward helpers and placeholder policies
ppo_training.py                Shared linear PPO implementation
train_shared_ppo.py            CLI for PPO training
evaluate_trained_policy.py     CLI for policy evaluation
run_simulation.py              One-off simulation runner
run_phi_experiment.py          Full phi sweep experiment driver
run_realism_diagnostics.py     Baseline realism report
run_sampling_comparison.py     Multi-frequency logging comparison
run_price_discovery_ablation.py Narrow ablation runner
```

## Setup

The project vendors ABIDES under [external/abides-jpmc-public](/Users/jean-paulvestjens/RL-Trading-in-Agent-Based-Markets/external/abides-jpmc-public). Import paths for `abides-core` and `abides-markets` are bootstrapped automatically by [abides_support.py](/Users/jean-paulvestjens/RL-Trading-in-Agent-Based-Markets/abides_support.py), so you do not need to manually edit `PYTHONPATH`.

Create a Python environment and install the local scientific stack first:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install numpy pandas matplotlib pytest
```

If your environment is missing ABIDES runtime dependencies, use the vendored requirement files as a reference:

```bash
python3 -m pip install -r external/abides-jpmc-public/requirements.txt
python3 -m pip install -r external/abides-jpmc-public/requirements-dev.txt
```

Matplotlib may try to write cache files outside the repo on first import. If that is a problem in your environment, set:

```bash
export MPLCONFIGDIR="$PWD/.matplotlib"
```

## Common Workflows

### Run a baseline simulation

```bash
python3 run_simulation.py \
  --phi 0.0 \
  --seed 7 \
  --end-time 09:35:00 \
  --output simulation_log.csv
```

This writes a market log CSV and prints summary statistics such as spread, depth, kurtosis, tail exposure, crash rate, and max drawdown.

### Visualize a saved market path

```bash
python3 visualize_market.py \
  --input simulation_log.csv \
  --output market_replay.png
```

### Train a shared PPO policy

```bash
python3 train_shared_ppo.py \
  --phi 0.10 \
  --episodes 20 \
  --end-time 09:35:00 \
  --checkpoint-output shared_ppo_policy.npz \
  --training-log-output ppo_training_log.csv \
  --combined-log-output ppo_progress_log.csv
```

The current PPO implementation in [ppo_training.py](/Users/jean-paulvestjens/RL-Trading-in-Agent-Based-Markets/ppo_training.py) is a compact shared linear policy/value model implemented directly with NumPy.

### Evaluate a trained or scripted policy

```bash
python3 evaluate_trained_policy.py \
  --policy trained \
  --checkpoint shared_ppo_policy.npz \
  --phi 0.10 \
  --evaluation-mode greedy \
  --seeds 7,8,9 \
  --output policy_evaluation.csv \
  --summary-output trained_eval_summary.csv
```

Scripted baselines are also available through `--policy random`, `inventory_aware`, `random_quoter`, and `inventory_aware_quoter`.

### Run the full phi sweep

```bash
python3 run_phi_experiment.py \
  --phi-grid 0.00,0.05,0.10 \
  --episodes 50 \
  --evaluation-interval 5 \
  --checkpoint-interval 5 \
  --output-dir experiments/phi_sweep_test
```

This produces an experiment folder with config, summary tables, plots, and a markdown report.

### Generate diagnostics

Baseline realism:

```bash
python3 run_realism_diagnostics.py \
  --phi 0.0 \
  --seed 7 \
  --output realism_run.csv \
  --diagnostics-output realism_diagnostics.csv \
  --report-output realism_report.md
```

Sampling-frequency sensitivity:

```bash
python3 run_sampling_comparison.py \
  --phi 0.0 \
  --frequencies 1s,5s,15s \
  --output-dir sampling_compare
```

Price-discovery ablation:

```bash
python3 run_price_discovery_ablation.py \
  --num-runs 3 \
  --ablate value_order_size=1,2 \
  --ablate noise_limit_probability=0.25,0.45
```

## Testing

Run the full test suite with:

```bash
python3 -m pytest -q
```

The tests cover core agent behavior, PPO training utilities, diagnostics, plotting helpers, and RL/ABIDES integration.

## Notes on Outputs

This repository has historically written many generated CSVs, PNGs, checkpoint files, and experiment folders directly into the project root. The new [.gitignore](/Users/jean-paulvestjens/RL-Trading-in-Agent-Based-Markets/.gitignore) excludes those artifacts so the code and docs stay reviewable.

For new runs, prefer passing explicit output paths instead of relying on root-level defaults. Good patterns are:

```text
experiments/<experiment-name>/...
reports/...
artifacts/...
```

## Key Concepts

- `phi`: the fraction of replaceable baseline traders that are replaced by RL agents.
- `market_profile`: the background trader ecology defined in [config.py](/Users/jean-paulvestjens/RL-Trading-in-Agent-Based-Markets/config.py).
- `taker_only`, `mixed`, `quoter_only`: the three RL liquidity modes supported by the simulator and training scripts.
- `abides_rmsc04_small_v1`: the default frozen baseline profile, currently 80 noise traders, 20 value traders, and 2 market makers.
