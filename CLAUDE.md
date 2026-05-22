# Project handoff — RL Trading in Agent-Based Markets

This is an agent-based market simulation (ABIDES-based) studying how reinforcement learning agent participation affects limit order book liquidity, written up as `final.tex` for an ICAIF-style submission. The paper claims persistent one-sided order book states emerge nonlinearly with the RL fraction φ, with a flow-rate threshold theory predicting φ* ≈ 0.30.

A prior session audited the paper against the code and the reviewer critique. **Several paper claims are not actually backed by the code, and the headline experiment data is missing from this checkout.** Do not start architectural changes (richer policy, MLP, multi-config sweeps) until the pre-flight items below are resolved — otherwise we'll be building on top of unsupported claims.

## Ground rules for this session

- **Do not push to remote, do not amend commits, do not force-push.** Ask before any destructive git operation.
- **Do not delete experiments/ directories or `.npz` policy files** without confirming with the user — they are gitignored and irreplaceable.
- Tests live in `tests/`. Run `python -m pytest tests/ -x -q` before and after changes — all 83 should pass.
- The `external/abides-jpmc-public/` submodule has a **local uncommitted patch** to `abides-markets/abides_markets/order_book.py` (adds `best_bid_agent_id` / `best_ask_agent_id` to `book_log2`). Preserve this patch; do not reset the submodule.
- The simulator policy in `ppo_training.py:121` (`SharedLinearPPOPolicy`) is a single linear softmax layer with no hidden layers. The paper does not disclose this. Treat replacing it with an MLP as a major change requiring user approval.

## Repo orientation

- `final.tex` — the paper (compiles to `final.pdf` via `pdflatex final.tex` × 2; needs `acmart` class, install via `tlmgr --usermode install acmart` if missing).
- `abides_agents.py` — agent classes. `BaseRLTrader` at line 1078; `RLTrader` (3 actions, market orders only) at 1507; `RLQuotingTrader` (4 actions, fixed-size quotes) at 1556.
- `env.py` — RL state (13-dim: 10 returns + spread + imbalance + inventory) and reward function (line 146).
- `ppo_training.py` — `SharedLinearPPOPolicy` linear softmax at line 121; shared-across-all-agents policy factory at line 782.
- `phi_experiment.py` — the main φ-sweep orchestration.
- `analysis.py:81` — `one_sided_book_metrics` defines the headline metric (volume-XOR, not price-XOR).
- `threshold_analysis.py` — analytical threshold + figure; hardcoded paths are stale (see below).
- `regenerate_paper_figures.py` — hardcoded paths to non-existent experiment dirs (see below).
- `experiments/` (gitignored, large) — only `phi_sweep_45min`, `mixed_full_reward_shaped`, `phi_sweep_test`, `phi_sweep_validation_onesided_20260321`, `paper_evidence_exports`, `comparison_plots` exist locally.

## Working-tree state at handoff

Uncommitted:
- `README.md` — adds docs for `run_reviewer_robustness.py` and `--evaluation-seed-count` flag.
- `market.py` — adds a per-PID/timestamp `log_dir` to the ABIDES Kernel call. Unclear why; verify before building on it.
- `phi_experiment.py` — adds signed-action-imbalance / quote-activity aggregations and more error-bar plots.
- `run_phi_experiment.py` — adds `--evaluation-seed-count` / `--evaluation-seed-start` convenience flags.
- `external/abides-jpmc-public` — the submodule patch above.

Untracked (mid-flight work responding to reviewer concerns):
- `reviewer_robustness.py` (724 lines) and `run_reviewer_robustness.py` (91 lines) — "trained vs random vs inventory-aware at matched composition" controls.

Resolve these before adding new code: decide what to commit and what to discard. Branch from a clean state.

## Pre-flight blockers — do these first, in order

### 1. Decide the truth about the anti-degeneracy intervention
Paper §2.9 and §3.5 describe an anti-degeneracy hold-streak penalty. **No simulator code implements it.** `hold_streak_penalty` / `hold_streak_grace` are config fields written into saved experiment JSON but read only by `behavior_mode_comparison.py` (a downstream plotter). The reward function in `env.py:191` has no anti-degeneracy term.

Either:
- (a) Implement it in `BaseRLTrader.on_observation` + `RLMarketEnvironment.compute_reward_components` (track per-agent consecutive hold streak; add a penalty term after `hold_streak_grace` decisions), and rerun §3.5; or
- (b) Cut §2.9 and §3.5 from the paper.

Ask the user which path before coding.

### 2. Reconcile the paper reward formula with `env.py`
Paper line 126: `r_t = ΔW_t − λ_q·100·q_t² − P_hold − P_anti-degeneracy`.
Code `env.py:191`: `ΔW − inventory_penalty − flat_hold_penalty + passive_fill_bonus + two_sided_quote_reward − missing_quote_penalty`.

The paper omits three terms that exist in code and includes one that doesn't. Also: paper describes `P_hold` as "an optional penalty for remaining inactive" but `flat_hold_penalty` only fires when `previous_action == 1 AND inventory == 0`.

Pick a canonical reward, update the other side. This is a text fix on the paper side, primarily.

### 3. Reconcile the one-sided definition
Paper §2.10 says "best bid or best ask missing." Code (`analysis.py:142`) computes `one_sided_book_fraction` as `bid_volume > 0 XOR ask_volume > 0` conditional on positive visible liquidity. The price-missing version is computed separately as `true_one_sided_book_fraction`.

The paper's figures plot the volume-XOR metric. Update §2.10 text to match what the code measures, or switch the figures to the `true_one_sided_book_fraction` series.

### 4. Restore (or replace) the missing experiment data
The paper's headline figures point at experiment dirs that **do not exist locally**:
- `experiments/paper_trained_nocap/` — referenced in `threshold_analysis.py:23` and `regenerate_paper_figures.py:115`.
- `experiments/more_seeds/merged_market_metrics.csv` — `regenerate_paper_figures.py:26`.
- `experiments/alt_profile_200ep_6seed/combined/phi_sweep_summary.csv` — `regenerate_paper_figures.py:114`.

The summary CSVs that DO exist (`phi_sweep_45min`, `mixed_full_reward_shaped`) have 136 columns, **none of which contain `one_sided` or `undefined_midprice`** — the headline metric was never aggregated into the saved summaries on disk. So even the existing data won't directly recompute the paper's figures.

Action: ask the user where these experiment dirs went (different machine? deleted?). If they're gone:
- Rerun the φ-sweep that produced `paper_trained_nocap` (it was 50 episodes × 3 seeds based on `run_phi_experiment.py` defaults, then merged with 3 more seeds via `run_more_seeds.py` for the 6-seed figure).
- Rerun the alt-profile experiment via `run_alt_profile_experiment.py` for the robustness check.
- Update `phi_experiment.py` aggregator to write `one_sided_book_fraction_mean` and `undefined_midprice_fraction_mean` into the summary CSV before rerunning.

### 5. Verify `threshold_analysis.py` matches the data
`threshold_analysis.py:33` hardcodes `SIM_SEC = 300` (5 min). The `phi_sweep_45min` experiment ran ~45 minutes. The λ_r estimate divides `executed_buy_action_count + executed_sell_action_count` by `n_rl * SIM_SEC` — if applied to a 45-min experiment with `SIM_SEC=300`, λ_r is off by 9×. Verify which simulation horizon produced the data feeding the threshold figure, and fix `SIM_SEC` (or compute it from the data).

### 6. Document or merge the ABIDES submodule patch
`external/abides-jpmc-public/abides-markets/abides_markets/order_book.py` has a local diff adding `best_bid_agent_id` / `best_ask_agent_id` to `book_log2`. Without it, any analysis that reads those fields silently differs.

Either: fork ABIDES and pin the submodule to a fork commit that includes this patch, or add a `patches/` dir with the diff and apply it on setup. Document in README.

### 7. Resolve the working-tree state
Get to a clean known-good commit before building anything new. Walk through the 5 modified files + 2 untracked modules with the user; commit what should be kept, discard the rest.

### 8. Add a regression test for the threshold result
Pin the empirical `one_sided_book_fraction` at φ ∈ {0.10, 0.30, 0.40} from a fixed seed → so any future architectural change has a tripwire.

## After pre-flight: prioritized architectural changes

These are the things that actually answer the reviewer critique (paper has the contribution but it's underdeveloped). Tackle only after the pre-flight list is clean.

### Group A — Make "learned behavior" mean something
1. **Replace `SharedLinearPPOPolicy` with a small MLP** (e.g., 2× 64-unit hidden + tanh). Lives at `ppo_training.py:121`. Add as `MLPPPOPolicy` alongside the existing class, same `sample_action` / `save` / `load` interface, swap via config flag. Prerequisite for any non-trivial "learned" claim.
2. **Independent policy weights per agent (or per-agent latent ID in state).** Currently `ppo_training.py:782` broadcasts one policy object to all RL agents — any coordinated-withdrawal claim built on this is structurally tautological. Either independent heads + independent training, or shared weights + agent-ID one-hot in the state vector.
3. **Per-step mechanism logging.** `BaseRLTrader.on_observation` already logs `time / agent_id / action / state` to `metrics_log`. Add a post-run analysis pass that computes, per (φ, seed): cross-agent action correlation per timestep, autocorrelation of own market-order arrivals, P(sell | low ask depth, negative recent return), sign asymmetry conditional on inventory. Output → `summaries/mechanism_diagnostics.csv`. This unlocks the §3.8 "trained vs random — why?" answer.

### Group B — Defensible action/state space
4. **Expand the action space.** Minimum: action type {market, limit-at-best, limit-improve-by-k, cancel, hold} × side {buy, sell} × size {1, S_mid, S_large}. Lives at `abides_agents.py:1548` (`_submit_effective_action`). Keep discrete; ~30 actions is fine for PPO.
5. **Real learnable market-maker head** for the quoter (replace fixed-size at-best/one-tick-inside). Action = (skew, half-spread, size) — discrete bins or small continuous head. `RLQuotingTrader` at `abides_agents.py:1556`.
6. **Expand state.** Add top-5 bid/ask sizes, recent trade-sign imbalance, time-since-own-last-action, recent own PnL, fraction-of-book-empty indicator. Lives at `env.py:107` (`build_state`).

### Group C — Robustness
7. **Make `num_market_makers` a first-class sweep parameter** and run a 2D (φ × MM count) grid. Predict φ\*(λ_m) from the analytical model and overlay. Turns the current "alt profile killed the effect" weakness into a quantitative validation of the threshold theory.
8. **Alternative fundamentals processes.** `abides_oracle.py` is 100 lines and currently Gaussian RW. Add a jump-diffusion oracle and a stochastic-vol oracle as drop-in replacements.
9. **Heterogeneous wake-up latency.** Sample per-RL-agent `wake_up_freq` from a distribution instead of using the fixed `"1s"` default at `config.py:241`.

## Paper-side polish (parallel track, no code dependency)

- Strip the `Conference'17` template boilerplate and fix the `\usepackage[hidelinks]{hyperref}` option clash (acmart already loads hyperref; move `hidelinks` into the documentclass options).
- Convert the URL-list bibliography to a `.bib` with real `\cite{}` calls.
- Cut ~40% redundancy: abstract / intro / results-summary / discussion / conclusion all restate the same four findings.
- Add a formal statistical test for the §3.8 "sharper than random" claim (bootstrap CI on the difference, or Wilcoxon across seeds).
- Disclose the linear policy explicitly if Group A item 1 isn't done before submission.

## Order of operations

1. Pre-flight items 1–7, with user check-in at each (especially #1, #2, #3, #4 — these change the paper).
2. Pre-flight item 8 (regression test).
3. Group A items 1, 3, 2 in that order. Each is ~1–2 days; items 1 and 3 unblock everything else.
4. Group B item 4. Then group C item 7. (These two together are where the new contribution actually lives.)
5. The rest (B5/B6, C8/C9) only if there's time before submission.

## Things to ask before starting

- Where did the missing experiment dirs go? (`paper_trained_nocap`, `more_seeds`, `alt_profile_200ep_6seed`)
- For pre-flight #1: implement anti-degeneracy or cut from the paper?
- For pre-flight #2: paper formula authoritative, or code authoritative?
- For pre-flight #3: switch figures to `true_one_sided_book_fraction`, or rewrite §2.10?
- ICAIF submission deadline?
