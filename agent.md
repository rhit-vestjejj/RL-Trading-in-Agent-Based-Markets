You are building a research prototype in Python for an agent-based financial market experiment using a limit order book (LOB) simulator structure inspired by ABIDES.

Your job is to generate clean, modular, well-commented code for a FIRST WORKING VERSION of this experiment.

Do not overcomplicate it.
Do not add extra features unless required.
Do not redesign the experiment.
Implement exactly what is specified below.

==================================================
PROJECT GOAL
==================================================

We want to study how increasing the fraction phi of RL traders changes market statistics such as:
- tail exposure
- volatility clustering
- crash rate
- liquidity

This first version should focus on:
1. building the baseline market ecology,
2. defining the RL environment,
3. allowing RL traders to replace non-structural traders according to phi,
4. logging the data needed for later analysis.

This is a prototype, not the final paper version.

==================================================
HIGH-LEVEL REQUIREMENTS
==================================================

Implement the code as a clean Python project with modular files.

Use:
- Python
- object-oriented structure where reasonable
- numpy / pandas as needed
- clear dataclasses or config objects where helpful

Do NOT:
- add neural network training code unless needed for environment stub
- implement an overly complex simulator if not necessary
- invent new agent classes beyond those listed
- add unnecessary abstractions

If ABIDES integration is too heavy for one pass, structure the code so that the agent logic, environment state extraction, phi replacement logic, and logging are reusable and easy to connect to ABIDES later.

==================================================
MARKET ECOLOGY
==================================================

Baseline market composition:
- 45% ZIC traders
- 10% Trend followers
- 10% Noise traders
- 20% Value traders
- 15% Adaptive market makers

RL traders will replace ONLY:
- ZIC
- Trend
- Noise

Value traders and market makers stay fixed.

Maximum phi is therefore 0.65.

==================================================
PHI REPLACEMENT LOGIC
==================================================

Use proportional replacement of the replaceable pool.

Replaceable pool:
- ZIC = 0.45
- Trend = 0.10
- Noise = 0.10
Total replaceable = 0.65

For phi in [0, 0.65], define:

ZIC(phi)   = 0.45 * (1 - phi / 0.65)
Trend(phi) = 0.10 * (1 - phi / 0.65)
Noise(phi) = 0.10 * (1 - phi / 0.65)
Value(phi) = 0.20
MM(phi)    = 0.15
RL(phi)    = phi

Implement a function that, given:
- total number of agents N
- phi

returns integer counts for each trader class, preserving total N as closely as possible.

Use sensible rounding and ensure the counts sum exactly to N.

==================================================
LIMIT ORDER BOOK / MARKET VARIABLES
==================================================

Assume the market exposes or simulates at least:
- best bid b_t
- best ask a_t
- bid depth D_bid
- ask depth D_ask
- midprice m_t = (a_t + b_t) / 2
- spread S_t = a_t - b_t

Also define order book imbalance:

I_t = (D_bid - D_ask) / (D_bid + D_ask)

Make sure division by zero is handled safely.

==================================================
FUNDAMENTAL VALUE PROCESS
==================================================

Implement a latent fundamental value process:

v_{t+1} = v_t + epsilon_t
epsilon_t ~ Normal(0, sigma_v^2)

Each trader that uses value information observes:

v_hat_{i,t} = v_t + eta_{i,t}
eta_{i,t} ~ Normal(0, sigma_eta^2)

Default starting parameters:
- v0 = 100.0
- sigma_v = 0.03

==================================================
TRADER CLASSES TO IMPLEMENT
==================================================

Implement these trader classes with clear methods such as:
- observe(...)
- decide(...)
- submit_order(...)

Keep the logic clean and simple.

1. NoiseTrader
Signal/Input:
- random generator only
Decision rule:
- buy/sell randomly with probability 0.5
- size random in {1,2,3}
Order type:
- market order or near-touch limit order
Role:
- background random order flow

2. ZICTrader
Signal/Input:
- private value estimate v_hat_{i,t}
- surplus s_{i,t} ~ Uniform[s_min, s_max]
- optional best bid / ask for crossing check
Decision rule:
- buy limit price: p_buy = v_hat - s
- sell limit price: p_sell = v_hat + s
- if order would cross spread, allow marketable execution
Order type:
- mostly limit orders
Default parameters:
- s_min = 0
- s_max = 3 ticks
- sigma_eta = 1.0
- size in {1,2,3}

3. TrendFollowerTrader
Signal/Input:
- recent midprice history or return history
Decision rule:
- compute moving averages:
  MA_short over L_s = 5
  MA_long over L_l = 20
- trend signal T_t = MA_short - MA_long
- if T_t > 0 buy
- if T_t < 0 sell
- else hold
Order type:
- market or aggressive near-touch limit order
Default parameters:
- order size = 1

4. ValueTrader
Signal/Input:
- private noisy fundamental estimate v_hat_{i,t}
- current midprice m_t
Decision rule:
- mispricing M_t = v_hat - m_t
- if M_t > delta buy
- if M_t < -delta sell
- else hold
Order type:
- market or near-touch limit order
Default parameters:
- sigma_eta = 0.5
- delta = 1 tick
- size in {1,2}

5. AdaptiveMarketMaker
Signal/Input:
- midprice m_t
- spread S_t
- inventory q_t
Decision rule:
- quote both sides:
  p_bid = m_t - s/2 - alpha * q_t
  p_ask = m_t + s/2 - alpha * q_t
- refresh quotes frequently
Order type:
- two-sided limit orders
Default parameters:
- target spread = 2 ticks
- alpha = 0.005
- quote size = 3

6. RLTrader
This first version should support a simple RL-compatible agent interface.
Signal/Input:
- recent returns window
- spread
- imbalance
- inventory
Action space:
- buy
- sell
- hold
Operational meaning:
- buy = fixed-size market buy
- sell = fixed-size market sell
- hold = do nothing
Default order size:
- 1

Do not implement full PPO training unless asked.
Just implement the environment-facing logic and a placeholder / random / scripted policy interface.

==================================================
RL ENVIRONMENT SPECIFICATION
==================================================

Model this as an MDP approximation.

State:
s_t = [recent_returns, spread, imbalance, inventory]

Use a compact fixed-length numeric vector.
For example:
- last k returns
- current spread
- current imbalance
- current inventory

Let k be configurable (default k = 10).

Action space:
- 0 = sell
- 1 = hold
- 2 = buy

Reward:
R_t = (W_{t+1} - W_t) - lambda_q * q_{t+1}^2

where:
W_t = c_t + q_t * m_t

Variables:
- c_t = agent cash
- q_t = inventory
- m_t = midprice

Use:
- lambda_q configurable
- default lambda_q = 0.01

Implement this reward carefully and clearly.

==================================================
PARAMETERS / DEFAULTS
==================================================

Use these first-pass defaults:
- tick size = 0.01
- initial price / fundamental = 100.0
- sigma_v = 0.03
- lambda_q = 0.01
- RL order size = 1
- trend MA windows = 5 and 20
- ZIC surplus range = [0, 3 ticks]
- Value threshold delta = 1 tick
- MM spread = 2 ticks
- MM alpha = 0.005
- MM quote size = 3

==================================================
LOGGING REQUIREMENTS
==================================================

Implement logging for later analysis.

At each simulation step or decision interval, log at least:
- time
- best bid
- best ask
- midprice
- spread
- bid depth
- ask depth
- imbalance
- traded volume
- signed order flow if available
- fundamental value v_t
- per-RL-agent inventory
- per-RL-agent cash
- per-RL-agent wealth
- per-RL-agent reward

Save logs in a pandas DataFrame and support CSV export.

==================================================
ANALYSIS UTILITIES
==================================================

Implement utility functions to compute:
1. log returns
2. excess kurtosis
3. volatility clustering proxy:
   autocorrelation of squared returns up to lag L
4. drawdown series
5. max drawdown
6. average spread
7. average top-of-book depth

Keep these as analysis helpers; do not overbuild.

==================================================
PROJECT STRUCTURE
==================================================

Create a simple structure like:

- config.py
- market.py
- lob.py
- agents/
    - base.py
    - noise.py
    - zic.py
    - trend.py
    - value.py
    - market_maker.py
    - rl_trader.py
- env.py
- logging_utils.py
- analysis.py
- run_simulation.py

If a lighter structure is better for a first pass, that is acceptable, but keep it organized.

==================================================
CODE QUALITY REQUIREMENTS
==================================================

- Write complete runnable code, not pseudocode.
- Use clear docstrings.
- Use type hints where reasonable.
- Handle edge cases cleanly.
- Keep the code readable.
- Do not leave TODO stubs unless absolutely necessary.
- If something must be mocked because the full ABIDES integration is unavailable, make that explicit and isolate it.

==================================================
DELIVERABLE
==================================================

Generate the full code for this first-pass prototype.

At the top, briefly explain the structure.
Then provide the code file by file.

Again:
keep it simple,
keep it coherent,
and implement exactly this design.
