"""
Aether Crystal — ML Trading System
====================================
Three optimizers, all pure NumPy:

  1. AnalyticalOptimizer  — closed-form gradient; gold standard / sanity check
  2. GradientOptimizer    — mini-batch MC gradient descent with Adam
                            uses reparameterization trick: q_i = MAX_Q * tanh(w_i)
  3. MLPPolicy            — two-layer MLP trained across randomized market
                            scenarios; generalizes to unseen S0 / vol conditions

Key insight (reparameterization trick)
---------------------------------------
  E[PnL] = Σ_i  q_i · (E[payoff_i(S_T)] − entry_price_i)

  ∂E[PnL]/∂q_i   = E[payoff_i(S_T)] − entry_price_i        (edge per unit)
  q_i             = MAX_Q · tanh(w_i)                        (unconstrained param)
  ∂E[PnL]/∂w_i   = edge_i · MAX_Q · (1 − tanh²(w_i))       (chain rule)

All gradients are computed analytically via MC — no finite differences needed.
"""

import numpy as np
import matplotlib.pyplot as plt
import time

# ── Constants ─────────────────────────────────────────────────────────────────

SIGMA         = 2.51            # 251% annualized vol
TRADING_DAYS  = 252
STEPS_PER_DAY = 4
DT            = 1.0 / (TRADING_DAYS * STEPS_PER_DAY)   # 1/1008 yr
MAX_Q         = 50            # max contracts per option

OPTION_CHAIN = [
    {"name" : "AC", "K" : 50, "type" : "security", "days" : 0, "bid" : 49.975, "ask" : 50.025, "max" : 200},
    {"name": "AC_50_P",   "K": 50, "type": "put",  "days": 21, "bid": 12.00, "ask": 12.05, "max" : 50},
    {"name": "AC_50_C",   "K": 50, "type": "call", "days": 21, "bid": 12.00, "ask": 12.05, "max" : 50},
    {"name": "AC_35_P",   "K": 35, "type": "put",  "days": 21, "bid":  4.33, "ask":  4.35, "max" : 50},
    {"name": "AC_40_P",   "K": 40, "type": "put",  "days": 21, "bid":  6.50, "ask":  6.55, "max" : 50},
    {"name": "AC_45_P",   "K": 45, "type": "put",  "days": 21, "bid":  9.05, "ask":  9.10, "max" : 50},
    {"name": "AC_60_C",   "K": 60, "type": "call", "days": 21, "bid":  8.80, "ask":  8.85, "max" : 50},
    {"name": "AC_50_P_2", "K": 50, "type": "put",  "days": 14, "bid":  9.70, "ask":  9.75, "max" : 50},
    {"name" : "AC_50_C_2", "K" : 50, "type" : "call", "days" : 14, "bid" : 9.7, "ask" : 9.75, "max" : 50},
#    {"name" : "AC_50_CO", "K" : 50, "type" : "choose", "days" : 21, "bid" : 22.2, "ask" : 22.3},
    {"name" : "AC_40_BP", "K" : 40, "type" : "binary", "days" : 21, "bid" : 5, "ask" : 5.1, "max" : 50},
    {"name" : "AC_45_KO", "K" : 50, "type" : "knockout", "days" : 21, "bid" : 0.15, "ask" : 0.175, "max" : 500},

]
N_OPT = len(OPTION_CHAIN)


# ── GBM core ──────────────────────────────────────────────────────────────────

def simulate_terminals(S0, days, n_paths, rng, sigma=SIGMA):
    """
    Exact GBM terminal prices via log-sum of independent increments.
    Returns: (n_paths,) array of S_T.
    """
    n_steps = days * STEPS_PER_DAY
    Z = rng.standard_normal((n_steps, n_paths))
    log_ret = (-0.5 * sigma**2 * DT + sigma * np.sqrt(DT)) * Z
    return S0 * np.exp(log_ret.sum(axis=0))

knockouts = {}

def build_st_cache(S0, n_paths, rng, sigma=SIGMA, chain=OPTION_CHAIN):
    #global knockouts
    days_needed = list({o["days"] for o in chain})
    cache = {d: simulate_terminals(S0, d, n_paths, rng, sigma) for d in days_needed}
    #knockouts = {d : (np.sum(np.min(cache[d]) < 35) >= 1) for d in cache.keys()}
    return cache

def option_payoff(ST, K, opt_type):
    global knockout
    if opt_type == "call":
        return np.maximum(ST - K, 0.0)
    elif opt_type == "put":
        return np.maximum(K - ST, 0.0)
    elif opt_type == "binary":
        return np.where(ST < 40, 10, 0)
    elif opt_type == "knockout":
        return np.zeros_like(ST, dtype=float) if np.any(ST < 35) else np.maximum(K - ST, 0.0)
    elif opt_type == "security":
        return np.maximum(-ST, 0.0)

def portfolio_pnl(quantities, ST_cache, chain=OPTION_CHAIN):
    global knockout
    """
    quantities : (N_OPT,) array — positive = long, negative = short
    ST_cache   : {days: (n_paths,) terminal prices}
    returns    : (n_paths,) realized PnL
    """
    n_paths = next(iter(ST_cache.values())).shape[0]
    pnl = np.zeros(n_paths)
    for i, opt in enumerate(chain):
        q  = quantities[i]
        p  = option_payoff(ST_cache[opt["days"]], opt["K"], opt["type"])
        ep = opt["ask"] if q >= 0 else opt["bid"]
        pnl += q * (p - ep)
    return pnl

def edge_vector(quantities, ST_cache, chain=OPTION_CHAIN):
    """∂E[PnL]/∂q_i  =  E[payoff_i − entry_price_i]  for each option."""
    edges = np.zeros(N_OPT)
    for i, opt in enumerate(chain):
        p  = option_payoff(ST_cache[opt["days"]], opt["K"], opt["type"])
        ep = opt["ask"] if quantities[i] >= 0 else opt["bid"]
        edges[i] = (p - ep).mean()
    return edges


# ── Adam optimizer ────────────────────────────────────────────────────────────

class Adam:
    """Vanilla Adam; params is a list of np.ndarray references."""
    def __init__(self, params, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8):
        self.params = params
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        lr_t = self.lr * np.sqrt(1 - self.b2**self.t) / (1 - self.b1**self.t)
        for i, (p, g) in enumerate(zip(self.params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g**2
            p += lr_t * self.m[i] / (np.sqrt(self.v[i]) + self.eps)


# ── 1. Analytical Optimizer ───────────────────────────────────────────────────

class AnalyticalOptimizer:
    """
    Because E[PnL] is linear in q_i (given the sign of q_i is fixed),
    the unconstrained optimum is bang-bang:

        q_i* = +MAX_Q   if  E[payoff_i] > ask_i   (buy — option is cheap)
        q_i* = −MAX_Q   if  E[payoff_i] < bid_i   (sell — option is rich)
        q_i* =  0       if  bid_i ≤ E[payoff_i] ≤ ask_i  (no edge)

    This is the provably optimal solution for risk-neutral PnL maximization
    with no position-size penalty and independent options.
    """
    def __init__(self, S0=50.0, n_paths=300_000, seed=0):
        self.S0, self.n_paths = S0, n_paths
        self.rng = np.random.default_rng(seed)

    def optimize(self, verbose=True):
        ST = build_st_cache(self.S0, self.n_paths, self.rng)
        quantities = np.zeros(N_OPT)
        rows = []

        for i, opt in enumerate(OPTION_CHAIN):
            ev = option_payoff(ST[opt["days"]], opt["K"], opt["type"]).mean()
            if ev > opt["ask"]:
                q, action, ep = +opt["max"], "BUY",  opt["ask"]
            elif ev < opt["bid"]:
                q, action, ep = -opt["max"], "SELL", opt["bid"]
            else:
                q, action, ep = 0,  "PASS", (opt["bid"] + opt["ask"]) / 2
            quantities[i] = q
            edge = abs(ev - ep)
            rows.append((opt["name"], action, q, ev, opt["bid"], opt["ask"], edge, q * (ev - ep)))

        pnl = portfolio_pnl(quantities, ST)

        if verbose:
            _header("ANALYTICAL OPTIMIZER")
            print(f"  {'Option':<14} {'Act':>5} {'Qty':>7}  {'EV payoff':>10}  "
                  f"{'Bid':>7}  {'Ask':>7}  {'Edge/c':>8}  {'E[PnL]':>10}")
            print("  " + "─" * 72)
            for r in rows:
                print(f"  {r[0]:<14} {r[1]:>5} {r[2]:>7.0f}  {r[3]:>10.4f}  "
                      f"{r[4]:>7.3f}  {r[5]:>7.3f}  {r[6]:>8.4f}  {r[7]:>10.1f}")
            _stats(pnl, prefix="  ")
        return quantities, pnl


# ── 2. Gradient Optimizer ─────────────────────────────────────────────────────

class GradientOptimizer:
    """
    Gradient ascent on E[PnL] using mini-batch MC.
    Uses reparameterization: q_i = MAX_Q · tanh(w_i), so w_i is unconstrained.

    The gradient of E[PnL] w.r.t. w_i is:
        g_i = E[payoff_i − price_i] · MAX_Q · sech²(w_i)

    Advantage over the analytical optimizer: discovers the optimal sign of
    each position through learning rather than a hard threshold, and can
    detect regime changes mid-training if the market shifts.
    """
    def __init__(self, S0=50.0, sigma=SIGMA, lr=0.1, batch=30_000,
                 n_epochs=200, seed=1):
        self.S0, self.sigma = S0, sigma
        self.batch, self.n_epochs = batch, n_epochs
        self.rng = np.random.default_rng(seed)
        self.w = np.zeros(N_OPT)           # start: all positions = 0
        self.opt = Adam([self.w], lr=lr)
        self.hist_epnl = []
        self.hist_q    = []

    @property
    def quantities(self):
        return 50 * np.tanh(self.w)

    def train(self, verbose=True):
        if verbose:
            _header("GRADIENT OPTIMIZER — training")
            print(f"  {'Epoch':>6}  {'E[PnL]':>10}  Quantities")
            print("  " + "─" * 75)

        for epoch in range(1, self.n_epochs + 1):
            ST = build_st_cache(self.S0, self.batch, self.rng, self.sigma)
            q  = self.quantities
            edges = edge_vector(q, ST)                     # ∂E[PnL]/∂q_i
            epnl  = float(np.dot(q, edges))
            sech2 = 1.0 - np.tanh(self.w)**2              # ∂tanh/∂w_i
            grad_w = edges * MAX_Q * sech2                 # chain rule
            self.opt.step([grad_w])
            self.hist_epnl.append(epnl)
            self.hist_q.append(q.copy())

            if verbose and (epoch == 1 or epoch % 40 == 0):
                q_str = "  ".join(f"{qi:+7.0f}" for qi in q)
                print(f"  {epoch:>6}  {epnl:>10.2f}  {q_str}")

        return self

    def evaluate(self, n_eval=200_000):
        rng = np.random.default_rng(99)
        ST  = build_st_cache(self.S0, n_eval, rng, self.sigma)
        q   = self.quantities
        pnl = portfolio_pnl(q, ST)
        print(f"\n  Gradient Optimizer — final quantities:")
        for i, opt in enumerate(OPTION_CHAIN):
            print(f"    {opt['name']:<14} {'BUY ' if q[i]>=0 else 'SELL'}  {abs(q[i]):>8.1f}")
        _stats(pnl, prefix="  ")
        return q, pnl


# ── 3. MLP Policy ─────────────────────────────────────────────────────────────

def _relu(x):      return np.maximum(0.0, x)
def _relu_d(x):    return (x > 0).astype(float)

class MLPPolicy:
    """
    Two-layer MLP: market_features → position_sizes.

    Input (N_OPT × 5 = 35 features):
        For each option: [S0/K,  T×10,  sigma/3,  bid/mid,  ask/mid]

    Output:  MAX_Q · tanh(z)  — position sizes in [−MAX_Q, MAX_Q]

    Training strategy:
        Each epoch samples n_scenarios random (S0, sigma) pairs and averages
        gradients, so the policy learns a generalizable market response rather
        than overfitting to a single price level.

    Gradient flow (reparameterization):
        ∂E[PnL]/∂θ  =  ∂E[PnL]/∂q  ·  ∂q/∂θ
                     =  edge_vector  ·  backprop(MLP)
    """
    def __init__(self, n_hidden=64, lr=3e-3, seed=42):
        rng = np.random.default_rng(seed)
        n_in  = N_OPT * 5
        # Xavier init
        self.W1 = rng.standard_normal((n_in, n_hidden)) * np.sqrt(2.0 / n_in)
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.standard_normal((n_hidden, N_OPT)) * np.sqrt(2.0 / n_hidden)
        self.b2 = np.zeros(N_OPT)
        self.opt = Adam([self.W1, self.b1, self.W2, self.b2], lr=lr)
        self.rng_train = np.random.default_rng(seed + 7)
        self.hist_epnl = []

    @staticmethod
    def _features(S0, sigma, chain=OPTION_CHAIN):
        feats = []
        for opt in chain:
            mid = (opt["bid"] + opt["ask"]) / 2
            feats += [
                S0 / opt["K"],            # moneyness
                opt["days"] / TRADING_DAYS * 10,  # scaled time-to-expiry
                sigma / 3.0,              # scaled vol
                opt["bid"] / mid,         # relative bid
                opt["ask"] / mid,         # relative ask
            ]
        return np.array(feats, dtype=float)

    def _forward(self, x):
        z1  = x @ self.W1 + self.b1          # (n_hidden,)
        a1  = _relu(z1)
        z2  = a1 @ self.W2 + self.b2         # (N_OPT,)
        out = MAX_Q * np.tanh(z2)            # (N_OPT,)
        return out, (x, z1, a1, z2)

    def _backward(self, grad_out, cache):
        x, z1, a1, z2 = cache
        d_z2 = grad_out * MAX_Q * (1.0 - np.tanh(z2)**2)   # output sech²
        dW2  = np.outer(a1, d_z2)
        db2  = d_z2
        d_a1 = self.W2 @ d_z2
        d_z1 = d_a1 * _relu_d(z1)
        dW1  = np.outer(x, d_z1)
        db1  = d_z1
        return [dW1, db1, dW2, db2]

    def predict(self, S0, sigma):
        feats = self._features(S0, sigma)
        out, _ = self._forward(feats)
        return np.round(out).astype(int)

    def train(self, n_epochs=300, batch=15_000, n_scenarios=4, verbose=True):
        rng_s = np.random.default_rng(77)
        if verbose:
            _header("MLP POLICY — training (generalizes across S0 and σ)")
            print(f"  {'Epoch':>6}  {'E[PnL]':>12}  (avg over {n_scenarios} scenarios/epoch)")
            print("  " + "─" * 40)

        for epoch in range(1, n_epochs + 1):
            S0s    = rng_s.uniform(35, 65, n_scenarios)
            sigmas = rng_s.uniform(1.5, 3.5, n_scenarios)
            agg_grads = [np.zeros_like(p) for p in [self.W1, self.b1, self.W2, self.b2]]
            epoch_epnl = 0.0

            for S0, sigma in zip(S0s, sigmas):
                feats = self._features(S0, sigma)
                q, cache = self._forward(feats)
                ST = build_st_cache(S0, batch, self.rng_train, sigma)
                edges = edge_vector(q, ST)              # ∂E[PnL]/∂q_i
                epoch_epnl += float(np.dot(q, edges)) / n_scenarios
                grads = self._backward(edges, cache)    # backprop
                for ag, g in zip(agg_grads, grads):
                    ag += g / n_scenarios

            self.opt.step(agg_grads)
            self.hist_epnl.append(epoch_epnl)

            if verbose and (epoch == 1 or epoch % 60 == 0):
                print(f"  {epoch:>6}  {epoch_epnl:>12.2f}")

        return self

    def recommend(self, S0=50.0, sigma=SIGMA, n_eval=200_000, verbose=True):
        q    = self.predict(S0, sigma)
        rng  = np.random.default_rng(0)
        ST   = build_st_cache(S0, n_eval, rng, sigma)
        pnl  = portfolio_pnl(q.astype(float), ST)

        if verbose:
            print(f"\n  MLP Policy — recommendations  (S0={S0}, σ={sigma*100:.0f}%)")
            print(f"  {'Option':<14} {'Action':>6}  {'Qty':>7}  {'Bid':>8}  {'Ask':>8}")
            print("  " + "─" * 50)
            for i, opt in enumerate(OPTION_CHAIN):
                action = "BUY " if q[i] >= 0 else "SELL"
                print(f"  {opt['name']:<14} {action}  {abs(q[i]):>7}  "
                      f"{opt['bid']:>8.3f}  {opt['ask']:>8.3f}")
            _stats(pnl, prefix="  ")
        return q, pnl


# ── Helpers ───────────────────────────────────────────────────────────────────

def _header(title):
    print("\n" + "━" * 65)
    print(f" {title}")
    print("━" * 65)

def _stats(pnl, prefix=""):
    print(f"\n{prefix}E[PnL] = {pnl.mean():.2f}   "
          f"Std = {pnl.std():.2f}   "
          f"Sharpe = {pnl.mean()/pnl.std():.4f}")
    print(f"{prefix}P5 = {np.percentile(pnl,5):.1f}   "
          f"P25 = {np.percentile(pnl,25):.1f}   "
          f"P75 = {np.percentile(pnl,75):.1f}   "
          f"P95 = {np.percentile(pnl,95):.1f}")


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_all(pnl_ana, pnl_grad, pnl_mlp, grad_epnl, mlp_epnl,
             q_ana, q_grad, q_mlp):
    fig = plt.figure(figsize=(16, 9))
    gs  = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.32)
    fig.suptitle("Aether Crystal — ML Optimizer Results", fontsize=14, y=0.98)

    colors = {"Analytical": "#1D9E75", "Gradient": "#378ADD", "MLP": "#D85A30"}

    # ── Top-left: PnL distribution comparison ────
    ax = fig.add_subplot(gs[0, 0])
    lo = min(np.percentile(p, 1)  for p in [pnl_ana, pnl_grad, pnl_mlp])
    hi = max(np.percentile(p, 99) for p in [pnl_ana, pnl_grad, pnl_mlp])
    bins = np.linspace(lo, hi, 60)
    for label, pnl, c in [("Analytical", pnl_ana, colors["Analytical"]),
                           ("Gradient",   pnl_grad, colors["Gradient"]),
                           ("MLP",        pnl_mlp,  colors["MLP"])]:
        ax.hist(pnl, bins=bins, alpha=0.5, color=c, density=True,
                label=f"{label}  μ={pnl.mean():.0f}")
    ax.set_xlabel("Episode PnL")
    ax.set_ylabel("Density")
    ax.set_title("PnL distributions")
    ax.legend(fontsize=8)

    # ── Top-middle: Gradient optimizer convergence ────
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(range(1, len(grad_epnl)+1), grad_epnl, color=colors["Gradient"], lw=1.3)
    ax.axhline(pnl_ana.mean(), color=colors["Analytical"], ls="--", lw=1,
               label=f"Analytical target ({pnl_ana.mean():.0f})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("E[PnL] estimate")
    ax.set_title("Gradient optimizer convergence")
    ax.legend(fontsize=8)

    # ── Top-right: MLP convergence ────
    ax = fig.add_subplot(gs[0, 2])
    # Smooth with rolling mean for clarity
    w_size = 20
    smooth = np.convolve(mlp_epnl, np.ones(w_size)/w_size, mode="valid")
    ax.plot(range(1, len(mlp_epnl)+1), mlp_epnl,
            color=colors["MLP"], alpha=0.25, lw=0.8)
    ax.plot(range(w_size, len(mlp_epnl)+1), smooth,
            color=colors["MLP"], lw=1.5, label=f"Smoothed (w={w_size})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("E[PnL] (avg over scenarios)")
    ax.set_title("MLP policy convergence")
    ax.legend(fontsize=8)

    # ── Bottom-left: Position comparison bar chart ────
    ax = fig.add_subplot(gs[1, 0])
    names  = [o["name"] for o in OPTION_CHAIN]
    x      = np.arange(N_OPT)
    w      = 0.28
    ax.bar(x - w,   q_ana,  w, color=colors["Analytical"], alpha=0.85, label="Analytical")
    ax.bar(x,       q_grad, w, color=colors["Gradient"],   alpha=0.85, label="Gradient")
    ax.bar(x + w,   q_mlp,  w, color=colors["MLP"],        alpha=0.85, label="MLP")
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Position (+ long, − short)")
    ax.set_title("Final positions — all methods")
    ax.legend(fontsize=8)

    # ── Bottom-middle: Cumulative PnL paths (sorted) ────
    ax = fig.add_subplot(gs[1, 1])
    pcts = np.linspace(0, 100, len(pnl_ana))
    for label, pnl, c in [("Analytical", pnl_ana, colors["Analytical"]),
                           ("Gradient",   pnl_grad, colors["Gradient"]),
                           ("MLP",        pnl_mlp,  colors["MLP"])]:
        ax.plot(pcts, np.sort(pnl), color=c, lw=1.2, label=label)
    ax.axhline(0, color="gray", lw=0.6, ls="--")
    ax.set_xlabel("Percentile")
    ax.set_ylabel("PnL")
    ax.set_title("PnL — sorted outcomes (P-P plot)")
    ax.legend(fontsize=8)

    # ── Bottom-right: Edge per option ────
    ax = fig.add_subplot(gs[1, 2])
    rng  = np.random.default_rng(0)
    ST_e = build_st_cache(50.0, 300_000, rng)
    edges = []
    for opt in OPTION_CHAIN:
        ev = option_payoff(ST_e[opt["days"]], opt["K"], opt["type"]).mean()
        edges.append(ev - (opt["bid"] + opt["ask"]) / 2)
    bar_colors = [colors["Gradient"] if e > 0 else "#D85A30" for e in edges]
    ax.bar(names, edges, color=bar_colors, alpha=0.85)
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("EV payoff − market mid")
    ax.set_title("Raw edge per option (buy = positive)")

    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t0 = time.time()

    print("━" * 65)
    print(" AETHER CRYSTAL — ML TRADING SYSTEM")
    print(f" σ={SIGMA*100:.0f}%  dt=1/{int(1/DT)}yr  {STEPS_PER_DAY} steps/day  MAX_Q={MAX_Q}")
    print("━" * 65)

    # ── 1. Analytical Optimizer ──────────────────────────────────────────────
    print("\n[1/3] Analytical Optimizer...")
    ana  = AnalyticalOptimizer(S0=50.0, n_paths=300_000, seed=0)
    q_ana, pnl_ana = ana.optimize(verbose=True)

    # ── 2. Gradient Optimizer ────────────────────────────────────────────────
    print("\n[2/3] Gradient Optimizer...")
    grad = GradientOptimizer(S0=50.0, sigma=SIGMA, lr=0.1,
                              batch=30_000, n_epochs=200, seed=1)
    grad.train(verbose=True)
    q_grad, pnl_grad = grad.evaluate(n_eval=200_000)

    # ── 3. MLP Policy ────────────────────────────────────────────────────────
    print("\n[3/3] MLP Policy (training across random market scenarios)...")
    mlp = MLPPolicy(n_hidden=64, lr=3e-3, seed=42)
    mlp.train(n_epochs=300, batch=15_000, n_scenarios=4, verbose=True)
    q_mlp, pnl_mlp = mlp.recommend(S0=50.0, sigma=SIGMA, n_eval=200_000)

    # ── Final comparison ─────────────────────────────────────────────────────
    _header("FINAL COMPARISON")
    print(f"  {'Method':<20}  {'E[PnL]':>10}  {'Std':>10}  {'Sharpe':>8}  {'P5':>10}  {'P95':>10}")
    print("  " + "─" * 72)
    for name, pnl in [("Analytical", pnl_ana), ("Gradient", pnl_grad), ("MLP Policy", pnl_mlp)]:
        print(f"  {name:<20}  {pnl.mean():>10.2f}  {pnl.std():>10.2f}  "
              f"{pnl.mean()/pnl.std():>8.4f}  "
              f"{np.percentile(pnl, 5):>10.2f}  {np.percentile(pnl, 95):>10.2f}")

    # ── Sensitivity: how does E[PnL] vary with S0? ───────────────────────────
    print("\n  MLP sensitivity — E[PnL] vs spot price:")
    print(f"  {'S0':>6}  {'Quantities':>55}  {'Rec. E[PnL]':>12}")
    print("  " + "─" * 78)
    rng_sens = np.random.default_rng(5)
    for S0_test in [35, 40, 45, 50, 55, 60, 65]:
        q_t = mlp.predict(S0_test, SIGMA)
        ST_t = build_st_cache(S0_test, 100_000, rng_sens, SIGMA)
        pnl_t = portfolio_pnl(q_t.astype(float), ST_t)
        q_str = " ".join(f"{qi:+6d}" for qi in q_t)
        print(f"  {S0_test:>6}  {q_str}  {pnl_t.mean():>12.1f}")

    print(f"\n  Total runtime: {time.time()-t0:.1f}s")

    # ── Plot ─────────────────────────────────────────────────────────────────
"""    fig = plot_all(pnl_ana, pnl_grad, pnl_mlp,
                   grad.hist_epnl, mlp.hist_epnl,
                   q_ana, q_grad, q_mlp.astype(float))
    out = "/mnt/user-data/outputs/ac_ml_results.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\n  Plot saved → {out}")"""