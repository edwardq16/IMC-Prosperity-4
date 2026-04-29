"""
Aether Crystal (AC) — GBM Price Simulator
==========================================
Parameters (fixed, per spec):
  - Zero risk-neutral drift (mu = 0)
  - Annualized volatility: 251%
  - 4 steps per trading day
  - 252 trading days per year  →  dt = 1 / (252 * 4) = 1 / 1008
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional


# ── Constants ────────────────────────────────────────────────────────────────

SIGMA        = 2.51          # 251% annualized vol
TRADING_DAYS = 252
STEPS_PER_DAY = 4
DT           = 1.0 / (TRADING_DAYS * STEPS_PER_DAY)   # ~0.000992 years
DRIFT        = 0.0           # zero risk-neutral drift


# ── Core GBM step ───────────────────────────────────────────────────────────

def gbm_step(S: np.ndarray, dt: float = DT, sigma: float = SIGMA, rng=None) -> np.ndarray:
    """
    Advance a set of prices by one discrete GBM step.

    S(t+dt) = S(t) * exp( (mu - sigma²/2)*dt  +  sigma*sqrt(dt)*Z )
    With mu=0: drift term is just -sigma²/2 * dt.

    Args:
        S     : array of current prices (any shape)
        dt    : time step in years
        sigma : annualized volatility
        rng   : numpy Generator (for reproducibility); uses default if None

    Returns:
        Array of next prices, same shape as S.
    """
    if rng is None:
        rng = np.random.default_rng()
    Z = rng.standard_normal(S.shape)
    return S * np.exp((DRIFT - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)


# ── Simulation engine ────────────────────────────────────────────────────────

@dataclass
class AetherCrystalSim:
    """
    Simulates Aether Crystal prices on the discrete GBM grid.

    Usage:
        sim = AetherCrystalSim(S0=50.0, n_paths=10_000, seed=42)
        paths = sim.run(days=21)        # shape: (n_steps+1, n_paths)
        terminal = paths[-1]            # terminal prices
    """
    S0:      float = 50.0
    n_paths: int   = 10_000
    sigma:   float = SIGMA
    dt:      float = DT
    seed:    Optional[int] = None

    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)

    def run(self, days: int = 21) -> np.ndarray:
        """
        Simulate `days` trading days (days * STEPS_PER_DAY discrete steps).

        Returns:
            paths : ndarray of shape (n_steps + 1, n_paths)
                    Row 0 is S0, row k is the price after k steps.
        """
        n_steps = days * STEPS_PER_DAY
        paths = np.empty((n_steps + 1, self.n_paths))
        paths[0] = self.S0

        for t in range(n_steps):
            paths[t + 1] = gbm_step(paths[t], self.dt, self.sigma, self._rng)

        return paths

    def run_vectorized(self, days: int = 21) -> np.ndarray:
        """
        Faster all-at-once version — generates all random numbers in one call.
        Identical distribution to run(), but avoids the Python loop.
        """
        n_steps = days * STEPS_PER_DAY
        Z = self._rng.standard_normal((n_steps, self.n_paths))
        log_returns = (DRIFT - 0.5 * self.sigma**2) * self.dt + self.sigma * np.sqrt(self.dt) * Z
        log_paths = np.vstack([np.zeros(self.n_paths), np.cumsum(log_returns, axis=0)])
        return self.S0 * np.exp(log_paths)


# ── Option pricers (Monte Carlo, using simulated paths) ──────────────────────

def mc_price(paths: np.ndarray, K: float, option_type: str = "call", r: float = 0.0) -> dict:
    """
    Price a European option from simulated terminal prices.

    Args:
        paths       : output of AetherCrystalSim.run()
        K           : strike price
        option_type : 'call' or 'put'
        r           : risk-free rate (default 0)

    Returns:
        dict with keys: price, stderr, ci_low, ci_high
    """
    T = paths.shape[0] * DT            # total time in years
    ST = paths[-1]                     # terminal prices

    if option_type == "call":
        payoffs = np.maximum(ST - K, 0)
    elif option_type == "put":
        payoffs = np.maximum(K - ST, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    discount = np.exp(-r * T)
    price  = discount * payoffs.mean()
    stderr = discount * payoffs.std(ddof=1) / np.sqrt(len(payoffs))

    return {
        "price":   price,
        "stderr":  stderr,
        "ci_low":  price - 1.96 * stderr,
        "ci_high": price + 1.96 * stderr,
    }


# ── Convenience: full chain pricer ───────────────────────────────────────────

OPTION_CHAIN = [
    {"name": "AC_50_P",   "K": 50, "type": "put",  "days": 21, "market_bid": 12.00,  "market_ask": 12.05},
    {"name": "AC_50_C",   "K": 50, "type": "call", "days": 21, "market_bid": 12.00,  "market_ask": 12.05},
    {"name": "AC_35_P",   "K": 35, "type": "put",  "days": 21, "market_bid": 4.33,   "market_ask": 4.35},
    {"name": "AC_40_P",   "K": 40, "type": "put",  "days": 21, "market_bid": 6.50,   "market_ask": 6.55},
    {"name": "AC_45_P",   "K": 45, "type": "put",  "days": 21, "market_bid": 9.05,   "market_ask": 9.10},
    {"name": "AC_60_C",   "K": 60, "type": "call", "days": 21, "market_bid": 8.80,   "market_ask": 8.85},
    {"name": "AC_50_P_2", "K": 50, "type": "put",  "days": 14, "market_bid": 9.70,   "market_ask": 9.75},
]

def price_chain(n_paths: int = 100_000, seed: int = 42) -> list[dict]:
    """Price every option in the chain via MC and compare to market."""
    results = []
    for opt in OPTION_CHAIN:
        sim = AetherCrystalSim(S0=50.0, n_paths=n_paths, seed=seed)
        paths = sim.run_vectorized(days=opt["days"])
        mc = mc_price(paths, opt["K"], opt["type"])

        market_mid = (opt["market_bid"] + opt["market_ask"]) / 2
        edge = market_mid - mc["price"]

        results.append({
            **opt,
            "mc_price": mc["price"],
            "mc_stderr": mc["stderr"],
            "market_mid": market_mid,
            "edge": edge,
        })
    return results


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_paths(paths: np.ndarray, n_display: int = 50, days: int = 21, S0: float = 50.0):
    """Plot a sample of simulated price paths."""
    n_steps = paths.shape[0]
    t = np.linspace(0, days, n_steps)   # in trading days

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("Aether Crystal — GBM Simulation (σ=251%, μ=0)", fontsize=13)

    # Left: sample paths
    ax = axes[0]
    ax.plot(t, paths[:, :n_display], lw=0.6, alpha=0.35, color="#1D9E75")
    ax.axhline(S0, color="#888", lw=0.8, ls="--", label=f"S₀ = {S0}")
    ax.set_xlabel("Trading days")
    ax.set_ylabel("Price (AC)")
    ax.set_title(f"{n_display} sample paths")
    ax.legend(fontsize=9)

    # Right: terminal distribution
    ax = axes[1]
    ST = paths[-1]
    ax.hist(ST, bins=80, color="#1D9E75", alpha=0.7, edgecolor="none", density=True)
    ax.axvline(ST.mean(), color="#0f6e56", lw=1.5, ls="--", label=f"mean={ST.mean():.2f}")
    ax.axvline(np.median(ST), color="#993c1d", lw=1.5, ls="--", label=f"median={np.median(ST):.2f}")
    ax.set_xlabel("Terminal price (AC)")
    ax.set_ylabel("Density")
    ax.set_title(f"Terminal distribution (T={days}d, N={paths.shape[1]:,})")
    ax.legend(fontsize=9)

    plt.tight_layout()
    return fig

# ── Main demo ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print(" Aether Crystal GBM Simulator")
    print(f" σ={SIGMA*100:.0f}%  dt=1/{int(1/DT)} yr  steps/day={STEPS_PER_DAY}")
    print("=" * 60)

    # 1. Run simulation
    N = 200_000
    sim = AetherCrystalSim(S0=50.0, n_paths=N, seed=42)
    paths_21 = sim.run_vectorized(days=21)

    # 2. Basic stats
    ST = paths_21[-1]
    print(f"\nTerminal prices after 21 days (N={N:,}):")
    print(f"  Mean:    {ST.mean():.4f}   (expect ≈ S0={sim.S0})")
    print(f"  Median:  {np.median(ST):.4f}")
    print(f"  Std dev: {ST.std():.4f}")
    print(f"  P5/P95:  {np.percentile(ST, 5):.2f} / {np.percentile(ST, 95):.2f}")

    # 3. Price the full option chain
    print("\nOption chain — MC prices vs market:")
    print(f"{'Option':<14} {'K':>5} {'Type':>5} {'Days':>5} {'MC Price':>10} "
          f"{'±95% CI':>10} {'Mkt mid':>9} {'Edge':>8}")
    print("-" * 75)
    for r in price_chain(n_paths=N, seed=42):
        ci = f"±{1.96*r['mc_stderr']:.3f}"
        edge_str = f"{r['edge']:+.3f}"
        print(f"{r['name']:<14} {r['K']:>5} {r['type']:>5} {r['days']:>5}  "
              f"{r['mc_price']:>9.4f} {ci:>11} {r['market_mid']:>9.3f} {edge_str:>8}")

    # 4. Plot
    fig = plot_paths(paths_21, n_display=80, days=21)
    out = "/mnt/user-data/outputs/ac_sim_paths.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nPlot saved to {out}")

