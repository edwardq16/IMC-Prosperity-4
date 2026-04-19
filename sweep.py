"""
Parameter sweep for IMC Prosperity 4 Round 1 trader.
Tries all combinations of MAX_SKEW and SPREAD, runs the backtester on each,
and prints a results table sorted by total PnL.

Usage:
    py sweep.py

Make sure trader_round_1.py is in the same folder as this script,
and that your Backtesting data folder is accessible.
"""

import subprocess
import re
import os
import shutil
import tempfile

# ── Parameters to sweep ───────────────────────────────────────────────────────

SKEW_VALUES   = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
SPREAD_VALUES = [1, 2, 3, 4]

# ── Paths ─────────────────────────────────────────────────────────────────────

TRADER_FILE = "trader_round_1.py"
DATA_DIR    = "Backtesting data"
ROUND       = 1

# ── Run one backtest ──────────────────────────────────────────────────────────

def run_backtest(skew: float, spread: int) -> dict:
    """
    Creates a temporary copy of the trader with the given parameters,
    runs the backtester on it, and returns the results.
    """
    with open(TRADER_FILE, 'r') as f:
        source = f.read()

    # Replace the parameter values
    source = re.sub(r'SPREAD\s*=\s*[\d.]+',   f'SPREAD = {spread}', source)
    source = re.sub(r'MAX_SKEW\s*=\s*[\d.]+', f'MAX_SKEW = {skew}', source)

    # Write to a temp file
    tmp = tempfile.NamedTemporaryFile(
        mode='w', suffix='.py', delete=False,
        dir='.', prefix='_sweep_tmp_'
    )
    tmp.write(source)
    tmp.close()

    try:
        result = subprocess.run(
            ['py', '-m', 'prosperity4bt', tmp.name, str(ROUND), '--no-out'],
            capture_output=True, text=True,
            encoding='utf-8', errors='replace'
        )
        output = (result.stdout or '') + (result.stderr or '')
    finally:
        os.unlink(tmp.name)

    return parse_output(output, skew, spread)


def parse_output(output: str, skew: float, spread: int) -> dict:
    """Extract per-product and total PnL from backtester output."""
    total_match = re.findall(r'Total profit:\s*([\d,]+)', output)
    aco_match   = re.findall(r'ASH_COATED_OSMIUM:\s*([\d,]+)', output)
    ipr_match   = re.findall(r'INTARIAN_PEPPER_ROOT:\s*([\d,]+)', output)

    def sum_matches(matches):
        return sum(int(m.replace(',', '')) for m in matches)

    return {
        'skew':   skew,
        'spread': spread,
        'total':  sum_matches(total_match),
        'aco':    sum_matches(aco_match),
        'ipr':    sum_matches(ipr_match),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    results = []
    total_runs = len(SKEW_VALUES) * len(SPREAD_VALUES)
    run = 0

    print(f"Running {total_runs} backtests...\n")

    for spread in SPREAD_VALUES:
        for skew in SKEW_VALUES:
            run += 1
            print(f"  [{run}/{total_runs}] spread={spread}, max_skew={skew}...", end=' ', flush=True)
            r = run_backtest(skew, spread)
            results.append(r)
            print(f"PnL={r['total']:,}")

    # Sort by total PnL descending
    results.sort(key=lambda x: x['total'], reverse=True)

    # Print results table
    print(f"\n{'Rank':<6} {'Spread':<8} {'Max skew':<10} {'ACO':>10} {'IPR':>12} {'Total PnL':>12}")
    print("-" * 62)
    for i, r in enumerate(results):
        marker = " <-- best" if i == 0 else ""
        print(f"{i+1:<6} {r['spread']:<8} {r['skew']:<10} "
              f"{r['aco']:>10,} {r['ipr']:>12,} {r['total']:>12,}{marker}")

    best = results[0]
    print(f"\nBest: spread={best['spread']}, max_skew={best['skew']}, "
          f"total PnL={best['total']:,}")
    print(f"ACO: {best['aco']:,}   IPR: {best['ipr']:,}")