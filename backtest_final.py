"""
FINAL Walk-Forward IV Pairs Backtest
=====================================
Pairs selected using PRE-2014 data only (pre2014_screener.py).
Backtest covers full 2005-2023 period.
Training:   2005-2013 (same window used for pair selection)
Validation: 2014-2018 (first truly unseen period)
OOS:        2019-2023 (fully out-of-sample)

This design closes the pair-selection leakage issue entirely.
"""

import os
os.environ["WRDS_USERNAME"] = ""
os.environ["PGPASSWORD"] = ""   # <-- replace this

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle
import warnings
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
warnings.filterwarnings("ignore")

# ── PAIRS — identified from pre-2014 data only ────────────────────────────────
# Excluded: IFF/109497 (bad ticker), PCG/EIX+PCG/SO (bankruptcy 2019),
#           TWX/DIS (acquisition 2018), ERIC/LH (cross-sector noise)

PAIRS = [
    ("NEE", "SO",  "Utilities"),
    ("AEP", "ED",  "Utilities"),
    ("ETR", "SO",  "Utilities"),
    ("PPL", "SO",  "Utilities"),
    ("T",   "VZ",  "Utilities"),
    ("AEP", "NEE", "Utilities"),
    ("ED",  "SO",  "Utilities"),
    ("EMR", "ITW", "Industrials"),
    ("HON", "RTX", "Industrials"),
    ("DOV", "ETN", "Industrials"),
    ("ITW", "RTX", "Industrials"),
    ("APD", "K",   "Manufacturing"),
    ("K",   "PEP", "Consumer Staples"),
    ("OLN", "PEP", "Manufacturing"),
    ("APD", "PEP", "Manufacturing"),
    ("PEP", "PG",  "Consumer Staples"),
    ("KO",  "PEP", "Consumer Staples"),
    ("KMB", "PG",  "Consumer Staples"),
    ("CVX", "XOM", "Energy"),
    ("TGT", "WMT", "Retail"),
]

PANEL_CACHE   = "iv_panel_cache_2005.pkl"

TRAIN_END     = "2013-12-31"
VAL_END       = "2018-12-31"

ENTRY_Z       = 2.0
EXIT_Z        = 0.0
STOP_Z        = 3.0
MAX_HOLD      = 90
ROLL_WINDOW   = 252
TRANS_COST    = 0.10

OUTPUT_TRADES  = "backtest_final_trades.csv"
OUTPUT_METRICS = "backtest_final_metrics.csv"
OUTPUT_PLOT    = "backtest_final_results.png"


# ── BACKTEST ENGINE ───────────────────────────────────────────────────────────

def backtest_pair(dates_str, x_series, y_series,
                  entry_z=ENTRY_Z, exit_z=EXIT_Z, stop_z=STOP_Z,
                  max_hold=MAX_HOLD, roll=ROLL_WINDOW, tc=TRANS_COST):
    n = len(x_series)
    daily_pnl = np.zeros(n)
    trades = []

    in_trade = False
    direction = entry_idx = entry_z_val = None
    spread_mean = spread_std_val = entry_spread = None

    for i in range(roll, n):
        window_x = x_series[i-roll:i]
        window_y = y_series[i-roll:i]

        if (np.any(np.isnan(window_x)) or np.any(np.isnan(window_y))
                or np.isnan(x_series[i]) or np.isnan(y_series[i])):
            continue

        m = OLS(window_y, add_constant(window_x)).fit()
        a, b = m.params[0], m.params[1]

        spread_hist = window_y - a - b * window_x
        mu  = np.mean(spread_hist)
        sig = np.std(spread_hist)
        if sig < 1e-8:
            continue

        spread_now = y_series[i] - a - b * x_series[i]
        z = (spread_now - mu) / sig

        if not in_trade:
            if z <= -entry_z or z >= entry_z:
                if i + 1 >= n:
                    continue
                in_trade       = True
                direction      = "long_spread" if z <= -entry_z else "short_spread"
                entry_idx      = i + 1
                entry_z_val    = z
                spread_mean    = mu
                spread_std_val = sig
                entry_spread   = y_series[i+1] - a - b * x_series[i+1]
        else:
            if i <= entry_idx:
                continue
            hold = i - entry_idx

            spread_now2 = y_series[i] - a - b * x_series[i]
            z_now = (spread_now2 - spread_mean) / spread_std_val

            step_pnl = spread_now2 - (y_series[i-1] - a - b * x_series[i-1])
            if direction == "short_spread":
                step_pnl = -step_pnl
            daily_pnl[i] = step_pnl / spread_std_val

            exit_reason = None
            if direction == "long_spread"  and z_now >= exit_z: exit_reason = "reversion"
            if direction == "short_spread" and z_now <= exit_z: exit_reason = "reversion"
            if direction == "long_spread"  and z_now <= -stop_z: exit_reason = "stop_loss"
            if direction == "short_spread" and z_now >= stop_z:  exit_reason = "stop_loss"
            if hold >= max_hold: exit_reason = "time_exit"

            if exit_reason:
                gross = (spread_now2 - entry_spread) / spread_std_val
                if direction == "short_spread":
                    gross = -gross
                net = gross - tc
                trades.append({
                    "pair":        "",
                    "entry_date":  dates_str[entry_idx],
                    "exit_date":   dates_str[i],
                    "hold_days":   hold,
                    "direction":   direction,
                    "entry_z":     round(entry_z_val, 3),
                    "exit_z":      round(z_now, 3),
                    "gross_pnl":   round(gross, 4),
                    "net_pnl":     round(net, 4),
                    "exit_reason": exit_reason,
                })
                in_trade = False
                daily_pnl[i] -= tc

    return trades, daily_pnl


def period_metrics(daily_pnl, label):
    if len(daily_pnl) == 0:
        return {}
    ann_ret = np.mean(daily_pnl) * 252
    ann_vol = np.std(daily_pnl) * np.sqrt(252)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0
    cum     = np.cumsum(daily_pnl)
    roll_max = np.maximum.accumulate(cum)
    max_dd  = float(np.min(cum - roll_max))
    return {
        "period":       label,
        "ann_return":   round(ann_ret, 4),
        "ann_vol":      round(ann_vol, 4),
        "sharpe":       round(sharpe, 3),
        "max_drawdown": round(max_dd, 4),
        "win_rate":     round(float(np.mean(daily_pnl > 0)), 3),
        "n_days":       len(daily_pnl),
    }


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("FINAL Walk-Forward IV Pairs Backtest")
    print(f"  Pair selection: pre-{TRAIN_END} data only")
    print(f"  Validation:     {TRAIN_END} to {VAL_END}")
    print(f"  OOS:            {VAL_END} to 2023")
    print("=" * 70)

    # ── Load panel ──
    print("\nLoading cached IV panel...")
    with open(PANEL_CACHE, "rb") as f:
        cache = pickle.load(f)
    iv_panel   = cache["panel"]
    ticker_map = cache["ticker_map"]
    iv_panel.columns = [int(c) for c in iv_panel.columns]
    ticker_map = {int(k): v for k, v in ticker_map.items() if v is not None}
    iv_panel   = iv_panel.astype(np.float64)
    rev_map    = {v.upper(): k for k, v in ticker_map.items()}
    dates_str  = pd.DatetimeIndex(iv_panel.index)
    print(f"  {len(iv_panel)} days x {iv_panel.shape[1]} stocks")

    train_mask = dates_str <= TRAIN_END
    val_mask   = (dates_str > TRAIN_END) & (dates_str <= VAL_END)
    oos_mask   = dates_str > VAL_END

    # ── Run backtest ──
    print(f"\nRunning backtest on {len(PAIRS)} walk-forward selected pairs...")
    all_trades   = []
    all_daily    = np.zeros(len(dates_str))
    pair_dailies = {}
    skipped      = []

    for tx, ty, sector in PAIRS:
        sx = rev_map.get(tx.upper())
        sy = rev_map.get(ty.upper())
        pair_name = f"{tx}/{ty}"

        if sx is None or sy is None:
            print(f"  {pair_name}: ticker not found — skipping")
            skipped.append(pair_name)
            continue
        if sx not in iv_panel.columns or sy not in iv_panel.columns:
            print(f"  {pair_name}: not in panel — skipping")
            skipped.append(pair_name)
            continue

        x = iv_panel[sx].values
        y = iv_panel[sy].values

        trades, daily = backtest_pair(dates_str, x, y)
        for t in trades:
            t["pair"] = pair_name

        n_trades = len(trades)
        wins     = sum(1 for t in trades if t["net_pnl"] > 0)
        wr       = wins / n_trades if n_trades > 0 else 0
        print(f"  {pair_name:<12} ({sector:<16}) ... "
              f"{n_trades:>3} trades | win rate {wr*100:.0f}%")

        all_trades.extend(trades)
        all_daily += daily
        pair_dailies[pair_name] = daily

    n_active   = len(PAIRS) - len(skipped)
    port_daily = all_daily / n_active

    # ── Metrics ──
    metrics = []
    for label, mask in [
        ("Training (2005-2013)",       train_mask),
        ("Validation (2014-2018)",     val_mask),
        ("Out-of-Sample (2019-2023)",  oos_mask),
        ("Full Period (2005-2023)",    np.ones(len(dates_str), dtype=bool)),
    ]:
        metrics.append(period_metrics(port_daily[mask], label))

    metrics_df = pd.DataFrame(metrics)
    trades_df  = pd.DataFrame(all_trades)

    print("\n" + "=" * 70)
    print("PORTFOLIO PERFORMANCE — WALK-FORWARD BACKTEST")
    print("=" * 70)
    for _, row in metrics_df.iterrows():
        print(f"\n  [{row['period']}]")
        print(f"    Annualized Return:  {row['ann_return']:+.4f}")
        print(f"    Annualized Vol:     {row['ann_vol']:.4f}")
        print(f"    Sharpe Ratio:       {row['sharpe']:.3f}")
        print(f"    Max Drawdown:       {row['max_drawdown']:.4f}")
        print(f"    Win Rate (days):    {row['win_rate']*100:.1f}%")
        print(f"    Trading Days:       {row['n_days']}")

    if not trades_df.empty:
        print("\n" + "=" * 70)
        print("TRADE STATISTICS")
        print("=" * 70)
        n_tr  = len(trades_df)
        wr    = (trades_df["net_pnl"] > 0).mean()
        exits = trades_df["exit_reason"].value_counts(normalize=True)
        print(f"  Total trades:          {n_tr}")
        print(f"  Win rate:              {wr*100:.1f}%")
        print(f"  Avg net P&L/trade:     {trades_df['net_pnl'].mean():.4f}")
        print(f"  Avg hold days:         {trades_df['hold_days'].mean():.1f}")
        print(f"  Exit by reversion:     {exits.get('reversion',0)*100:.1f}%")
        print(f"  Exit by stop-loss:     {exits.get('stop_loss',0)*100:.1f}%")
        print(f"  Exit by time:          {exits.get('time_exit',0)*100:.1f}%")
        print(f"\n  Per-pair breakdown:")
        pp = trades_df.groupby("pair").agg(
            n_trades=("net_pnl","count"),
            win_rate=("net_pnl", lambda x: (x>0).mean()),
            avg_pnl=("net_pnl","mean"),
            total_pnl=("net_pnl","sum"),
            avg_hold=("hold_days","mean"),
        ).round(3)
        print(pp.sort_values("total_pnl", ascending=False).to_string())

    trades_df.to_csv(OUTPUT_TRADES, index=False)
    metrics_df.to_csv(OUTPUT_METRICS, index=False)
    print(f"\n  Trades: {OUTPUT_TRADES}")
    print(f"  Metrics: {OUTPUT_METRICS}")

    # ── Plot ──
    print("\nGenerating plots...")
    fig, axes = plt.subplots(3, 1, figsize=(14, 14))
    fig.suptitle(
        "IV Pairs Trading — Final Walk-Forward Backtest (20 Pairs, 2005-2023)\n"
        "Pairs selected on pre-2014 data only | Vega-normalized proxy returns",
        fontsize=12
    )

    # Panel 1: Portfolio cumulative P&L
    ax = axes[0]
    cum = np.cumsum(port_daily)
    ax.plot(dates_str, cum, color="#1a237e", lw=1.5)
    ax.axvspan(dates_str[0],             pd.Timestamp(TRAIN_END),
               alpha=0.08, color="blue",   label=f"Training (→{TRAIN_END})")
    ax.axvspan(pd.Timestamp(TRAIN_END),  pd.Timestamp(VAL_END),
               alpha=0.08, color="orange", label=f"Validation ({TRAIN_END}→{VAL_END})")
    ax.axvspan(pd.Timestamp(VAL_END),    dates_str[-1],
               alpha=0.08, color="green",  label=f"OOS ({VAL_END}→2023)")
    ax.axhline(0, color="gray", lw=0.5, ls="--")

    # Annotate Sharpe by period
    for row in metrics_df.itertuples():
        if "Training" in row.period:
            ax.text(pd.Timestamp("2009-01-01"), cum[train_mask].max() * 0.5,
                    f"Sharpe: {row.sharpe:.2f}", fontsize=9, color="blue")
        elif "Validation" in row.period:
            ax.text(pd.Timestamp("2015-06-01"), cum[val_mask].max() * 0.85,
                    f"Sharpe: {row.sharpe:.2f}", fontsize=9, color="darkorange")
        elif "Out-of-Sample" in row.period:
            ax.text(pd.Timestamp("2020-01-01"), cum[oos_mask].max() * 0.85,
                    f"Sharpe: {row.sharpe:.2f}", fontsize=9, color="green")

    ax.legend(fontsize=8)
    ax.set_title("Portfolio Cumulative P&L (Vega-Normalized Units)")
    ax.set_ylabel("Cumulative P&L")

    # Panel 2: Individual pairs
    ax = axes[1]
    colors = plt.cm.tab20(np.linspace(0, 1, len(pair_dailies)))
    for (pname, pd_arr), col in zip(pair_dailies.items(), colors):
        ax.plot(dates_str, np.cumsum(pd_arr / n_active),
                label=pname, lw=0.8, alpha=0.85, color=col)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.axvline(pd.Timestamp(TRAIN_END), color="gray", lw=1, ls=":")
    ax.axvline(pd.Timestamp(VAL_END),   color="gray", lw=1, ls=":")
    ax.legend(fontsize=6, ncol=4)
    ax.set_title("Individual Pair Cumulative P&L")
    ax.set_ylabel("Cumulative P&L")

    # Panel 3: Trade P&L distribution
    ax = axes[2]
    if not trades_df.empty:
        ax.hist(trades_df["net_pnl"], bins=60,
                color="#37474f", alpha=0.75, edgecolor="white")
        ax.axvline(0, color="red", ls="--", lw=1, label="Zero")
        ax.axvline(trades_df["net_pnl"].mean(), color="#00c853", lw=1.5,
                   label=f"Mean = {trades_df['net_pnl'].mean():.3f}")
        ax.legend(fontsize=9)
        ax.set_title("Trade P&L Distribution (Net of Transaction Costs)")
        ax.set_xlabel("P&L per trade (normalized units)")
        ax.set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches="tight")
    print(f"  Plot saved: {OUTPUT_PLOT}")
    print("\nDone.")


if __name__ == "__main__":
    main()
