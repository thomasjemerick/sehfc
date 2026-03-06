"""
S&P 500 IV Pairs Screener — Fast (122x speedup via OLS residual ADF)
=====================================================================
Replaces statsmodels coint() with fast OLS+ADF implementation.
153k pairs in ~8 minutes, full 438k in ~20 minutes.
"""

import os
os.environ["WRDS_USERNAME"] = ""
os.environ["PGPASSWORD"] = ""   # <-- replace this

import numpy as np
import pandas as pd
import wrds
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from itertools import combinations
import warnings
import time
import pickle
warnings.filterwarnings("ignore")

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

START_YEAR    = 2010
END_YEAR      = 2023
TARGET_DAYS   = 182
N_YEARS       = END_YEAR - START_YEAR + 1
TRADING_DAYS  = N_YEARS * 252

MIN_COVERAGE  = 0.70
MIN_CORR      = 0.70
COINT_PVAL    = 0.05
ADF_PVAL      = 0.05
MIN_HL        = 5
MAX_HL        = 150
MIN_OBS       = 756

N_CLUSTERS    = 25
OUTPUT_FILE   = "sp500_pairs_screener_results.csv"
PANEL_CACHE   = "iv_panel_cache.pkl"


# ── FAST COINTEGRATION ────────────────────────────────────────────────────────

def fast_coint(y, x):
    """
    Engle-Granger cointegration: OLS regression + ADF on residuals.
    Both directions, worst p-value. 122x faster than statsmodels coint().
    Returns: (p_value, intercept, slope)
    """
    b1 = np.polyfit(x, y, 1)
    p_yx = adfuller(y - b1[0]*x - b1[1], maxlag=1, autolag=None)[1]

    b2 = np.polyfit(y, x, 1)
    p_xy = adfuller(x - b2[0]*y - b2[1], maxlag=1, autolag=None)[1]

    if p_yx <= p_xy:
        return max(p_yx, p_xy), b1[1], b1[0], "Y~X"
    else:
        return max(p_yx, p_xy), b2[1], b2[0], "X~Y"


# ── WRDS DATA FUNCTIONS ───────────────────────────────────────────────────────

def get_liquid_secids(db):
    years = range(START_YEAR, END_YEAR + 1)
    union_parts = [f"""
        SELECT secid, COUNT(*) as n_obs
        FROM optionm.stdopd{yr}
        WHERE days = {TARGET_DAYS}
          AND ABS(strike_price - forward_price) < 0.01 * forward_price
          AND impl_volatility IS NOT NULL AND impl_volatility > 0
          AND forward_price > 0
        GROUP BY secid
    """ for yr in years]
    query = f"""
        SELECT secid, SUM(n_obs) as total_obs
        FROM ({" UNION ALL ".join(union_parts)}) t
        GROUP BY secid
        HAVING SUM(n_obs) >= {int(TRADING_DAYS * MIN_COVERAGE)}
        ORDER BY total_obs DESC
    """
    df = db.raw_sql(query)
    secids = df["secid"].astype(int).tolist()
    print(f"  Found {len(secids)} liquid secids")
    return secids


def pull_iv_batched(db, secids, batch_size=50):
    years   = range(START_YEAR, END_YEAR + 1)
    batches = [secids[i:i+batch_size] for i in range(0, len(secids), batch_size)]
    print(f"  Pulling IV in {len(batches)} batches...")
    frames  = []
    for bi, batch in enumerate(batches):
        sid_str = ", ".join(str(s) for s in batch)
        parts   = [f"""
            SELECT date, secid, AVG(impl_volatility) AS impl_volatility
            FROM optionm.stdopd{yr}
            WHERE secid IN ({sid_str})
              AND days = {TARGET_DAYS}
              AND ABS(strike_price - forward_price) < 0.01 * forward_price
              AND impl_volatility IS NOT NULL AND impl_volatility > 0
              AND forward_price > 0
            GROUP BY date, secid
        """ for yr in years]
        try:
            df = db.raw_sql(" UNION ALL ".join(parts) + " ORDER BY secid, date",
                            date_cols=["date"])
            frames.append(df)
        except Exception as e:
            print(f"    Batch {bi+1} failed: {e}")
        if (bi+1) % 20 == 0:
            print(f"    {bi+1}/{len(batches)} batches done...")
    combined = pd.concat(frames, ignore_index=True)
    print(f"  Total rows: {len(combined):,}")
    return combined


def build_iv_panel(raw_iv):
    raw_iv["secid"] = raw_iv["secid"].astype(int)
    panel = raw_iv.pivot_table(
        index="date", columns="secid", values="impl_volatility", aggfunc="mean"
    ).sort_index().ffill(limit=3)
    min_obs = int(len(panel) * MIN_COVERAGE)
    panel   = panel.dropna(thresh=min_obs, axis=1)
    panel.columns = [int(c) for c in panel.columns]
    panel = panel.astype(np.float64)
    print(f"  IV panel: {len(panel)} days x {panel.shape[1]} stocks")
    return panel


def get_ticker_map(db, secids):
    sid_str = ", ".join(str(s) for s in secids)
    df = db.raw_sql(f"""
        SELECT DISTINCT ON (secid) secid, ticker
        FROM optionm.secnmd
        WHERE secid IN ({sid_str})
        ORDER BY secid, effect_date DESC
    """)
    return {int(k): v for k, v in zip(df["secid"], df["ticker"])}


# ── CLUSTERING ────────────────────────────────────────────────────────────────

def cluster_stocks(iv_panel):
    print(f"  Clustering {iv_panel.shape[1]} stocks into {N_CLUSTERS} clusters...")
    features = pd.DataFrame({
        "mean_iv":  iv_panel.mean(),
        "std_iv":   iv_panel.std(),
        "skew_iv":  iv_panel.skew(),
        "trend_iv": iv_panel.apply(
            lambda c: np.polyfit(np.arange(len(c.dropna())), c.dropna().values, 1)[0]
            if len(c.dropna()) > 10 else 0
        ),
    }).dropna()
    n = min(N_CLUSTERS, len(features) // 3)
    X = StandardScaler().fit_transform(features)
    features["cluster"] = KMeans(n_clusters=n, random_state=42, n_init=10).fit_predict(X)
    counts = features["cluster"].value_counts()
    print(f"  {n} clusters | min={counts.min()} max={counts.max()} mean={counts.mean():.1f}")
    return features["cluster"]


def get_candidate_pairs(cluster_labels):
    pairs = []
    for cid in cluster_labels.unique():
        members = [int(m) for m in cluster_labels[cluster_labels == cid].index.tolist()]
        if len(members) >= 2:
            pairs.extend(combinations(members, 2))
    print(f"  Candidate pairs from clustering: {len(pairs):,}")
    return pairs


# ── CORRELATION PRE-FILTER ────────────────────────────────────────────────────

def corr_prefilter(panel_values, panel_columns, candidate_pairs):
    print(f"  Pre-filtering {len(candidate_pairs):,} pairs by correlation >= {MIN_CORR}...")
    t0 = time.time()
    col_idx = {int(c): i for i, c in enumerate(panel_columns)}
    kept = []
    for sx, sy in candidate_pairs:
        ix = col_idx.get(int(sx))
        iy = col_idx.get(int(sy))
        if ix is None or iy is None:
            continue
        x = panel_values[:, ix]
        y = panel_values[:, iy]
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() < MIN_OBS:
            continue
        if abs(np.corrcoef(x[mask], y[mask])[0, 1]) >= MIN_CORR:
            kept.append((sx, sy))
    print(f"  Kept {len(kept):,} pairs ({len(kept)/len(candidate_pairs)*100:.1f}%) "
          f"in {time.time()-t0:.0f}s")
    return kept


# ── SCREENING ─────────────────────────────────────────────────────────────────

def screen_pairs(panel_values, panel_columns, candidate_pairs, ticker_map):
    col_idx  = {int(c): i for i, c in enumerate(panel_columns)}
    results  = []
    n        = len(candidate_pairs)
    t0       = time.time()

    print(f"  Screening {n:,} pairs with fast cointegration...")

    for i, (sx, sy) in enumerate(candidate_pairs):
        sx = int(sx)
        sy = int(sy)
        ix = col_idx.get(sx)
        iy = col_idx.get(sy)
        if ix is None or iy is None:
            continue

        try:
            x = panel_values[:, ix]
            y = panel_values[:, iy]
            mask = ~(np.isnan(x) | np.isnan(y))
            x = x[mask];  y = y[mask]
            if len(x) < MIN_OBS:
                continue

            # Stage 1: fast cointegration
            p, intercept, slope, direction = fast_coint(y, x)
            if p >= COINT_PVAL:
                continue

            spread = (y - intercept - slope*x) if direction == "Y~X" \
                     else (x - intercept - slope*y)

            # Stage 2: ADF on spread
            adf_p = adfuller(spread, maxlag=1, autolag=None)[1]
            if adf_p >= ADF_PVAL:
                continue

            # Stage 3: OU half-life
            diff = np.diff(spread)
            lag  = spread[:-1]
            lam  = OLS(diff, add_constant(lag)).fit().params[1]
            if lam >= 0:
                continue
            hl = -np.log(2) / lam
            if not (MIN_HL < hl < MAX_HL):
                continue

            results.append({
                "secid_x":     sx,  "secid_y":     sy,
                "ticker_x":    ticker_map.get(sx, str(sx)),
                "ticker_y":    ticker_map.get(sy, str(sy)),
                "coint_pval":  round(p, 5),
                "adf_pval":    round(adf_p, 5),
                "hedge_ratio": round(slope, 4),
                "ou_halflife": round(hl, 1),
                "spread_std":  round(float(np.std(spread)), 6),
                "n_obs":       int(len(spread)),
            })

        except Exception:
            continue

        if (i + 1) % 5000 == 0:
            elapsed  = time.time() - t0
            rate     = (i + 1) / elapsed
            eta_min  = (n - i - 1) / rate / 60
            print(f"    {i+1:,}/{n:,} | {len(results):,} viable | "
                  f"{rate:.0f} pairs/sec | ETA {eta_min:.1f} min")

        if len(results) > 0 and len(results) % 2000 == 0:
            pd.DataFrame(results).to_csv(OUTPUT_FILE + ".tmp", index=False)
            print(f"    Checkpoint: {len(results):,} viable pairs saved")

    return results


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    print("=" * 70)
    print("S&P 500 IV Pairs Screener — Fast OLS+ADF Cointegration")
    print("=" * 70)

    if os.path.exists(PANEL_CACHE):
        print("\nLoading cached IV panel...")
        with open(PANEL_CACHE, "rb") as f:
            cache = pickle.load(f)
        iv_panel   = cache["panel"]
        ticker_map = cache["ticker_map"]
        iv_panel.columns = [int(c) for c in iv_panel.columns]
        ticker_map = {int(k): v for k, v in ticker_map.items()}
        iv_panel   = iv_panel.astype(np.float64)
        print(f"  {len(iv_panel)} days x {iv_panel.shape[1]} stocks | dtype={iv_panel.dtypes.iloc[0]}")
    else:
        print("\nConnecting to WRDS...")
        db = wrds.Connection(wrds_username="thomasjemerick")
        print("\n[Step 1] Discovering liquid universe...")
        secids = get_liquid_secids(db)
        print(f"\n[Step 2] Pulling ATM IV for {len(secids)} stocks...")
        raw_iv   = pull_iv_batched(db, secids)
        iv_panel = build_iv_panel(raw_iv)
        print("\n[Step 2b] Building ticker map...")
        ticker_map = get_ticker_map(db, iv_panel.columns.tolist())
        db.close()
        with open(PANEL_CACHE, "wb") as f:
            pickle.dump({"panel": iv_panel, "ticker_map": ticker_map}, f)
        print(f"  Cached to {PANEL_CACHE}")

    print(f"\n[Step 3] Clustering...")
    cluster_labels  = cluster_stocks(iv_panel)
    candidate_pairs = get_candidate_pairs(cluster_labels)

    panel_columns = [int(c) for c in iv_panel.columns]
    panel_values  = iv_panel.to_numpy(dtype=np.float64)

    print(f"\n[Step 4] Correlation pre-filter...")
    filtered_pairs = corr_prefilter(panel_values, panel_columns, candidate_pairs)

    print(f"\n[Step 5] Cointegration screening...")
    results = screen_pairs(panel_values, panel_columns, filtered_pairs, ticker_map)

    if not results:
        print("\nNo viable pairs found.")
        return

    results_df = pd.DataFrame(results).sort_values(
        ["adf_pval", "ou_halflife"], ascending=[True, True]
    ).reset_index(drop=True)
    results_df["rank"] = results_df.index + 1
    results_df.to_csv(OUTPUT_FILE, index=False)

    elapsed = (time.time() - t_start) / 60
    print("\n" + "=" * 70)
    print("SCREENING COMPLETE")
    print("=" * 70)
    print(f"  Universe:           {iv_panel.shape[1]} stocks")
    print(f"  After clustering:   {len(candidate_pairs):,} pairs")
    print(f"  After corr filter:  {len(filtered_pairs):,} pairs")
    print(f"  Viable pairs:       {len(results_df):,}")
    print(f"  Avg ADF p-val:      {results_df['adf_pval'].mean():.5f}")
    print(f"  Avg OU half-life:   {results_df['ou_halflife'].mean():.1f} days")
    print(f"  Runtime:            {elapsed:.1f} minutes")
    print(f"\n  Saved to: {OUTPUT_FILE}")
    print(f"\n  Top 30 pairs:")
    cols = ["rank","ticker_x","ticker_y","coint_pval","adf_pval","ou_halflife","spread_std"]
    print(results_df[cols].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
