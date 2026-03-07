"""
Pre-2014 IV Pairs Screener — Walk-Forward Universe Selection
=============================================================
Screens pairs using ONLY 2005-2013 data (training period).
This closes the pair-selection leakage issue entirely.
Pairs identified here are then backtested on 2014-2023 (truly unseen).

Runtime: ~15-20 minutes using fast OLS+ADF cointegration.
"""

import os
os.environ["WRDS_USERNAME"] = "thomasjemerick"
os.environ["PGPASSWORD"] = ""   # <-- replace this

import numpy as np
import pandas as pd
import wrds
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import warnings
import time
import pickle
warnings.filterwarnings("ignore")

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

SCREEN_END    = "2013-12-31"    # only use data up to here for pair selection
PANEL_CACHE   = "iv_panel_cache_2005.pkl"

COINT_PVAL    = 0.05
ADF_PVAL      = 0.05
MIN_HL        = 10
MAX_HL        = 120
MIN_OBS       = 500             # ~2 years minimum in training window
MIN_CORR      = 0.70

OUTPUT_FILE   = "pre2014_pairs.csv"

ETF_BLACKLIST = {
    "TECL","TECS","UPRO","SPXU","SOXL","SOXS","TNA","TZA","UDOW","SDOW",
    "LABU","LABD","NUGT","DUST","JNUG","JDST","NAIL","DRN","DRV","CURE",
    "DFEN","WANT","HIBL","HIBS","BULZ","BERZ","FNGU","FNGD","QID","QLD",
    "SSO","SDS","SH","PSQ","DOG","SKF","UYG","FAZ","FAS","ERX","ERY",
    "EDC","EDZ","ROM","REW","SAA","SDD","UWM","TWM","MVV","MZZ",
    "DDM","DXD","UVXY","SVXY","VXX","VIXY","VIXM","TVIX","ZIV",
    "TMF","TMV","TBT","TLT","TTT","URTY","SRTY","UMDD","SMDD",
    "MIDU","MIDZ","INDL","EFO","EFU","EZJ","DPK","LBJ","EPV",
    "RUI","NDX","DJX","SPX","VIX",
}


# ── FAST COINTEGRATION ────────────────────────────────────────────────────────

def fast_coint(y, x):
    b1 = np.polyfit(x, y, 1)
    p_yx = adfuller(y - b1[0]*x - b1[1], maxlag=1, autolag=None)[1]
    b2 = np.polyfit(y, x, 1)
    p_xy = adfuller(x - b2[0]*y - b2[1], maxlag=1, autolag=None)[1]
    if p_yx <= p_xy:
        return max(p_yx, p_xy), b1[1], b1[0]
    else:
        return max(p_yx, p_xy), b2[1], b2[0]


def ou_half_life(spread):
    diff = np.diff(spread)
    lag  = spread[:-1]
    lam  = OLS(diff, add_constant(lag)).fit().params[1]
    if lam >= 0:
        return np.nan
    return -np.log(2) / lam


# ── SIC SECTOR MAP ────────────────────────────────────────────────────────────

def sic_to_sector(sic):
    if pd.isna(sic):
        return None
    sic = int(sic)
    if sic == 0: return None
    if   100   <= sic <= 999:   return "Agriculture"
    elif 1000  <= sic <= 1499:  return "Mining"
    elif 1500  <= sic <= 1799:  return "Construction"
    elif 2000  <= sic <= 3999:  return "Manufacturing"
    elif 4000  <= sic <= 4999:  return "Transport/Utilities"
    elif 5000  <= sic <= 5199:  return "Wholesale"
    elif 5200  <= sic <= 5999:  return "Retail"
    elif 6000  <= sic <= 6799:  return "Finance"
    elif 7000  <= sic <= 8999:  return "Services"
    elif 9000  <= sic <= 9999:  return "Government"
    return None


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    print("=" * 70)
    print("Pre-2014 IV Pairs Screener — Walk-Forward Universe Selection")
    print(f"  Screening window: 2005 - {SCREEN_END}")
    print(f"  Pairs found here are tested on 2014-2023 (truly unseen)")
    print("=" * 70)

    # ── Load full panel from cache ──
    print("\nLoading cached IV panel...")
    with open(PANEL_CACHE, "rb") as f:
        cache = pickle.load(f)
    iv_panel_full = cache["panel"]
    ticker_map    = cache["ticker_map"]
    sec_info      = cache.get("sec_info", None)

    iv_panel_full.columns = [int(c) for c in iv_panel_full.columns]
    ticker_map = {int(k): v for k, v in ticker_map.items() if v is not None}
    iv_panel_full = iv_panel_full.astype(np.float64)

    # ── Slice to pre-2014 only ──
    dates = pd.DatetimeIndex(iv_panel_full.index)
    pre2014_mask = dates <= SCREEN_END
    iv_panel = iv_panel_full[pre2014_mask].copy()
    print(f"  Full panel:    {len(iv_panel_full)} days x {iv_panel_full.shape[1]} stocks")
    print(f"  Pre-2014 slice:{len(iv_panel)} days x {iv_panel.shape[1]} stocks")
    print(f"  Date range:    {iv_panel.index[0].date()} to {iv_panel.index[-1].date()}")

    # ── Build reverse ticker map ──
    rev_map = {v.upper(): int(k) for k, v in ticker_map.items()}

    # ── Get sector info ──
    if sec_info is None:
        print("\nNo sec_info in cache — fetching from WRDS...")
        db = wrds.Connection(wrds_username="thomasjemerick")
        sid_str = ", ".join(str(s) for s in iv_panel.columns.tolist())
        sec_info = db.raw_sql(f"""
            SELECT DISTINCT ON (secid) secid, ticker, sic
            FROM optionm.secnmd
            WHERE secid IN ({sid_str})
            ORDER BY secid, effect_date DESC
        """)
        sec_info["secid"] = sec_info["secid"].astype(int)
        sec_info["sic"]   = pd.to_numeric(sec_info["sic"], errors="coerce")
        sec_info = sec_info.set_index("secid")
        db.close()

    if "sector" not in sec_info.columns:
        sec_info["sector"] = sec_info["sic"].apply(sic_to_sector)

    # ── Filter universe ──
    print("\nFiltering universe...")
    valid = sec_info.loc[sec_info.index.isin(iv_panel.columns)].copy()
    valid = valid[~valid["ticker"].str.upper().isin(ETF_BLACKLIST)]
    valid = valid[valid["sector"].notna()]
    print(f"  Stocks after ETF/sector filter: {len(valid)}")

    # ── Drop stocks with < MIN_OBS observations in pre-2014 window ──
    panel_sub = iv_panel[valid.index.tolist()]
    obs_counts = panel_sub.notna().sum()
    valid = valid[obs_counts[valid.index] >= MIN_OBS]
    print(f"  Stocks with >= {MIN_OBS} pre-2014 observations: {len(valid)}")
    print(f"  Sector breakdown:")
    print(valid["sector"].value_counts().to_string())

    # ── Generate intra-sector pairs ──
    from itertools import combinations
    pairs = []
    for sector, group in valid.groupby("sector"):
        members = group.index.tolist()
        if len(members) >= 2:
            pairs.extend(combinations(members, 2))
    print(f"\n  Total intra-sector candidate pairs: {len(pairs):,}")

    # ── Extract numpy arrays for pre-2014 panel ──
    panel_cols   = [int(c) for c in panel_sub.columns]
    panel_values = panel_sub[valid.index.tolist()].reindex(
        columns=valid.index.tolist()
    ).to_numpy(dtype=np.float64)
    col_idx = {int(c): i for i, c in enumerate(valid.index.tolist())}

    # ── Screen pairs ──
    print(f"\nScreening {len(pairs):,} pairs on pre-2014 data only...")
    results = []
    t0 = time.time()

    for i, (sx, sy) in enumerate(pairs):
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
            x = x[mask]; y = y[mask]
            if len(x) < MIN_OBS:
                continue

            # Correlation pre-filter
            if abs(np.corrcoef(x, y)[0, 1]) < MIN_CORR:
                continue

            # Fast cointegration
            p, intercept, slope = fast_coint(y, x)
            if p >= COINT_PVAL:
                continue

            spread = y - intercept - slope * x
            adf_p  = adfuller(spread, maxlag=1, autolag=None)[1]
            if adf_p >= ADF_PVAL:
                continue

            hl = ou_half_life(spread)
            if np.isnan(hl) or not (MIN_HL < hl < MAX_HL):
                continue

            tx = ticker_map.get(sx, str(sx))
            ty = ticker_map.get(sy, str(sy))
            sector = valid.loc[sx, "sector"] if sx in valid.index else ""

            results.append({
                "secid_x":     sx,
                "secid_y":     sy,
                "ticker_x":    tx,
                "ticker_y":    ty,
                "sector":      sector,
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
            elapsed = time.time() - t0
            rate    = (i + 1) / elapsed
            eta     = (len(pairs) - i - 1) / rate / 60
            print(f"  {i+1:,}/{len(pairs):,} | {len(results):,} viable | "
                  f"{rate:.0f} pairs/sec | ETA {eta:.1f} min")

    # ── Score and rank ──
    if not results:
        print("No viable pairs found.")
        return

    df = pd.DataFrame(results)
    df["adf_pval_safe"] = df["adf_pval"].replace(0, 1e-10)
    df["score"] = (
        -np.log(df["adf_pval_safe"]) *
        (1.0 / df["ou_halflife"]) *
        (1.0 / (df["spread_std"] + 0.001))
    )
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    df.to_csv(OUTPUT_FILE, index=False)

    elapsed = (time.time() - t_start) / 60
    print("\n" + "=" * 70)
    print("SCREENING COMPLETE")
    print("=" * 70)
    print(f"  Screening window:   2005 - {SCREEN_END}")
    print(f"  Universe:           {len(valid)} stocks")
    print(f"  Pairs tested:       {len(pairs):,}")
    print(f"  Viable pairs:       {len(df):,}")
    print(f"  Runtime:            {elapsed:.1f} minutes")
    print(f"\n  Saved to: {OUTPUT_FILE}")
    print(f"\n  Top 30 pairs (identified using pre-2014 data only):")
    cols = ["rank","ticker_x","ticker_y","sector",
            "coint_pval","adf_pval","ou_halflife","spread_std","score"]
    print(df[cols].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
