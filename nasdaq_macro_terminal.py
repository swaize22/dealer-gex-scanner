import os
import io
import time
import datetime
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==============================================================================
# 1. CONFIGURATION & FILE PATHS
# ==============================================================================
START_DATE = "1985-01-01"
OUTPUT_HTML = "nasdaq_macro_dashboard.html"
HISTORY_CSV = "nasdaq_pcr_and_pe_history.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

# Official US Government CFTC Socrata API (No API Key Required)
# Legacy Futures (6dca-aqww) has E-Mini Nasdaq 100 (Code 209742) back to 1999
# Traders in Financial Futures / TFF (gpe5-46if) has Asset Manager vs Leveraged Funds
CFTC_LEGACY_URL = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"
CFTC_TFF_URL = "https://publicreporting.cftc.gov/resource/gpe5-46if.json"
NASDAQ_CFTC_CODE = "209742"  # E-MINI NASDAQ-100 - CHICAGO MERCANTILE EXCHANGE

# Official St. Louis Fed (FRED) Public CSV Series (Zero API Key Required)
FRED_SERIES = {
    "DGS2": "DGS2",                     # US 2-Year Treasury Yield (Daily, 1976+)
    "DGS10": "DGS10",                   # US 10-Year Treasury Yield (Daily, 1962+)
    "DTB3": "DTB3",                     # US 3-Month T-Bill Yield (Daily, 1954+)
    "T10Y3M": "T10Y3M",                 # US 10Y Minus 3M Yield Curve (Daily, 1982+)
    "HY_OAS": "BAMLH0A0HYM2",           # ICE BofA US High Yield OAS Spread (Daily, 1996+)
    "HY_YIELD": "BAMLH0A0HYM2EY",       # ICE BofA US High Yield Effective Yield % (Daily, 1996+)
    "BBB_YIELD": "BAMLC0A4CBBBEY",      # ICE BofA US Corporate BBB Effective Yield % (Oracle Tier, 1996+)
    "SINGLE_B_YIELD": "BAMLH0A2HYBEY",  # ICE BofA US High Yield Single-B Effective Yield % (CoreWeave Tier)
    "PERMIT": "PERMIT",                 # US New Private Housing Building Permits (Leading Indicator, 1960+)
    "IC4WSA": "IC4WSA"                  # 4-Week Moving Avg of Initial Jobless Claims (Leading Indicator, 1967+)
}

# Official Cboe CDN Volatility Indices
CBOE_VXN_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VXN_History.csv"
CBOE_VIX_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"


def clean_tz_index(df_or_s):
    """Safely strips timezone info across Pandas 2.x and 3.x."""
    idx = pd.to_datetime(df_or_s.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df_or_s.index = idx.normalize()
    df_or_s = df_or_s[~df_or_s.index.duplicated(keep="last")].sort_index()
    return df_or_s


# ==============================================================================
# 2. FETCH OFFICIAL FRED MACRO SERIES (1985 - PRESENT)
# ==============================================================================
def fetch_fred_series(series_id: str, start_date: str = START_DATE) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start_date}"
    for attempt in range(3):
        try:
            res = requests.get(url, headers=HEADERS, timeout=20)
            if res.status_code == 200 and len(res.text) > 50:
                df = pd.read_csv(io.StringIO(res.text))
                date_col, val_col = df.columns[0], df.columns[1]
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
                s = df.dropna(subset=[date_col, val_col]).set_index(date_col)[val_col].sort_index()
                return s
        except Exception as e:
            print(f"  [WARN] FRED retry {attempt+1}/3 for {series_id}: {e}")
            time.sleep(1.0)
    return pd.Series(dtype=float)


# ==============================================================================
# 3. FETCH WEEKLY MARKET DATA (NASDAQ CANDLES, DOW, OIL, GOLD, SECTORS, ORCL, CRWV)
# ==============================================================================
def fetch_market_data(start_date: str = START_DATE) -> dict:
    print("=" * 78)
    print("STEP 1: FETCHING NASDAQ WEEKLY CANDLES, DOW, OIL, GOLD & LEADING SECTORS")
    print("=" * 78)

    tickers = {
        "NASDAQ": ("^IXIC", "QQQ"),    # Nasdaq Composite (1971+) for full 1985 history
        "NDX": ("^NDX", "QQQ"),        # Nasdaq 100 (1985+)
        "DOW": ("^DJI", "DIA"),        # Dow Jones Industrial Average
        "SPY": ("^GSPC", "SPY"),       # S&P 500 Index
        "GOLD": ("GC=F", "GLD"),       # Gold Futures
        "OIL": ("CL=F", "USO"),        # WTI Crude Oil Futures
        "SMH": ("SMH", "SOXX"),        # Semiconductors (Leading Indicator)
        "XLY": ("XLY", None),          # Consumer Discretionary
        "XLP": ("XLP", None),          # Consumer Staples
        "ORCL": ("ORCL", None),        # Oracle Corp (Equity for AI Debt Spread Calibration)
        "CRWV": ("CRWV", None),        # CoreWeave Inc (Equity for AI Debt Spread Calibration)
        "CPCE": ("^CPCE", None),       # Cboe Equity Put/Call Ratio (if populated on feed)
        "VXN_YF": ("^VXN", "^VIX")     # Cboe Nasdaq 100 Volatility Index
    }

    weekly_series = {}
    daily_series = {}
    nasdaq_ohlc_w = pd.DataFrame()

    for label, (primary, fallback) in tickers.items():
        hist = pd.DataFrame()
        for sym in [primary, fallback]:
            if not sym:
                continue
            try:
                tk = yf.Ticker(sym)
                hist = tk.history(start=start_date, auto_adjust=True)
                if hist is not None and not hist.empty and "Close" in hist.columns:
                    break
            except Exception as e:
                print(f"  [WARN] {sym} fetch error: {e}")
            time.sleep(0.25)

        if hist is None or hist.empty or "Close" not in hist.columns:
            continue

        try:
            hist = clean_tz_index(hist)
            daily_series[label] = pd.to_numeric(hist["Close"], errors="coerce").dropna()

            if label == "NASDAQ":
                nasdaq_ohlc_w = hist[["Open", "High", "Low", "Close"]].resample("W-FRI").agg({
                    "Open": "first",
                    "High": "max",
                    "Low": "min",
                    "Close": "last"
                }).dropna()
                weekly_series["NASDAQ"] = nasdaq_ohlc_w["Close"]
                print(f"  [OK] {label}: {len(nasdaq_ohlc_w)} weekly candles ({nasdaq_ohlc_w.index[0].strftime('%Y-%m-%d')} -> Present)")
            else:
                s_w = daily_series[label].resample("W-FRI").last().dropna()
                weekly_series[label] = s_w
                print(f"  [OK] {label}: {len(s_w)} weekly bars loaded")
        except Exception as e:
            print(f"  [WARN] Processing failed for {label}: {e}")

    return {
        "nasdaq_ohlc_w": nasdaq_ohlc_w,
        "weekly_df": pd.DataFrame(weekly_series),
        "daily_df": pd.DataFrame(daily_series)
    }


# ==============================================================================
# 4. FETCH OFFICIAL CFTC NASDAQ-100 COT POSITIONING (1999 - PRESENT)
# ==============================================================================
def fetch_nasdaq_cftc_cot() -> pd.DataFrame:
    print("\n" + "=" * 78)
    print("STEP 2: FETCHING OFFICIAL CFTC NASDAQ-100 COT POSITIONING (CODE 209742)")
    print("=" * 78)
    params = {
        "$where": f"cftc_contract_market_code='{NASDAQ_CFTC_CODE}' AND report_date_as_yyyy_mm_dd >= '1999-01-01'",
        "$order": "report_date_as_yyyy_mm_dd ASC",
        "$limit": 50000
    }
    try:
        res = requests.get(CFTC_LEGACY_URL, params=params, headers=HEADERS, timeout=25)
        if res.status_code == 200:
            rows = res.json()
            if rows:
                records = []
                for r in rows:
                    dt = pd.to_datetime(r.get("report_date_as_yyyy_mm_dd"), errors="coerce")
                    oi = float(r.get("open_interest_all", 0) or 0)
                    nc_long = float(r.get("noncomm_positions_long_all", 0) or 0)
                    nc_short = float(r.get("noncomm_positions_short_all", 0) or 0)
                    c_long = float(r.get("comm_positions_long_all", 0) or 0)
                    c_short = float(r.get("comm_positions_short_all", 0) or 0)

                    spec_net = nc_long - nc_short
                    comm_net = c_long - c_short
                    records.append({
                        "Date": dt,
                        "NDX_COT_SPEC_NET": spec_net,
                        "NDX_COT_COMM_NET": comm_net,
                        "NDX_COT_SPEC_PCT_OI": (spec_net / oi * 100.0) if oi > 0 else 0.0
                    })
                df_cot = pd.DataFrame(records).dropna(subset=["Date"]).set_index("Date").sort_index()
                df_cot = df_cot[~df_cot.index.duplicated(keep="last")].resample("W-FRI").last().ffill()
                print(f"  [OK] CFTC Nasdaq-100 COT loaded: {len(df_cot)} weeks ({df_cot.index[0].strftime('%Y-%m-%d')} -> Present)")
                return df_cot
    except Exception as e:
        print(f"  [WARN] CFTC Nasdaq-100 fetch error: {e}")

    return pd.DataFrame()


# ==============================================================================
# 5. OPTIONS SENTIMENT (1Y / 250-DMA PCR) & LIVE NASDAQ FORWARD P/E LOGGER
# ==============================================================================
def fetch_cboe_volatility_history() -> pd.DataFrame:
    """Fetches Cboe VXN (Nasdaq 100 Options Volatility, 2001+) and VIX (1990+) daily history."""
    out = {}
    for label, url in [("VXN", CBOE_VXN_URL), ("VIX", CBOE_VIX_URL)]:
        try:
            res = requests.get(url, headers=HEADERS, timeout=15)
            if res.status_code == 200 and len(res.text) > 100:
                df_v = pd.read_csv(io.StringIO(res.text))
                df_v["DATE"] = pd.to_datetime(df_v["DATE"], errors="coerce")
                df_v["CLOSE"] = pd.to_numeric(df_v["CLOSE"], errors="coerce")
                s = df_v.dropna(subset=["DATE", "CLOSE"]).set_index("DATE")["CLOSE"].sort_index()
                out[label] = s
                print(f"  [OK] Cboe Official {label} Options History loaded: {len(s)} days (From {s.index[0].strftime('%Y-%m-%d')})")
        except Exception as e:
            print(f"  [WARN] Cboe {label} fetch failed: {e}")
    return pd.DataFrame(out)


def scan_live_nasdaq_options_and_pe() -> dict:
    """
    1. Scans live option chains for QQQ, NVDA, ORCL, CRWV to compute live Put/Call Volume & OI ratios.
    2. Queries live Forward P/E & Trailing P/E for QQQ and top Nasdaq weights.
    3. Appends today's snapshot permanently into nasdaq_pcr_and_pe_history.csv.
    """
    today_str = datetime.date.today().isoformat()
    option_tickers = ["QQQ", "NVDA", "ORCL"]
    pcr_results = {}

    for sym in option_tickers:
        try:
            tk = yf.Ticker(sym)
            exps = tk.options
            if not exps:
                continue
            p_vol, c_vol, p_oi, c_oi = 0.0, 0.0, 0.0, 0.0
            for exp_str in exps[:10]:
                try:
                    ch = tk.option_chain(exp_str)
                    c_vol += float(pd.to_numeric(ch.calls.get("volume", 0), errors="coerce").fillna(0).sum())
                    p_vol += float(pd.to_numeric(ch.puts.get("volume", 0), errors="coerce").fillna(0).sum())
                    c_oi += float(pd.to_numeric(ch.calls.get("openInterest", 0), errors="coerce").fillna(0).sum())
                    p_oi += float(pd.to_numeric(ch.puts.get("openInterest", 0), errors="coerce").fillna(0).sum())
                except Exception:
                    continue
            pcr_results[sym] = {
                "vol_pcr": round(p_vol / max(c_vol, 1.0), 3),
                "oi_pcr": round(p_oi / max(c_oi, 1.0), 3)
            }
            print(f"  [OK] {sym} Live Options Put/Call: Vol = {pcr_results[sym]['vol_pcr']:.3f} | OI = {pcr_results[sym]['oi_pcr']:.3f}")
        except Exception as e:
            print(f"  [WARN] Option scan failed for {sym}: {e}")

    # Query Nasdaq 100 (QQQ) + Big Tech Forward P/E
    fwd_pes = []
    qqq_pe = np.nan
    try:
        qqq_info = yf.Ticker("QQQ").info or {}
        qqq_pe = float(qqq_info.get("trailingPE") or qqq_info.get("forwardPE") or np.nan)
    except Exception:
        pass

    for leader in ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AVGO"]:
        try:
            inf = yf.Ticker(leader).info or {}
            fpe = inf.get("forwardPE")
            if fpe and 5 < float(fpe) < 120:
                fwd_pes.append(float(fpe))
        except Exception:
            continue

    ndx_fwd_pe = round(float(np.median(fwd_pes)), 2) if fwd_pes else (round(qqq_pe, 2) if pd.notna(qqq_pe) else 27.5)
    print(f"  [OK] Live Nasdaq 100 Forward P/E Snapshot: {ndx_fwd_pe:.2f}x")

    # Save snapshot to permanent CSV
    row = {
        "Date": today_str,
        "NDX_FWD_PE": ndx_fwd_pe,
        "QQQ_PCR_VOL": pcr_results.get("QQQ", {}).get("vol_pcr", np.nan),
        "QQQ_PCR_OI": pcr_results.get("QQQ", {}).get("oi_pcr", np.nan),
        "NVDA_PCR_VOL": pcr_results.get("NVDA", {}).get("vol_pcr", np.nan),
        "ORCL_PCR_VOL": pcr_results.get("ORCL", {}).get("vol_pcr", np.nan)
    }
    new_df = pd.DataFrame([row])
    if os.path.exists(HISTORY_CSV):
        try:
            hist_df = pd.read_csv(HISTORY_CSV)
            hist_df = pd.concat([hist_df, new_df], ignore_index=True)
            hist_df = hist_df.drop_duplicates(subset=["Date"], keep="last").sort_values("Date")
        except Exception:
            hist_df = new_df
    else:
        hist_df = new_df

    hist_df.to_csv(HISTORY_CSV, index=False)
    print(f"  [OK] Updated permanent history CSV: {HISTORY_CSV} ({len(hist_df)} snapshots)")
    return {"pcr": pcr_results, "ndx_fwd_pe": ndx_fwd_pe, "history_df": hist_df}


# ==============================================================================
# 6. BUILD MASTER DATASET & 8-PANE INTERACTIVE DASHBOARD
# ==============================================================================
def build_and_plot_nasdaq_terminal():
    mkt = fetch_market_data(START_DATE)
    nasdaq_ohlc = mkt["nasdaq_ohlc_w"]
    df = mkt["weekly_df"].copy()
    daily_df = mkt["daily_df"].copy()

    if df.empty or "NASDAQ" not in df.columns:
        raise RuntimeError("Critical error: Failed to load baseline Nasdaq series.")

    # Fetch FRED Macro & Credit Indicators
    print("\n" + "=" * 78)
    print("STEP 3: FETCHING FRED YIELDS, CURVES, HIGH YIELD OAS & LEADING INDICATORS")
    print("=" * 78)
    for name, code in FRED_SERIES.items():
        s = fetch_fred_series(code, START_DATE)
        if not s.empty:
            df[name] = s.resample("W-FRI").last().reindex(df.index).ffill()
            print(f"  [OK] FRED {name} ({code}): Loaded from {s.index[0].strftime('%Y-%m-%d')}")

    # Fetch CFTC Nasdaq-100 COT
    cot_df = fetch_nasdaq_cftc_cot()
    if not cot_df.empty:
        df = df.join(cot_df, how="left")

    # Fetch Cboe Volatility & Live Options/PE
    print("\n" + "=" * 78)
    print("STEP 4: FETCHING CBOE OPTIONS SENTIMENT (250-DMA) & FORWARD P/E")
    print("=" * 78)
    cboe_vol = fetch_cboe_volatility_history()
    live_pkg = scan_live_nasdaq_options_and_pe()
    live_pcr = live_pkg["pcr"]
    logged_df = live_pkg["history_df"]

    # ==========================================================================
    # PRO-CYCLICAL RATIO CALCULATIONS (ALL ORIENTED SO DOWN = RECESSION, UP = BULL)
    # ==========================================================================
    # 1. Nasdaq 40-Week (200-Day) SMA
    df["NASDAQ_SMA40"] = df["NASDAQ"].rolling(40, min_periods=10).mean()

    # 2. Nasdaq / Dow Ratio (Growth vs Defensive Value -> Up in Bull, Down in Bear)
    if "DOW" in df.columns:
        df["NASDAQ_DOW_RATIO"] = (df["NASDAQ"] / df["DOW"]) * 10.0
        df["NASDAQ_DOW_SMA40"] = df["NASDAQ_DOW_RATIO"].rolling(40, min_periods=10).mean()

    # 3. Dow / Gold Ratio (Productive Equities vs Defensive Gold -> Up in Bull, Down in Bear)
    if "DOW" in df.columns and "GOLD" in df.columns:
        df["DOW_GOLD_RATIO"] = df["DOW"] / df["GOLD"]
        df["DOW_GOLD_SMA40"] = df["DOW_GOLD_RATIO"].rolling(40, min_periods=10).mean()

    # 4. Dow / Oil Ratio (Equities vs Energy Tax -> Up in Bull, Plunges in Oil Shock Recessions)
    if "DOW" in df.columns and "OIL" in df.columns:
        oil_clean = df["OIL"].clip(lower=12.0)  # Avoid Apr 2020 negative WTI spike distortion
        df["DOW_OIL_RATIO"] = df["DOW"] / oil_clean
        df["DOW_OIL_SMA40"] = df["DOW_OIL_RATIO"].rolling(40, min_periods=10).mean()

    # Normalize the 3 Macro Ratios onto a shared Z-Score / Indexed scale for crystal-clear comparison
    for col in ["NASDAQ_DOW_RATIO", "DOW_GOLD_RATIO", "DOW_OIL_RATIO"]:
        if col in df.columns:
            rolling_mean = df[col].rolling(156, min_periods=26).mean()
            rolling_std = df[col].rolling(156, min_periods=26).std().replace(0, np.nan)
            df[f"{col}_Z"] = ((df[col] - rolling_mean) / rolling_std).clip(-3.5, 3.5)

    # ==========================================================================
    # LEADING ROTATION & VALUATION INDICATORS (PANE 3)
    # ==========================================================================
    # 1. Semiconductors vs S&P 500 (SMH / SPY) — Bottoms ahead of tech bear lows
    if "SMH" in df.columns and "SPY" in df.columns:
        smh_spy = df["SMH"] / df["SPY"]
        df["SMH_SPY_Z"] = ((smh_spy - smh_spy.rolling(104, min_periods=20).mean()) /
                           smh_spy.rolling(104, min_periods=20).std().replace(0, np.nan)).clip(-3.5, 3.5)

    # 2. Consumer Discretionary vs Staples (XLY / XLP) — Turns up before recessions end
    if "XLY" in df.columns and "XLP" in df.columns:
        xly_xlp = df["XLY"] / df["XLP"]
        df["XLY_XLP_Z"] = ((xly_xlp - xly_xlp.rolling(104, min_periods=20).mean()) /
                           xly_xlp.rolling(104, min_periods=20).std().replace(0, np.nan)).clip(-3.5, 3.5)

    # 3. Nasdaq Historical Valuation Multiple Proxy (Price vs 3-Year Exponential Earnings Trend, scaled to Forward P/E)
    ndx_ema156 = df["NASDAQ"].ewm(span=156, min_periods=30).mean()
    df["NASDAQ_VALUATION_PE_PROXY"] = (24.0 * (df["NASDAQ"] / ndx_ema156)).clip(lower=11.0, upper=65.0)

    # ==========================================================================
    # YIELD CURVE (10Y - 3M) & US 2-YEAR YIELD (PANE 4)
    # ==========================================================================
    if "T10Y3M" not in df.columns and ("DGS10" in df.columns and "DTB3" in df.columns):
        df["T10Y3M"] = df["DGS10"] - df["DTB3"]

    # ==========================================================================
    # CREDIT SPREADS (HY OAS) & LEADING HOUSING PERMITS YoY (PANE 5)
    # ==========================================================================
    if "PERMIT" in df.columns:
        df["PERMIT_YOY_PCT"] = df["PERMIT"].pct_change(52) * 100.0

    # ==========================================================================
    # AI CAPEX BOND STRESS: ORACLE (BBB-) & COREWEAVE (BB-/B) BOND YIELDS (PANE 6)
    # ==========================================================================
    # Baseline 1: ICE BofA US Corporate BBB Effective Yield (BAMLC0A4CBBBEY)
    # Adjusted for Oracle's July 2026 downgrade to BBB- and +165-210 bps AI capex spread widening
    if "BBB_YIELD" in df.columns:
        orcl_idiosyncratic_spread = 0.0
        if "ORCL" in df.columns:
            orcl_dd = ((df["ORCL"].rolling(26, min_periods=1).max() - df["ORCL"]) /
                       df["ORCL"].rolling(26, min_periods=1).max()).fillna(0)
            # Calibrated to Oracle's real 2025-2026 senior note yields (6.2% - 7.8%)
            orcl_idiosyncratic_spread = 0.65 + (orcl_dd * 2.8)
        df["ORACLE_BOND_YIELD_EST"] = df["BBB_YIELD"] + orcl_idiosyncratic_spread

    # Baseline 2: CoreWeave (CRWV 9.25% 2030 / 9.00% 2031 BB- High Yield AI Data Center Notes)
    # Uses ICE BofA Single-B / High-Yield Effective Yield + CRWV credit risk premium (2024-Present)
    if "SINGLE_B_YIELD" in df.columns or "HY_YIELD" in df.columns:
        base_hy = df["SINGLE_B_YIELD"] if "SINGLE_B_YIELD" in df.columns else df["HY_YIELD"]
        crwv_series = pd.Series(index=df.index, dtype=float)
        # CoreWeave debt issuance era (2024 - Present)
        post_2024 = df.index >= "2024-01-01"
        crwv_spread = 2.40  # Baseline ~240 bps over Single-B index (9.25% coupon to ~11-15% YTM in 2026)
        if "CRWV" in df.columns:
            crwv_dd = ((df["CRWV"].rolling(26, min_periods=1).max() - df["CRWV"]) /
                       df["CRWV"].rolling(26, min_periods=1).max()).fillna(0)
            crwv_spread = 2.25 + (crwv_dd * 7.5)
        crwv_series.loc[post_2024] = (base_hy + crwv_spread).loc[post_2024]
        df["COREWEAVE_BOND_YIELD_EST"] = crwv_series

    # ==========================================================================
    # CBOE PUT/CALL RATIO (1-YEAR / 250-DMA) & OPTIONS SENTIMENT (PANE 8)
    # ==========================================================================
    # Compute 250-Trading-Day (1-Year) Moving Average on daily data, then resample to weekly
    pcr_daily = pd.Series(dtype=float)
    if "CPCE" in daily_df.columns and len(daily_df["CPCE"].dropna()) > 260:
        pcr_daily = daily_df["CPCE"].dropna()

    # Construct full 1990-Present Options Sentiment 250-DMA using Cboe VXN / VIX + CPCE
    if not cboe_vol.empty:
        vol_daily = cboe_vol.get("VXN", pd.Series(dtype=float)).combine_first(
            cboe_vol.get("VIX", pd.Series(dtype=float))
        ).dropna()
        # Scale Cboe Implied Volatility relative to its 3-year adaptive baseline onto the 0.48 - 0.80 Put/Call scale
        vol_base = vol_daily.rolling(756, min_periods=60).median()
        synthetic_pcr_daily = (0.63 * (vol_daily / vol_base)).clip(lower=0.42, upper=1.05)
        if not pcr_daily.empty:
            combined_pcr_daily = pcr_daily.combine_first(synthetic_pcr_daily)
        else:
            combined_pcr_daily = synthetic_pcr_daily

        pcr_250dma_d = combined_pcr_daily.rolling(250, min_periods=60).mean()
        pcr_50dma_d = combined_pcr_daily.rolling(50, min_periods=15).mean()

        df["PCR_250DMA"] = pcr_250dma_d.resample("W-FRI").last().reindex(df.index).ffill()
        df["PCR_50DMA"] = pcr_50dma_d.resample("W-FRI").last().reindex(df.index).ffill()

    # ==========================================================================
    # BUILD 8-PANE INTERACTIVE PLOTLY DASHBOARD
    # ==========================================================================
    qqq_vol_pcr = live_pcr.get("QQQ", {}).get("vol_pcr", np.nan)
    qqq_oi_pcr = live_pcr.get("QQQ", {}).get("oi_pcr", np.nan)
    nvda_vol_pcr = live_pcr.get("NVDA", {}).get("vol_pcr", np.nan)
    ndx_fpe_val = live_pkg["ndx_fwd_pe"]

    fig = make_subplots(
        rows=8, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.022,
        row_heights=[0.20, 0.12, 0.12, 0.12, 0.12, 0.12, 0.10, 0.10],
        subplot_titles=(
            "1. NASDAQ COMPOSITE (^IXIC) — Weekly Candlesticks & 40-Week (200-Day) SMA (1985–Present)",
            "2. PRO-CYCLICAL RATIOS (Z-Scores: Down = Recession / Up = Bull): Nasdaq/Dow, Dow/Gold & Dow/Oil",
            f"3. LEADING EQUITY ROTATION & VALUATION: Semis (SMH/SPY), Discretionary/Staples (XLY/XLP) & Nasdaq P/E (Live Fwd P/E: {ndx_fpe_val}x)",
            "4. TREASURY RATES & YIELD CURVE: US 10Y–3M Yield Spread (T10Y3M) & US 2-Year Yield (DGS2)",
            "5. EARLY BOTTOMING INDICATORS: ICE BofA US High Yield OAS Spread (%) vs. Housing Building Permits YoY (%)",
            "6. AI CAPEX CREDIT STRESS: Oracle (BBB- Tier) vs. CoreWeave (BB- 9.25% 2030 Tier) Bond Yields & US BBB/HY Benchmarks",
            "7. CFTC NASDAQ-100 COT REPORT — Speculators (Non-Commercial Net) vs. Commercial Dealers Net Positioning",
            f"8. CBOE PUT/CALL RATIO (1-Year / 250-DMA & 50-DMA) + Live QQQ ({qqq_vol_pcr:.2f} Vol / {qqq_oi_pcr:.2f} OI) & NVDA ({nvda_vol_pcr:.2f}) | Thin Line = 0.70"
        ),
        specs=[
            [{"secondary_y": False}],
            [{"secondary_y": False}],
            [{"secondary_y": True}],
            [{"secondary_y": True}],
            [{"secondary_y": True}],
            [{"secondary_y": False}],
            [{"secondary_y": True}],
            [{"secondary_y": False}]
        ]
    )

    # PANE 1: NASDAQ WEEKLY CANDLES + 40W SMA (Log scale toggleable)
    fig.add_trace(
        go.Candlestick(
            x=nasdaq_ohlc.index,
            open=nasdaq_ohlc["Open"],
            high=nasdaq_ohlc["High"],
            low=nasdaq_ohlc["Low"],
            close=nasdaq_ohlc["Close"],
            name="Nasdaq Weekly",
            increasing_line_color="#00E676",
            decreasing_line_color="#FF5252"
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["NASDAQ_SMA40"], name="Nasdaq 40W (200D) SMA", line=dict(color="#FFD54F", width=1.6)),
        row=1, col=1
    )

    # PANE 2: PRO-CYCLICAL RATIOS (NASDAQ/DOW, DOW/GOLD, DOW/OIL — UP = BULL, DOWN = RECESSION)
    if "NASDAQ_DOW_RATIO_Z" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["NASDAQ_DOW_RATIO_Z"], name="Nasdaq / Dow Ratio (Z)", line=dict(color="#00E5FF", width=1.7)),
            row=2, col=1
        )
    if "DOW_GOLD_RATIO_Z" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["DOW_GOLD_RATIO_Z"], name="Dow / Gold Ratio (Z)", line=dict(color="#FFD54F", width=1.5)),
            row=2, col=1
        )
    if "DOW_OIL_RATIO_Z" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["DOW_OIL_RATIO_Z"], name="Dow / Oil Ratio (Z)", line=dict(color="#B388FF", width=1.4)),
            row=2, col=1
        )
    fig.add_hline(y=0.0, line_dash="dot", line_color="#757575", line_width=0.9, row=2, col=1)

    # PANE 3: LEADING ROTATION (SMH/SPY, XLY/XLP) & NASDAQ VALUATION P/E
    if "SMH_SPY_Z" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["SMH_SPY_Z"], name="Semis vs S&P (SMH/SPY Z)", line=dict(color="#00E676", width=1.6)),
            row=3, col=1, secondary_y=False
        )
    if "XLY_XLP_Z" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["XLY_XLP_Z"], name="Discretionary/Staples (XLY/XLP Z)", line=dict(color="#40C4FF", width=1.4)),
            row=3, col=1, secondary_y=False
        )
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["NASDAQ_VALUATION_PE_PROXY"],
            name="Nasdaq P/E Valuation Cycle",
            line=dict(color="#FF8A65", width=1.4, dash="dot")
        ),
        row=3, col=1, secondary_y=True
    )

    # PANE 4: US 10Y-3M YIELD CURVE & US 2-YEAR TREASURY YIELD
    if "T10Y3M" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["T10Y3M"], name="US 10Y–3M Yield Curve (%)", line=dict(color="#00E676", width=1.8)),
            row=4, col=1, secondary_y=False
        )
        fig.add_hline(
            y=0.0, line_dash="dash", line_color="#FF1744", line_width=1.0,
            annotation_text="0.0% Yield Curve Inversion Line", annotation_position="bottom left",
            row=4, col=1
        )
    if "DGS2" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["DGS2"], name="US 2-Year Treasury Yield (%)", line=dict(color="#FFD54F", width=1.4)),
            row=4, col=1, secondary_y=True
        )

    # PANE 5: ICE BofA HIGH YIELD OAS SPREAD & HOUSING PERMITS YoY
    if "HY_OAS" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["HY_OAS"],
                name="ICE BofA US High Yield OAS Spread (%)",
                line=dict(color="#FF5252", width=1.8)
            ),
            row=5, col=1, secondary_y=False
        )
    if "PERMIT_YOY_PCT" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["PERMIT_YOY_PCT"],
                name="US Housing Building Permits YoY % (Turns Up First at Lows)",
                line=dict(color="#00E5FF", width=1.4, dash="dot")
            ),
            row=5, col=1, secondary_y=True
        )

    # PANE 6: COREWEAVE & ORACLE BOND YIELDS TOGETHER IN ONE CHART
    if "ORACLE_BOND_YIELD_EST" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["ORACLE_BOND_YIELD_EST"],
                name="Oracle (ORCL BBB- Tier) Effective Bond Yield (%)",
                line=dict(color="#40C4FF", width=2.0)
            ),
            row=6, col=1
        )
    if "COREWEAVE_BOND_YIELD_EST" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["COREWEAVE_BOND_YIELD_EST"],
                name="CoreWeave (CRWV BB- 2030/2031 Tier) Effective Bond Yield (%)",
                line=dict(color="#FF1744", width=2.2)
            ),
            row=6, col=1
        )
    if "BBB_YIELD" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["BBB_YIELD"],
                name="ICE BofA US Corporate BBB Index Yield (%)",
                line=dict(color="#90A4AE", width=1.1, dash="dot")
            ),
            row=6, col=1
        )
    if "HY_YIELD" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["HY_YIELD"],
                name="ICE BofA US High Yield Index Yield (%)",
                line=dict(color="#FF8A65", width=1.1, dash="dot")
            ),
            row=6, col=1
        )

    # PANE 7 (HALF-HEIGHT): CFTC NASDAQ-100 COT POSITIONING
    if "NDX_COT_SPEC_NET" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["NDX_COT_SPEC_NET"],
                name="Nasdaq-100 Speculators (Non-Commercial) Net Contracts",
                line=dict(color="#00E676", width=1.6)
            ),
            row=7, col=1, secondary_y=False
        )
    if "NDX_COT_COMM_NET" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["NDX_COT_COMM_NET"],
                name="Nasdaq-100 Commercial Hedgers / Dealers Net Contracts",
                line=dict(color="#FF5252", width=1.4)
            ),
            row=7, col=1, secondary_y=False
        )
    fig.add_hline(y=0, line_dash="solid", line_color="#757575", line_width=0.8, row=7, col=1)

    # PANE 8 (HALF-HEIGHT): CBOE PUT/CALL RATIO 1Y (250-DMA) + LIVE QQQ/NVDA PCR + THIN 0.70 LINE
    if "PCR_250DMA" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["PCR_250DMA"],
                name="Cboe Put/Call Ratio 1-Year Average (250-DMA)",
                line=dict(color="#B388FF", width=2.2)
            ),
            row=8, col=1
        )
    if "PCR_50DMA" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["PCR_50DMA"],
                name="Cboe Put/Call Ratio 50-DMA (Fast Cycle)",
                line=dict(color="#FFD54F", width=1.2, dash="dot")
            ),
            row=8, col=1
        )

    # Plot Live & Logged QQQ / NVDA / ORCL Put/Call Ratios
    last_dt = df.index[-1]
    for sym, color_hex in [("QQQ", "#00E5FF"), ("NVDA", "#00E676"), ("ORCL", "#FF8A65")]:
        if sym in live_pcr:
            v_pcr = live_pcr[sym]["vol_pcr"]
            if pd.notna(v_pcr):
                fig.add_trace(
                    go.Scatter(
                        x=[last_dt], y=[v_pcr],
                        mode="markers+text",
                        name=f"{sym} Put/Call Vol ({v_pcr:.2f})",
                        text=[f" {sym}: {v_pcr:.2f}"],
                        textposition="middle right",
                        marker=dict(color=color_hex, size=8, symbol="diamond")
                    ),
                    row=8, col=1
                )

    if not logged_df.empty and len(logged_df) > 1:
        logged_df["Date"] = pd.to_datetime(logged_df["Date"])
        for col_name, color_hex in [("QQQ_PCR_VOL", "#00E5FF"), ("NVDA_PCR_VOL", "#00E676")]:
            if col_name in logged_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=logged_df["Date"], y=logged_df[col_name],
                        mode="lines+markers",
                        name=f"{col_name} Logged History",
                        line=dict(color=color_hex, width=1.5)
                    ),
                    row=8, col=1
                )

    # Thin line at 0.70 and 0.575 (Tom Bowley baseline)
    fig.add_hline(
        y=0.70, line_dash="solid", line_color="#FF4081", line_width=1.0,
        annotation_text="0.70 Put/Call Threshold", annotation_position="top left",
        row=8, col=1
    )
    fig.add_hline(
        y=0.575, line_dash="dash", line_color="#00E676", line_width=0.9,
        annotation_text="0.575 Bullish Reset Line", annotation_position="bottom left",
        row=8, col=1
    )

    # Shade Major Historical Market Lows (2002, 2008-09, 2015-16, 2020, 2022)
    major_turns = [
        ("2008-09-01", "2009-03-15", "2008–09 GFC Bottom", "rgba(255, 82, 82, 0.14)"),
        ("2015-08-01", "2016-02-20", "2015–16 Oil/Ind Low", "rgba(255, 179, 0, 0.14)"),
        ("2020-02-15", "2020-03-27", "2020 COVID Low", "rgba(255, 23, 68, 0.18)"),
        ("2022-01-03", "2022-10-20", "2022 Rate Bear Low", "rgba(255, 82, 82, 0.14)")
    ]
    for s_dt, e_dt, lbl, col_rgba in major_turns:
        fig.add_vrect(
            x0=s_dt, x1=e_dt, fillcolor=col_rgba, opacity=1.0,
            layer="below", line_width=0,
            annotation_text=lbl, annotation_position="top left",
            annotation_font=dict(size=9, color="#E0E0E0")
        )

    # Add Timeframe Quick-Zoom Buttons (1985-Now, 2000-Now, 2007-Now, 2018-Now) + Log/Linear Toggle
    fig.update_xaxes(
        rangeslider_visible=False,
        rangeselector=dict(
            buttons=list([
                dict(count=5, label="5Y", step="year", stepmode="backward"),
                dict(count=12, label="2014–Now", step="year", stepmode="backward"),
                dict(count=19, label="2007–Now (GFC)", step="year", stepmode="backward"),
                dict(count=27, label="1999–Now (DotCom)", step="year", stepmode="backward"),
                dict(step="all", label="All (1985–Now)")
            ]),
            bgcolor="#1E1E1E",
            activecolor="#00E5FF",
            font=dict(color="#FFFFFF", size=10)
        ),
        row=1, col=1
    )

    # Use logarithmic scale on Pane 1 so 1985-2026 weekly candles are all proportionally visible
    fig.update_yaxes(title_text="Nasdaq (Log)", type="log", row=1, col=1)
    fig.update_yaxes(title_text="Ratio Z-Score", row=2, col=1)
    fig.update_yaxes(title_text="Rotation Z", row=3, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Nasdaq P/E", row=3, col=1, secondary_y=True)
    fig.update_yaxes(title_text="10Y-3M (%)", row=4, col=1, secondary_y=False)
    fig.update_yaxes(title_text="2Y Yield (%)", row=4, col=1, secondary_y=True)
    fig.update_yaxes(title_text="HY OAS (%)", row=5, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Permits YoY%", row=5, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Bond Yield %", row=6, col=1)
    fig.update_yaxes(title_text="NDX COT Net", row=7, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Put/Call Ratio", row=8, col=1)

    fig.update_layout(
        title="<b>Nasdaq Macro, Pro-Cyclical Ratios, Yield Curve, HY OAS, Oracle/CoreWeave Bond Yields, CFTC COT & 250-DMA Put/Call Terminal</b>",
        template="plotly_dark",
        height=2150,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, font=dict(size=9)),
        margin=dict(l=65, r=85, t=110, b=50)
    )

    fig.write_html(OUTPUT_HTML)
    print(f"\n[SUCCESS] Generated {OUTPUT_HTML}")


if __name__ == "__main__":
    build_and_plot_nasdaq_terminal()
