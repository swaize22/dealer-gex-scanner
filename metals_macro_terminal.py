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
OUTPUT_HTML = "precious_metals_cot_options_dashboard.html"
PCR_LOG_CSV = "metals_etf_pcr_history.csv"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

CBOE_GVZ_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/GVZ_History.csv"
CFTC_DISAGG_URL = "https://publicreporting.cftc.gov/resource/72hh-3qpy.json"
CFTC_LEGACY_URL = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"

CFTC_CODES = {
    "GOLD": "088691",
    "SILVER": "084691",
    "COPPER": "085692"
}


def clean_tz_index(df_or_s):
    """Safely strips timezone info across both Pandas 2.x and Pandas 3.x."""
    idx = pd.to_datetime(df_or_s.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df_or_s.index = idx.normalize()
    df_or_s = df_or_s[~df_or_s.index.duplicated(keep="last")].sort_index()
    return df_or_s


# ==============================================================================
# 2. FETCH WEEKLY MARKET DATA (GOLD CANDLES, SILVER, COPPER, ETFs)
# ==============================================================================
def fetch_weekly_prices(start_date: str = "2006-06-01") -> dict:
    print("=" * 75)
    print("STEP 1: FETCHING GOLD (WEEKLY CANDLES), SILVER, COPPER & ETFs")
    print("=" * 75)

    tickers = {
        "GOLD": ("GC=F", "GLD"),      # Primary: Gold Futures, Fallback: GLD
        "SILVER": ("SI=F", "SLV"),    # Primary: Silver Futures, Fallback: SLV
        "COPPER": ("HG=F", "CPER"),   # Primary: Copper Futures, Fallback: CPER
        "GLD": ("GLD", None),
        "SLV": ("SLV", None),
        "GDX": ("GDX", None),
        "CPCE": ("^CPCE", None),
        "GVZ_YF": ("^GVZ", None)
    }

    weekly_data = {}
    gold_ohlc_w = pd.DataFrame()

    for label, (sym, fallback_sym) in tickers.items():
        hist = pd.DataFrame()
        for candidate in [sym, fallback_sym]:
            if not candidate:
                continue
            try:
                tk = yf.Ticker(candidate)
                hist = tk.history(start=start_date, auto_adjust=True)
                if hist is not None and not hist.empty and "Close" in hist.columns:
                    break
            except Exception as e:
                print(f"  [WARN] {candidate} attempt failed: {e}")
            time.sleep(0.3)

        if hist is None or hist.empty or "Close" not in hist.columns:
            print(f"  [SKIP] No historical price series returned for {label}")
            continue

        try:
            hist = clean_tz_index(hist)
            if label == "GOLD":
                gold_ohlc_w = hist[["Open", "High", "Low", "Close"]].resample("W-FRI").agg({
                    "Open": "first",
                    "High": "max",
                    "Low": "min",
                    "Close": "last"
                }).dropna()
                weekly_data["GOLD"] = gold_ohlc_w["Close"]
                print(f"  [OK] {label}: {len(gold_ohlc_w)} weekly candles loaded")
            else:
                s_w = pd.to_numeric(hist["Close"], errors="coerce").resample("W-FRI").last().dropna()
                weekly_data[label] = s_w
                print(f"  [OK] {label}: {len(s_w)} weekly points loaded")
        except Exception as e:
            print(f"  [WARN] Resample failed for {label}: {e}")

    return {"gold_ohlc": gold_ohlc_w, "series": pd.DataFrame(weekly_data)}


# ==============================================================================
# 3. FETCH OFFICIAL CFTC COMMITMENTS OF TRADERS (COT) HISTORY
# ==============================================================================
def fetch_cftc_cot_history(contract_code: str = "088691", metal_name: str = "GOLD") -> pd.DataFrame:
    print(f"  -> Querying CFTC Socrata API for {metal_name} (Contract {contract_code})...")
    params = {
        "$where": f"cftc_contract_market_code='{contract_code}' AND report_date_as_yyyy_mm_dd >= '2006-06-01'",
        "$order": "report_date_as_yyyy_mm_dd ASC",
        "$limit": 50000
    }
    try:
        res = requests.get(CFTC_DISAGG_URL, params=params, headers=HEADERS, timeout=25)
        if res.status_code == 200:
            rows = res.json()
            if rows:
                records = []
                for r in rows:
                    dt = pd.to_datetime(r.get("report_date_as_yyyy_mm_dd"), errors="coerce")
                    oi = float(r.get("open_interest_all", 0) or 0)
                    mm_long = float(r.get("m_money_positions_long_all", 0) or 0)
                    mm_short = float(r.get("m_money_positions_short_all", 0) or 0)
                    pm_long = float(r.get("prod_merc_positions_long_all", 0) or 0)
                    pm_short = float(r.get("prod_merc_positions_short_all", 0) or 0)
                    swap_long = float(r.get("swap_positions_long_all", 0) or 0)
                    swap_short = float(r.get("swap__positions_short_all", r.get("swap_positions_short_all", 0)) or 0)

                    mm_net = mm_long - mm_short
                    comm_net = (pm_long + swap_long) - (pm_short + swap_short)
                    records.append({
                        "Date": dt,
                        f"{metal_name}_OI": oi,
                        f"{metal_name}_MM_NET": mm_net,
                        f"{metal_name}_COMM_NET": comm_net,
                        f"{metal_name}_MM_PCT_OI": (mm_net / oi * 100.0) if oi > 0 else 0.0
                    })
                df_cot = pd.DataFrame(records).dropna(subset=["Date"]).set_index("Date").sort_index()
                df_cot = df_cot[~df_cot.index.duplicated(keep="last")]
                df_cot_w = df_cot.resample("W-FRI").last().ffill()
                print(f"  [OK] CFTC {metal_name} COT loaded: {len(df_cot_w)} weeks")
                return df_cot_w
    except Exception as e:
        print(f"  [WARN] Disaggregated CFTC fetch error for {metal_name}: {e}")

    try:
        res_leg = requests.get(CFTC_LEGACY_URL, params=params, headers=HEADERS, timeout=25)
        if res_leg.status_code == 200:
            rows = res_leg.json()
            if rows:
                records = []
                for r in rows:
                    dt = pd.to_datetime(r.get("report_date_as_yyyy_mm_dd"), errors="coerce")
                    oi = float(r.get("open_interest_all", 0) or 0)
                    nc_long = float(r.get("noncomm_positions_long_all", 0) or 0)
                    nc_short = float(r.get("noncomm_positions_short_all", 0) or 0)
                    c_long = float(r.get("comm_positions_long_all", 0) or 0)
                    c_short = float(r.get("comm_positions_short_all", 0) or 0)
                    records.append({
                        "Date": dt,
                        f"{metal_name}_OI": oi,
                        f"{metal_name}_MM_NET": nc_long - nc_short,
                        f"{metal_name}_COMM_NET": c_long - c_short,
                        f"{metal_name}_MM_PCT_OI": ((nc_long - nc_short) / oi * 100.0) if oi > 0 else 0.0
                    })
                df_cot = pd.DataFrame(records).dropna(subset=["Date"]).set_index("Date").sort_index()
                df_cot = df_cot[~df_cot.index.duplicated(keep="last")].resample("W-FRI").last().ffill()
                return df_cot
    except Exception as e:
        print(f"  [WARN] Legacy CFTC fetch failed for {metal_name}: {e}")

    return pd.DataFrame()


# ==============================================================================
# 4. OPTIONS SENTIMENT: CBOE HISTORICAL GVZ + LIVE GLD, SLV, GDX PUT/CALL RATIOS
# ==============================================================================
def fetch_cboe_gvz_history() -> pd.Series:
    try:
        res = requests.get(CBOE_GVZ_URL, headers=HEADERS, timeout=15)
        if res.status_code == 200 and len(res.text) > 100:
            df_gvz = pd.read_csv(io.StringIO(res.text))
            df_gvz["DATE"] = pd.to_datetime(df_gvz["DATE"], errors="coerce")
            df_gvz["CLOSE"] = pd.to_numeric(df_gvz["CLOSE"], errors="coerce")
            s = df_gvz.dropna(subset=["DATE", "CLOSE"]).set_index("DATE")["CLOSE"].sort_index()
            s_w = s.resample("W-FRI").last().dropna()
            print(f"  [OK] Cboe Official GLD Options Volatility (GVZ) loaded: {len(s_w)} weeks")
            return s_w
    except Exception as e:
        print(f"  [WARN] Cboe CDN GVZ fetch failed: {e}")
    return pd.Series(dtype=float)


def compute_etf_live_pcr_and_save_history() -> dict:
    etfs = ["GLD", "SLV", "GDX"]
    live_summary = {}
    today_str = datetime.date.today().isoformat()

    for sym in etfs:
        try:
            tk = yf.Ticker(sym)
            exps = tk.options
            if not exps:
                continue

            tot_put_vol = 0.0
            tot_call_vol = 0.0
            tot_put_oi = 0.0
            tot_call_oi = 0.0

            for exp_str in exps[:10]:
                try:
                    chain = tk.option_chain(exp_str)
                    c_vol = float(pd.to_numeric(chain.calls.get("volume", 0), errors="coerce").fillna(0).sum())
                    p_vol = float(pd.to_numeric(chain.puts.get("volume", 0), errors="coerce").fillna(0).sum())
                    c_oi = float(pd.to_numeric(chain.calls.get("openInterest", 0), errors="coerce").fillna(0).sum())
                    p_oi = float(pd.to_numeric(chain.puts.get("openInterest", 0), errors="coerce").fillna(0).sum())

                    tot_call_vol += c_vol
                    tot_put_vol += p_vol
                    tot_call_oi += c_oi
                    tot_put_oi += p_oi
                except Exception:
                    continue

            vol_pcr = tot_put_vol / max(tot_call_vol, 1.0)
            oi_pcr = tot_put_oi / max(tot_call_oi, 1.0)

            live_summary[sym] = {
                "vol_pcr": round(vol_pcr, 3),
                "oi_pcr": round(oi_pcr, 3)
            }
            print(f"  [OK] {sym} Put/Call Vol = {vol_pcr:.3f} | OI = {oi_pcr:.3f}")
        except Exception as e:
            print(f"  [WARN] Options scan failed for {sym}: {e}")

    # Load existing CSV history (if any) and append today's new row
    row_dict = {"Date": today_str}
    for sym in etfs:
        row_dict[f"{sym}_PCR_VOL"] = live_summary.get(sym, {}).get("vol_pcr", np.nan)
        row_dict[f"{sym}_PCR_OI"] = live_summary.get(sym, {}).get("oi_pcr", np.nan)

    new_row_df = pd.DataFrame([row_dict])
    if os.path.exists(PCR_LOG_CSV):
        try:
            hist_pcr = pd.read_csv(PCR_LOG_CSV)
            hist_pcr = pd.concat([hist_pcr, new_row_df], ignore_index=True)
            hist_pcr = hist_pcr.drop_duplicates(subset=["Date"], keep="last").sort_values("Date")
        except Exception:
            hist_pcr = new_row_df
    else:
        hist_pcr = new_row_df

    hist_pcr.to_csv(PCR_LOG_CSV, index=False)
    print(f"  [OK] Updated {PCR_LOG_CSV} ({len(hist_pcr)} historical snapshots stored)")

    return {"live": live_summary, "logged_history": hist_pcr}


# ==============================================================================
# 5. MASTER DATASET BUILDER & INTERACTIVE CHART GENERATOR
# ==============================================================================
def build_and_plot_dashboard():
    mkt = fetch_weekly_prices(start_date="2006-06-01")
    gold_ohlc = mkt["gold_ohlc"]
    df = mkt["series"].copy()

    if df.empty or "GOLD" not in df.columns:
        raise RuntimeError("Critical error: Unable to fetch baseline Gold price series.")

    print("\n" + "=" * 75)
    print("STEP 2: FETCHING OFFICIAL US GOVT CFTC COT DATA (GOLD & SILVER)")
    print("=" * 75)
    cot_gold = fetch_cftc_cot_history(CFTC_CODES["GOLD"], "GOLD")
    cot_silver = fetch_cftc_cot_history(CFTC_CODES["SILVER"], "SILVER")

    if not cot_gold.empty:
        df = df.join(cot_gold, how="left")
    if not cot_silver.empty:
        df = df.join(cot_silver, how="left")

    print("\n" + "=" * 75)
    print("STEP 3: FETCHING OPTIONS DATA (CBOE GVZ 2008-NOW + GLD/SLV/GDX CHAINS)")
    print("=" * 75)
    gvz_cboe = fetch_cboe_gvz_history()
    if not gvz_cboe.empty:
        df["GVZ"] = gvz_cboe.reindex(df.index).ffill()
    elif "GVZ_YF" in df.columns:
        df["GVZ"] = df["GVZ_YF"].ffill()

    options_pkg = compute_etf_live_pcr_and_save_history()
    live_pcr = options_pkg["live"]
    logged_pcr = options_pkg["logged_history"]

    earliest_options_date = "2008-07-18"
    if "GVZ" in df.columns and not df["GVZ"].dropna().empty:
        earliest_options_date = df["GVZ"].dropna().index[0].strftime("%Y-%m-%d")

    df = df.loc[earliest_options_date:].copy()
    gold_ohlc = gold_ohlc.loc[earliest_options_date:].copy()

    # Ratios & Moving Averages
    if "SILVER" in df.columns:
        df["GOLD_SILVER_RATIO"] = df["GOLD"] / df["SILVER"]
        df["GSR_SMA40"] = df["GOLD_SILVER_RATIO"].rolling(40, min_periods=5).mean()
        df["SILVER_SMA40"] = df["SILVER"].rolling(40, min_periods=5).mean()

    if "COPPER" in df.columns:
        df["COPPER_GOLD_RATIO"] = (df["COPPER"] / df["GOLD"]) * 1000.0
        df["CGR_SMA40"] = df["COPPER_GOLD_RATIO"].rolling(40, min_periods=5).mean()

    df["GOLD_SMA40"] = df["GOLD"].rolling(40, min_periods=5).mean()

    if "GVZ" in df.columns:
        gvz_med = df["GVZ"].rolling(104, min_periods=15).median()
        df["GLD_OPT_SENTIMENT_PROXY"] = (0.70 * (df["GVZ"] / gvz_med)).clip(lower=0.35, upper=1.35)
        df["GLD_OPT_SENTIMENT_SMA10"] = df["GLD_OPT_SENTIMENT_PROXY"].rolling(10, min_periods=3).mean()

    gld_vol_pcr = live_pcr.get("GLD", {}).get("vol_pcr", np.nan)
    gld_oi_pcr = live_pcr.get("GLD", {}).get("oi_pcr", np.nan)
    slv_vol_pcr = live_pcr.get("SLV", {}).get("vol_pcr", np.nan)
    slv_oi_pcr = live_pcr.get("SLV", {}).get("oi_pcr", np.nan)
    gdx_vol_pcr = live_pcr.get("GDX", {}).get("vol_pcr", np.nan)
    gdx_oi_pcr = live_pcr.get("GDX", {}).get("oi_pcr", np.nan)

    pcr_banner = (
        f"Live ETF Put/Call — GLD: {gld_vol_pcr:.2f} Vol / {gld_oi_pcr:.2f} OI | "
        f"SLV: {slv_vol_pcr:.2f} Vol / {slv_oi_pcr:.2f} OI | "
        f"GDX: {gdx_vol_pcr:.2f} Vol / {gdx_oi_pcr:.2f} OI"
        if pd.notna(gld_vol_pcr) else "Options Sentiment & Put/Call Ratio (Threshold = 0.70)"
    )

    fig = make_subplots(
        rows=6, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.028,
        row_heights=[0.26, 0.16, 0.16, 0.16, 0.13, 0.13],
        subplot_titles=(
            "1. GOLD (GC=F) — Weekly Candlesticks & 40-Week (200-Day) SMA",
            "2. SILVER (SI=F) — Weekly Close & 40-Week SMA",
            "3. GOLD / SILVER RATIO (GC ÷ SI) — Monetary Metals Relative Value",
            "4. COPPER / GOLD RATIO (HG ÷ GC × 1000) — Real Tangible Economic Demand vs. Gold",
            "5. CFTC COMMITMENTS OF TRADERS (COT) — Gold & Silver Managed Money (Specs) vs. Commercial Hedgers",
            f"6. OPTION-BASED SENTIMENT (GLD, SLV, GDX Put/Call Ratios + Cboe GLD Options Sentiment) | {pcr_banner}"
        ),
        specs=[
            [{"secondary_y": False}],
            [{"secondary_y": False}],
            [{"secondary_y": False}],
            [{"secondary_y": False}],
            [{"secondary_y": True}],
            [{"secondary_y": False}]
        ]
    )

    # Pane 1: Gold Weekly Candlesticks
    fig.add_trace(
        go.Candlestick(
            x=gold_ohlc.index,
            open=gold_ohlc["Open"],
            high=gold_ohlc["High"],
            low=gold_ohlc["Low"],
            close=gold_ohlc["Close"],
            name="Gold Weekly (GC=F)",
            increasing_line_color="#00E676",
            decreasing_line_color="#FF5252"
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["GOLD_SMA40"], name="Gold 40W SMA", line=dict(color="#FFD54F", width=1.6)),
        row=1, col=1
    )

    # Pane 2: Silver
    if "SILVER" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["SILVER"], name="Silver Weekly", line=dict(color="#E0E0E0", width=1.8)),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["SILVER_SMA40"], name="Silver 40W SMA", line=dict(color="#00B0FF", width=1.4, dash="dot")),
            row=2, col=1
        )

    # Pane 3: Gold/Silver Ratio
    if "GOLD_SILVER_RATIO" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["GOLD_SILVER_RATIO"], name="Gold/Silver Ratio", line=dict(color="#B388FF", width=1.8)),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["GSR_SMA40"], name="GSR 40W SMA", line=dict(color="#FFD54F", width=1.3, dash="dot")),
            row=3, col=1
        )
        fig.add_hline(y=80.0, line_dash="dash", line_color="#00E676", line_width=1, row=3, col=1)
        fig.add_hline(y=55.0, line_dash="dash", line_color="#FF5252", line_width=1, row=3, col=1)

    # Pane 4: Copper/Gold Ratio
    if "COPPER_GOLD_RATIO" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["COPPER_GOLD_RATIO"], name="Copper/Gold (×1000)", line=dict(color="#FF8A65", width=1.8)),
            row=4, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["CGR_SMA40"], name="Cu/Au 40W SMA", line=dict(color="#40C4FF", width=1.4, dash="dot")),
            row=4, col=1
        )

    # Pane 5: CFTC COT
    if "GOLD_MM_NET" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["GOLD_MM_NET"], name="Gold Managed Money Net", line=dict(color="#00E676", width=1.7)),
            row=5, col=1, secondary_y=False
        )
    if "GOLD_COMM_NET" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["GOLD_COMM_NET"], name="Gold Commercial Hedgers Net", line=dict(color="#FF1744", width=1.5)),
            row=5, col=1, secondary_y=False
        )
    if "SILVER_MM_PCT_OI" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["SILVER_MM_PCT_OI"], name="Silver MM Net (% OI)", line=dict(color="#00E5FF", width=1.3, dash="dot")),
            row=5, col=1, secondary_y=True
        )
    fig.add_hline(y=0, line_dash="solid", line_color="#757575", line_width=0.8, row=5, col=1)

    # Pane 6: Option-Based Sentiment + Thin 0.70 Line
    if "GLD_OPT_SENTIMENT_SMA10" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["GLD_OPT_SENTIMENT_SMA10"],
                name="Cboe GLD Options Sentiment Regime (2008–Present)",
                line=dict(color="#FFD54F", width=1.8)
            ),
            row=6, col=1
        )

    etf_colors = {"GLD": "#FFD700", "SLV": "#00E5FF", "GDX": "#00E676"}
    last_dt = df.index[-1]

    for sym, color_hex in etf_colors.items():
        if sym in live_pcr:
            vol_val = live_pcr[sym]["vol_pcr"]
            oi_val = live_pcr[sym]["oi_pcr"]
            if pd.notna(vol_val):
                fig.add_trace(
                    go.Scatter(
                        x=[last_dt], y=[vol_val],
                        mode="markers+text",
                        name=f"{sym} Put/Call Vol ({vol_val:.2f})",
                        text=[f" {sym} Vol: {vol_val:.2f}"],
                        textposition="middle right",
                        marker=dict(color=color_hex, size=9, symbol="diamond")
                    ),
                    row=6, col=1
                )
            if pd.notna(oi_val):
                fig.add_trace(
                    go.Scatter(
                        x=[last_dt], y=[oi_val],
                        mode="markers",
                        name=f"{sym} Put/Call OI ({oi_val:.2f})",
                        marker=dict(color=color_hex, size=7, symbol="circle-open")
                    ),
                    row=6, col=1
                )

    if not logged_pcr.empty and len(logged_pcr) > 1:
        logged_pcr["Date"] = pd.to_datetime(logged_pcr["Date"])
        for sym, color_hex in etf_colors.items():
            col_name = f"{sym}_PCR_VOL"
            if col_name in logged_pcr.columns:
                fig.add_trace(
                    go.Scatter(
                        x=logged_pcr["Date"], y=logged_pcr[col_name],
                        mode="lines+markers",
                        name=f"{sym} Logged Put/Call History",
                        line=dict(color=color_hex, width=1.6)
                    ),
                    row=6, col=1
                )

    # Thin line at 0.70
    fig.add_hline(
        y=0.70,
        line_dash="solid",
        line_color="#FF4081",
        line_width=1.0,
        annotation_text="0.70 Put/Call Line",
        annotation_position="bottom left",
        row=6, col=1
    )

    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
    fig.update_layout(
        title=f"<b>Precious Metals, Ratios, CFTC COT & ETF Put/Call Sentiment ({earliest_options_date} – Present)</b>",
        template="plotly_dark",
        height=1650,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, font=dict(size=10)),
        margin=dict(l=65, r=85, t=95, b=50)
    )

    fig.write_html(OUTPUT_HTML)
    print(f"\n[SUCCESS] Generated {OUTPUT_HTML}")


if __name__ == "__main__":
    build_and_plot_dashboard()
