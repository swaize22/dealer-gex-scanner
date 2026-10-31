import os
import time
import datetime
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

# ==============================================================================
# 1. CONFIGURATION: WATCHLIST & ACTIVE SQUEEZE TRADES
# ==============================================================================
# Universe of stocks to scan for NEW Short Squeeze Entry Setups
SQUEEZE_WATCHLIST = [
    "CVNA", "UPST", "AI", "MARA", "RIOT", "RIVN", "LCID", "BYND",
    "GME", "AMC", "SOFI", "PLTR", "ENVX", "IONQ", "RKLB", "ACHR",
    "LITE", "FORM", "ALAB", "TER", "MU", "RKT", "UWMC"

    # =========================================================================
    # 1. CORE BENCHMARKS, BITCOIN PROXY & MOMENTUM LEADERS
    # =========================================================================
    "SPY", "QQQ", "IWM", "IBIT", 
    "NVDA", "PLTR", "MU", "LITE", 
    "FORM", "ALAB", "TER", 
    "LMT", "RTX", "LHX",

    # =========================================================================
    # 2. HIGH SHORT-INTEREST / SQUEEZE FUEL UNIVERSE (High SI % Float + Liquid Options)
    # =========================================================================
    "CVNA", "UPST", "GME", "AMC", "AI", "MARA", "RIOT", "RIVN", "LCID", 
    "BYND", "SOFI", "ENVX", "IONQ", "RKLB", "ACHR", "SMR", "EOSE", 
    "HTZ", "FLNC", "QXO", "DUOL", "PCT", "CLSK", "RUN", "ENPH", "SEDG",

    # =========================================================================
    # 3. KOYFIN SCREENER: STRUCTURAL DECAY & HIGH-DEBT REFINANCING CANDIDATES
    # (Office REITs, Non-Bank Lenders, 5G Telecom, Levered Consumer)
    # =========================================================================
    "UWMC", "RKT", "DEI", "KRC", "ARE", "CUZ", "HIW", "DOC", "KRG", 
    "CUBE", "AMT", "CCI", "IRM", "SPG", "KMI", "BCE", "MGM", "D", 
    "ES", "GPK", "EXC", "ENB", "KIM", "FTS", "AVAH", "WSC", "INVH", 
    "LNT", "SGRY", "BLDR", "WELL", "RNW", "COLD", "CZR", "MAC", "WY", 
    "CAG", "KDP", "SLGN", "VMRK", "VNO", "BXP", "MPW", "WHR", "DVA"
]

# Put tickers you CURRENTLY OWN here -> Activates the 3-Stage Exit Radar!
ACTIVE_TRADES = [
    "CVNA", "UPST"
]

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
SEC_HEADERS = {"User-Agent": "ArgonQuantResearch contact@argonquant.org"}

# Lazy-loaded SEC CIK cache so startup never crashes
_CIK_CACHE = None

def get_cik_map():
    global _CIK_CACHE
    if _CIK_CACHE is not None:
        return _CIK_CACHE
    try:
        res = requests.get("https://www.sec.gov/files/company_tickers.json", headers=SEC_HEADERS, timeout=10)
        if res.status_code == 200:
            data = res.json()
            _CIK_CACHE = {item["ticker"].upper(): str(item["cik_str"]).zfill(10) for item in data.values()}
            return _CIK_CACHE
    except Exception as e:
        print(f"[WARN] SEC CIK map fetch failed: {e}")
    _CIK_CACHE = {}
    return _CIK_CACHE

def send_telegram(message: str):
    """Delivers HTML-formatted Telegram alerts with automatic chunking & error handling."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print(f"\n[LOG ONLY - NO TELEGRAM CREDENTIALS]\n{message}")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    # Telegram max message length is 4096 chars
    chunks = [message[i:i+3900] for i in range(0, len(message), 3900)]
    for chunk in chunks:
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": chunk,
            "parse_mode": "HTML",
            "disable_web_page_preview": True
        }
        try:
            res = requests.post(url, json=payload, timeout=15)
            if res.status_code != 200:
                print(f"[ERROR] Telegram API ({res.status_code}): {res.text}")
        except Exception as e:
            print(f"[ERROR] Telegram request failed: {e}")

# ==============================================================================
# 2. SEC DILUTION & FUNDAMENTAL RISK ENGINE
# ==============================================================================
def audit_sec_and_fundamental_risks(ticker: str, info: dict):
    flags = []
    fatal_risk = False

    # Check 1: SEC EDGAR Dilution Filings (S-3, 424B5 ATM offerings, S-1)
    cik_map = get_cik_map()
    cik = cik_map.get(ticker.upper())
    if cik:
        try:
            url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            res = requests.get(url, headers=SEC_HEADERS, timeout=10)
            if res.status_code == 200:
                recent = res.json().get("filings", {}).get("recent", {})
                forms = recent.get("form", [])
                dates = recent.get("filingDate", [])
                cutoff = (datetime.date.today() - datetime.timedelta(days=75)).isoformat()
                dilution_forms = {"S-3", "S-3/A", "S-1", "S-1/A", "424B5", "424B3"}
                found = [f"{f} ({d})" for f, d in zip(forms, dates) if d >= cutoff and f in dilution_forms]
                if found:
                    flags.append(f"⚠️ <b>SEC Dilution Filing:</b> {', '.join(found[:2])}")
                    fatal_risk = True
        except Exception:
            pass

    # Check 2: Cash Runway / Burn Cliff
    total_cash = float(info.get("totalCash", 0) or 0)
    op_cashflow = float(info.get("operatingCashflow", 0) or 0)
    if op_cashflow < 0 and total_cash > 0:
        runway_months = (total_cash / abs(op_cashflow)) * 12.0
        if runway_months < 6.0:
            flags.append(f"⚠️ <b>Cash Cliff:</b> ~{runway_months:.1f} mos cash remaining (High offering risk)")
            fatal_risk = True

    # Check 3: Underlying Bid-Ask Spread Slippage
    bid = float(info.get("bid", 0) or 0)
    ask = float(info.get("ask", 0) or 0)
    if bid > 0 and ask > 0:
        spread_pct = ((ask - bid) / ask) * 100.0
        if spread_pct > 2.0:
            flags.append(f"⚠️ <b>Wide Spread:</b> {spread_pct:.2f}% slippage risk")

    if fatal_risk or len(flags) >= 2:
        risk_dot = "🔴 [HIGH SPOIL RISK / DILUTION HAZARD]"
    elif len(flags) == 1:
        risk_dot = "🟡 [MODERATE RISK / SIZE SMALL]"
    else:
        risk_dot = "🟢 [LOW RISK / CLEAN RUNWAY]"

    return {"risk_dot": risk_dot, "flags": flags, "fatal": fatal_risk}

# ==============================================================================
# 3. OPTIONS GAMMA RAMP, TECHNICALS & 3-STAGE EXIT ENGINE
# ==============================================================================
def compute_rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return float(rsi.iloc[-1]) if not rsi.empty and pd.notna(rsi.iloc[-1]) else 50.0

def analyze_stock(ticker: str):
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period="3mo")
        if hist is None or len(hist) < 22:
            return None

        info = tk.info or {}
        close_s = hist["Close"].dropna()
        high_s = hist["High"].dropna()
        low_s = hist["Low"].dropna()
        vol_s = hist["Volume"].dropna()

        spot = float(close_s.iloc[-1])
        prev_close = float(close_s.iloc[-2])
        prev_low = float(low_s.iloc[-2])
        today_high = float(high_s.iloc[-1])
        today_open = float(hist["Open"].dropna().iloc[-1])
        high_5d = float(high_s.iloc[-5:].max())

        sma_20 = float(close_s.rolling(20).mean().iloc[-1])
        ema_5 = float(close_s.ewm(span=5, adjust=False).mean().iloc[-1])
        rsi_14 = compute_rsi(close_s, 14)

        avg_vol_20 = float(vol_s.iloc[-21:-1].mean()) if len(vol_s) > 21 else 1.0
        today_vol = float(vol_s.iloc[-1])
        rvol = today_vol / max(avg_vol_20, 1.0)

        # Short Interest Metrics
        float_shares = float(info.get("floatShares", 0) or 0)
        short_pct = float(info.get("shortPercentOfFloat", 0) or 0)
        if short_pct > 1.0:  # Normalize if returned as 25.4 instead of 0.254
            short_pct = short_pct / 100.0
        dtc = float(info.get("shortRatio", 0) or 0)

        # Options Chain Analysis (Front 2 Expirations for Gamma Squeeze Mechanics)
        expirations = tk.options
        if not expirations:
            return None

        today = datetime.date.today()
        otm_call_vol = 0.0
        otm_call_oi = 0.0
        total_call_vol = 0.0
        total_put_vol = 0.0
        total_call_oi = 0.0
        call_wall_strike = spot * 1.10
        max_call_oi_strike = 0.0
        nearest_dte = 30

        for idx, exp_str in enumerate(expirations[:2]):
            exp_date = datetime.datetime.strptime(exp_str, "%Y-%m-%d").date()
            dte = max((exp_date - today).days, 1)
            if idx == 0:
                nearest_dte = dte

            chain = tk.option_chain(exp_str)
            if chain is None or chain.calls.empty:
                continue

            calls = chain.calls.copy()
            puts = chain.puts.copy()

            calls["vol"] = pd.to_numeric(calls.get("volume", 0), errors="coerce").fillna(0)
            calls["oi"] = pd.to_numeric(calls.get("openInterest", 0), errors="coerce").fillna(0)
            puts["vol"] = pd.to_numeric(puts.get("volume", 0), errors="coerce").fillna(0)

            total_call_vol += float(calls["vol"].sum())
            total_put_vol += float(puts["vol"].sum())
            total_call_oi += float(calls["oi"].sum())

            # OTM Gamma Ramp Zone: Strikes between 1.00x and 1.25x Spot
            otm_mask = (calls["strike"] >= spot) & (calls["strike"] <= spot * 1.25)
            otm_calls = calls[otm_mask]
            otm_call_vol += float(otm_calls["vol"].sum())
            otm_call_oi += float(otm_calls["oi"].sum())

            # Identify Absolute Call Wall (Highest Call OI at or above Spot)
            above_spot = calls[calls["strike"] >= spot * 0.98]
            if not above_spot.empty:
                top_row = above_spot.loc[above_spot["oi"].idxmax()]
                if float(top_row["oi"]) > max_call_oi_strike:
                    max_call_oi_strike = float(top_row["oi"])
                    call_wall_strike = float(top_row["strike"])

        # Minimum Option Liquidity Filter (Avoid noise on dead option chains)
        if total_call_oi < 1500:
            return None

        cp_vol_ratio = total_call_vol / max(total_put_vol, 1.0)
        otm_vol_to_oi = otm_call_vol / max(otm_call_oi, 1.0)

        # ======================================================================
        # ENTRY SCORING ENGINE (0 - 100 Points)
        # ======================================================================
        score = 0
        # 1. Fuel (Max 45 pts)
        if short_pct >= 0.25: score += 20
        elif short_pct >= 0.15: score += 12
        elif short_pct >= 0.10: score += 6

        if dtc >= 6.0: score += 15
        elif dtc >= 3.5: score += 10
        elif dtc >= 2.0: score += 5

        if 0 < float_shares <= 50_000_000: score += 10
        elif 0 < float_shares <= 120_000_000: score += 5

        # 2. Spark: OTM Options Gamma Ramp & Volume Velocity (Max 40 pts)
        if cp_vol_ratio >= 3.0 and otm_call_vol >= 2000: score += 20
        elif cp_vol_ratio >= 2.0 and otm_call_vol >= 800: score += 12
        elif cp_vol_ratio >= 1.5: score += 6

        if rvol >= 2.5: score += 20
        elif rvol >= 1.5: score += 12
        elif rvol >= 1.1: score += 5

        # 3. Technical Trend Structure (Max 15 pts)
        if spot > sma_20 and spot > ema_5: score += 15
        elif spot > sma_20: score += 8

        # ======================================================================
        # 3-STAGE EXIT RADAR EVALUATION (For Active Squeeze Positions)
        # ======================================================================
        exit_stage = 0
        exit_badge = ""
        exit_reasons = []
        exit_action = ""

        # Upper wick exhaustion calculation
        upper_wick_pct = ((today_high - spot) / spot) * 100.0 if today_high > spot else 0.0
        dist_from_5d_high = ((spot - high_5d) / high_5d) * 100.0
        ext_above_ema5 = ((spot - ema_5) / ema_5) * 100.0
        near_call_wall = spot >= (call_wall_strike * 0.985)
        daily_return_pct = ((spot - prev_close) / prev_close) * 100.0

        # Check Stage 3 First (Highest Urgency: Collapse / Structural Breakdown)
        stage3_triggers = []
        if spot < ema_5 and spot < prev_low:
            stage3_triggers.append(f"Snapped 5-Day EMA (${ema_5:.2f}) & broke prior day's low (${prev_low:.2f})")
        if dist_from_5d_high <= -8.0 and cp_vol_ratio < 0.90:
            stage3_triggers.append(f"Dropped {dist_from_5d_high:.1f}% off 5D peak with Put volume overtaking Calls (Dealer Delta Unwind)")
        if rvol >= 2.0 and daily_return_pct <= -6.0 and spot < today_open:
            stage3_triggers.append(f"High-volume distribution flush ({daily_return_pct:.1f}% on {rvol:.1f}x RVOL)")

        # Check Stage 2 (Time to Get Out: Call Wall Pinning / Climax Churn / OpEx Charm)
        stage2_triggers = []
        if near_call_wall and daily_return_pct < 1.5:
            stage2_triggers.append(f"Pinned at Call Wall (${call_wall_strike:.2f}) — Dealer gamma buying exhausted")
        if rvol >= 3.0 and abs(daily_return_pct) <= 1.0:
            stage2_triggers.append(f"Heavy volume churn ({rvol:.1f}x RVOL with flat {daily_return_pct:.1f}% close — institutional distribution)")
        if nearest_dte <= 3 and ext_above_ema5 > 5.0:
            stage2_triggers.append(f"OpEx Charm Cliff ({nearest_dte} DTE) — Expiring OTM calls will force dealer selling")

        # Check Stage 1 (Risk Rising: Parabolic Extension / Upper Wick Rejection)
        stage1_triggers = []
        if upper_wick_pct >= 4.5:
            stage1_triggers.append(f"Severe intraday upper-wick rejection (-{upper_wick_pct:.1f}% off today's high of ${today_high:.2f})")
        if rsi_14 >= 78.0 or ext_above_ema5 >= 18.0:
            stage1_triggers.append(f"Parabolic overextension (RSI: {rsi_14:.0f} | +{ext_above_ema5:.1f}% above 5-EMA)")
        if otm_vol_to_oi < 0.15 and rvol < 1.0 and spot > sma_20 * 1.15:
            stage1_triggers.append("Fresh OTM call volume drying up while elevated — gamma momentum stalling")

        if stage3_triggers:
            exit_stage = 3
            exit_badge = "🔴 STAGE 3 EXIT: COLLAPSE / GAMMA UNWIND"
            exit_reasons = stage3_triggers
            exit_action = "🚨 <b>LIQUIDATE 100% IMMEDIATELY.</b> Reverse gamma unwind underway; dealers are dumping hedged shares."
        elif stage2_triggers:
            exit_stage = 2
            exit_badge = "🟠 STAGE 2 EXIT: TIME TO GET OUT (PEAK ZONE)"
            exit_reasons = stage2_triggers
            exit_action = "🎯 <b>TAKE PROFIT ON 80%–100%.</b> Squeeze mechanics are peaking at structural resistance / Charm decay."
        elif stage1_triggers:
            exit_stage = 1
            exit_badge = "🟡 STAGE 1 EXIT: RISK RISING (EARLY WARNING)"
            exit_reasons = stage1_triggers
            exit_action = "⚠️ <b>TRIM 30%–50% & RAISE STOP</b> to ${:.2f} (5-Day EMA). Momentum is showing exhaustion.".format(ema_5)

        return {
            "ticker": ticker,
            "spot": spot,
            "score": score,
            "short_pct": round(short_pct * 100, 1),
            "dtc": round(dtc, 1),
            "rvol": round(rvol, 2),
            "cp_ratio": round(cp_vol_ratio, 2),
            "otm_call_vol": int(otm_call_vol),
            "otm_call_oi": int(otm_call_oi),
            "call_wall": call_wall_strike,
            "nearest_dte": nearest_dte,
            "sma_20": sma_20,
            "ema_5": ema_5,
            "rsi_14": round(rsi_14, 1),
            "info": info,
            "exit_stage": exit_stage,
            "exit_badge": exit_badge,
            "exit_reasons": exit_reasons,
            "exit_action": exit_action
        }
    except Exception as e:
        print(f"[WARN] Failed analyzing {ticker}: {e}")
        return None

# ==============================================================================
# 4. FORMATTERS FOR 3-DOT ENTRY & 3-STAGE EXIT TELEGRAM ALERTS
# ==============================================================================
def format_entry_alert(d: dict, risk: dict) -> str:
    # 1. Setup Quality Dot
    if d["score"] >= 80:
        setup_dot = "🟢 [AMAZING CONFLUENCE]"
    elif d["score"] >= 65:
        setup_dot = "🟡 [DEVELOPING / MEDIUM]"
    else:
        setup_dot = "🔴 [EARLY / WATCHLIST ONLY]"

    # 2. Timing Horizon Dot
    if d["rvol"] >= 1.8 and d["spot"] > d["sma_20"] and d["nearest_dte"] <= 10:
        horizon_dot = "🟢 [IMMINENT: < 7 DAYS]"
    elif d["dtc"] <= 7.0 or d["nearest_dte"] <= 21:
        horizon_dot = "🟡 [DEVELOPING: ~14 DAYS]"
    else:
        horizon_dot = "🔴 [DISTANT: 20+ DAYS OUT]"

    risk_dot = risk["risk_dot"]
    hazards = "\n".join([f"  └ {h}" for h in risk["flags"]]) if risk["flags"] else "  └ Clean: No active S-3/ATM dilution or cash cliffs."

    return (
        f"🚀 <b>SQUEEZE ENTRY RADAR: ${d['ticker']}</b> (${d['spot']:.2f})\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🎯 <b>SETUP:</b>    {setup_dot} ({d['score']}/100)\n"
        f"⏳ <b>HORIZON:</b>  {horizon_dot}\n"
        f"🛡️ <b>RISK:</b>     {risk_dot}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>SQUEEZE & GAMMA METRICS:</b>\n"
        f"• Short Float: <b>{d['short_pct']}%</b> | Days to Cover: <b>{d['dtc']}d</b>\n"
        f"• Relative Volume: <b>{d['rvol']}x</b> | C/P Vol Ratio: <b>{d['cp_ratio']}x</b>\n"
        f"• OTM Call Ramp: <b>{d['otm_call_vol']:,} vol</b> vs <b>{d['otm_call_oi']:,} OI</b>\n"
        f"• Target Call Wall: <b>${d['call_wall']:.2f}</b> | Support (20d SMA): <b>${d['sma_20']:.2f}</b>\n\n"
        f"🛡️ <b>RISK AUDIT:</b>\n{hazards}"
    )

def format_exit_alert(d: dict) -> str:
    reasons_str = "\n".join([f"  • {r}" for r in d["exit_reasons"]])
    return (
        f"🔔 <b>ACTIVE TRADE EXIT RADAR: ${d['ticker']}</b> (${d['spot']:.2f})\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>{d['exit_badge']}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🔎 <b>QUANTITATIVE TRIGGERS DETECTED:</b>\n"
        f"{reasons_str}\n\n"
        f"📉 <b>KEY LEVELS:</b>\n"
        f"• 5-Day EMA (Trail Stop): <b>${d['ema_5']:.2f}</b>\n"
        f"• Call Wall Ceiling: <b>${d['call_wall']:.2f}</b> | RSI(14): <b>{d['rsi_14']}</b>\n\n"
        f"💡 <b>ACTION:</b> {d['exit_action']}"
    )

def run_pipeline():
    all_tickers = list(dict.fromkeys(ACTIVE_TRADES + SQUEEZE_WATCHLIST))
    entry_alerts = []
    exit_alerts = []

    for sym in all_tickers:
        data = analyze_stock(sym)
        time.sleep(0.35)  # Prevents Yahoo Finance HTTP 429 Rate Limiting
        if not data:
            continue

        # 1. Check 3-Stage Exit Radar for Active Trades
        if sym in ACTIVE_TRADES and data["exit_stage"] > 0:
            exit_alerts.append(format_exit_alert(data))

        # 2. Check Squeeze Entry Setups (Score >= 65)
        if sym in SQUEEZE_WATCHLIST and data["score"] >= 65:
            risk_info = audit_sec_and_fundamental_risks(sym, data["info"])
            entry_alerts.append(format_entry_alert(data, risk_info))

    messages = []
    if exit_alerts:
        messages.append("🚨 <b>3-STAGE EXIT ALERTS (ACTIVE TRADES)</b> 🚨\n\n" + "\n\n───────────────\n\n".join(exit_alerts))
    if entry_alerts:
        messages.append("🔥 <b>NEW SHORT SQUEEZE SETUPS DETECTED</b> 🔥\n\n" + "\n\n───────────────\n\n".join(entry_alerts))

    if messages:
        send_telegram("\n\n═══════════════════════\n\n".join(messages))
    else:
        heartbeat = (
            f"✅ <b>Argon Squeeze & Exit Radar Online</b>\n"
            f"• <b>Watchlist Scanned:</b> {len(all_tickers)} symbols\n"
            f"• <b>Active Trades Monitored:</b> {', '.join(ACTIVE_TRADES) if ACTIVE_TRADES else 'None'}\n"
            f"• <b>Status:</b> No new 65+ squeeze setups or exit triggers today."
        )
        send_telegram(heartbeat)

if __name__ == "__main__":
    run_pipeline()
