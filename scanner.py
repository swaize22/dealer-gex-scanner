import os
import math
import requests
import numpy as np
import pandas as pd
from scipy.stats import norm
import yfinance as yf
from datetime import datetime, date, timedelta

# ==========================================
# 1. CONFIGURATION & WATCHLIST
# ==========================================
WATCHLIST = [
    "SPY", "QQQ", "IWM", "IBIT", 
    "NVDA", "PLTR", "MU", "LITE", 
    "FORM", "ALAB", "TER", 
    "LMT", "RTX", "LHX", "GOOG"
]

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram(message: str):
    """Sends a robust HTML-formatted message to Telegram."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print(f"[NO TELEGRAM CONFIGURED - LOGGING ONLY]:\n{message}")
        return
    
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }
    try:
        res = requests.post(url, json=payload, timeout=15)
        if res.status_code != 200:
            print(f"Telegram API Error ({res.status_code}): {res.text}")
    except Exception as e:
        print(f"Failed to deliver Telegram message: {e}")

def get_third_friday(year: int, month: int) -> date:
    """Finds the 3rd Friday (monthly OpEx) for a given year and month."""
    first_day = date(year, month, 1)
    first_friday_offset = (4 - first_day.weekday()) % 7
    first_friday = first_day + timedelta(days=first_friday_offset)
    return first_friday + timedelta(weeks=2)

def is_opex_drift_window():
    """True if today is 7 to 14 calendar days prior to monthly OpEx."""
    today = date.today()
    opex = get_third_friday(today.year, today.month)
    if today > opex:
        next_month = today.month + 1 if today.month < 12 else 1
        next_year = today.year if today.month < 12 else today.year + 1
        opex = get_third_friday(next_year, next_month)
    days_to_opex = (opex - today).days
    return (7 <= days_to_opex <= 14), days_to_opex, opex

def bs_gamma(S, K, T, sigma, r=0.04):
    """Calculates Black-Scholes gamma defensively."""
    try:
        if T <= 0.0001 or sigma <= 0.001 or S <= 0 or K <= 0:
            return 0.0
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return float(norm.pdf(d1) / (S * sigma * np.sqrt(T)))
    except Exception:
        return 0.0

def calculate_gex_and_levels(ticker_symbol: str):
    """Computes Net GEX, Call Wall, Put Wall, and Zero Gamma."""
    try:
        tk = yf.Ticker(ticker_symbol)
        hist = tk.history(period="5d")
        if hist is None or hist.empty:
            return None
            
        spot = float(hist['Close'].dropna().iloc[-1])
        expirations = tk.options
        if not expirations:
            return None
            
        records = []
        today = datetime.now().date()
        
        # Scan front 4 expirations
        for exp_str in expirations[:4]:
            try:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                dte = (exp_date - today).days
                if dte <= 0:
                    continue
                T = max(dte / 365.0, 4.0 / (365.0 * 24.0))
                
                chain = tk.option_chain(exp_str)
                if chain is None:
                    continue

                # Process calls
                if hasattr(chain, 'calls') and not chain.calls.empty:
                    calls = chain.calls.copy()
                    calls['oi'] = pd.to_numeric(calls.get('openInterest', 0), errors='coerce').fillna(0)
                    calls['iv'] = pd.to_numeric(calls.get('impliedVolatility', 0), errors='coerce').fillna(0)
                    
                    for _, row in calls.iterrows():
                        k = float(row['strike'])
                        oi = float(row['oi'])
                        iv = float(row['iv'])
                        if oi > 0 and iv > 0.01 and (0.6 * spot <= k <= 1.4 * spot):
                            gamma = bs_gamma(spot, k, T, iv)
                            gex = gamma * oi * 100.0 * (spot ** 2) * 0.01
                            records.append({'strike': k, 'type': 'call', 'oi': oi, 'gex': gex})

                # Process puts
                if hasattr(chain, 'puts') and not chain.puts.empty:
                    puts = chain.puts.copy()
                    puts['oi'] = pd.to_numeric(puts.get('openInterest', 0), errors='coerce').fillna(0)
                    puts['iv'] = pd.to_numeric(puts.get('impliedVolatility', 0), errors='coerce').fillna(0)
                    
                    for _, row in puts.iterrows():
                        k = float(row['strike'])
                        oi = float(row['oi'])
                        iv = float(row['iv'])
                        if oi > 0 and iv > 0.01 and (0.6 * spot <= k <= 1.4 * spot):
                            gamma = bs_gamma(spot, k, T, iv)
                            gex = -1.0 * (gamma * oi * 100.0 * (spot ** 2) * 0.01)
                            records.append({'strike': k, 'type': 'put', 'oi': oi, 'gex': gex})

            except Exception as e:
                print(f"Error processing expiration {exp_str} for {ticker_symbol}: {e}")
                continue

        if not records:
            return None

        df = pd.DataFrame(records)
        net_gex = float(df['gex'].sum())

        per_strike = df.groupby('strike')['gex'].sum().sort_index()
        call_df = df[df['type'] == 'call'].groupby('strike')['oi'].sum()
        put_df = df[df['type'] == 'put'].groupby('strike')['oi'].sum()

        call_wall = float(call_df.idxmax()) if not call_df.empty else float(per_strike.idxmax())
        put_wall = float(put_df.idxmax()) if not put_df.empty else float(per_strike.idxmin())

        # Gamma Flip estimation
        cum_gex = per_strike.cumsum()
        zero_crossing = (cum_gex.shift(1) * cum_gex < 0)
        flip_candidates = per_strike.index[zero_crossing]
        gamma_flip = float(flip_candidates[0]) if len(flip_candidates) > 0 else (spot * 0.98)

        return {
            'spot': spot,
            'net_gex': net_gex,
            'call_wall': call_wall,
            'put_wall': put_wall,
            'gamma_flip': gamma_flip
        }
    except Exception as e:
        print(f"Failed analyzing {ticker_symbol}: {e}")
        return None

def run():
    in_window, days_to_opex, opex_date = is_opex_drift_window()

    # VIX 5-Day Trend
    vix_cooling = False
    try:
        vix = yf.Ticker("^VIX")
        vix_hist = vix.history(period="6d")
        if vix_hist is not None and len(vix_hist) >= 5:
            vix_cooling = float(vix_hist['Close'].iloc[-1]) < float(vix_hist['Close'].iloc[0])
    except Exception as e:
        print(f"VIX calculation error: {e}")

    signals = []

    for symbol in WATCHLIST:
        data = calculate_gex_and_levels(symbol)
        if not data:
            continue

        spot = data['spot']
        flip = data['gamma_flip']
        call_w = data['call_wall']
        net_gex = data['net_gex']

        is_positive_regime = (spot > flip) and (net_gex > 0)

        # 1. Hard Invalidation / Stop warning
        if spot <= flip:
            signals.append(
                f"🚨 <b>REGIME WARNING: {symbol}</b>\n"
                f"• Spot (${spot:.2f}) closed <b>below Gamma Flip</b> (${flip:.2f})\n"
                f"• Negative gamma regime. Downside risk elevated."
            )
            continue

        # 2. Take Profit Alert (within 0.5% of Call Wall)
        if spot >= (call_w * 0.995):
            signals.append(
                f"🎯 <b>TARGET REACHED: {symbol}</b>\n"
                f"• Spot (${spot:.2f}) is at/near Call Wall (${call_w:.2f})\n"
                f"• Lock in swing profits ahead of pinning flow."
            )
            continue

        # 3. High Probability Entry Setup
        if in_window and is_positive_regime and vix_cooling:
            upside_to_wall = ((call_w - spot) / spot) * 100.0
            if upside_to_wall >= 1.5:
                signals.append(
                    f"🟢 <b>HIGH PROB SETUP: {symbol}</b>\n"
                    f"• <b>Days to OpEx:</b> {days_to_opex} days ({opex_date})\n"
                    f"• <b>Spot:</b> ${spot:.2f}\n"
                    f"• <b>Regime:</b> Positive GEX (+${net_gex/1e6:.1f}M)\n"
                    f"• <b>Gamma Flip (Stop):</b> ${flip:.2f}\n"
                    f"• <b>Call Wall (Target):</b> ${call_w:.2f} (+{upside_to_wall:.1f}%)\n"
                    f"• <b>Flow:</b> Vanna/Charm tailwind active."
                )

    if signals:
        full_msg = "\n\n━━━━━━━━━━━━━━━\n\n".join(signals)
        send_telegram(f"🏖️ <b>Options Quantitative Radar</b>\n\n{full_msg}")
    else:
        # Diagnostic heartbeat message so you know your setup is healthy
        status_msg = (
            f"✅ <b>Scanner Active & Online</b>\n\n"
            f"• <b>Days to Next OpEx:</b> {days_to_opex} days ({opex_date})\n"
            f"• <b>In Pre-OpEx Window (7-14d):</b> {'Yes' if in_window else 'No'}\n"
            f"• <b>VIX Cooling:</b> {'Yes' if vix_cooling else 'No'}\n"
            f"• <b>Tickers Monitored:</b> {', '.join(WATCHLIST)}\n\n"
            f"<i>No new setups triggered today. Standing by for OpEx window.</i>"
        )
        send_telegram(status_msg)
        print(status_msg)

if __name__ == "__main__":
    run()
