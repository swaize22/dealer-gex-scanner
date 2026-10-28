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
# To track new tickers, simply add them here!
# ==========================================
WATCHLIST = ["SPY", "QQQ", "NVDA", "TSLA", "AAPL"]

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram(message: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print(f"[LOG ONLY]\n{message}")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    requests.post(url, json=payload, timeout=15)

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
        # Check next month's OpEx if current has passed
        next_month = today.month + 1 if today.month < 12 else 1
        next_year = today.year if today.month < 12 else today.year + 1
        opex = get_third_friday(next_year, next_month)
    days_to_opex = (opex - today).days
    return (7 <= days_to_opex <= 14), days_to_opex, opex

def bs_gamma(S, K, T, sigma, r=0.04):
    """Calculates Black-Scholes gamma."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

def calculate_gex_and_levels(ticker_symbol: str):
    tk = yf.Ticker(ticker_symbol)
    hist = tk.history(period="5d")
    if hist.empty:
        return None
    spot = hist['Close'].iloc[-1]
    
    expirations = tk.options
    if not expirations:
        return None
        
    records = []
    today = datetime.now().date()
    
    # Process options up to 45 days out (captures front-month gamma concentration)
    for exp_str in expirations[:4]:
        exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
        dte = (exp_date - today).days
        if dte <= 0:
            continue
        T = max(dte / 365.0, 4 / (365 * 24))
        
        try:
            chain = tk.option_chain(exp_str)
        except Exception:
            continue
            
        # Calls: Dealer GEX is positive (+1)
        for _, row in chain.calls.iterrows():
            k = row['strike']
            oi = row.get('openInterest', 0)
            iv = row.get('impliedVolatility', 0)
            if oi and oi > 0 and iv and iv > 0 and (0.6 * spot <= k <= 1.4 * spot):
                gamma = bs_gamma(spot, k, T, iv)
                # Dollar GEX per 1% move
                gex = gamma * oi * 100 * (spot ** 2) * 0.01
                records.append({'strike': k, 'type': 'call', 'oi': oi, 'gex': gex})
                
        # Puts: Dealer GEX is negative (-1)
        for _, row in chain.puts.iterrows():
            k = row['strike']
            oi = row.get('openInterest', 0)
            iv = row.get('impliedVolatility', 0)
            if oi and oi > 0 and iv and iv > 0 and (0.6 * spot <= k <= 1.4 * spot):
                gamma = bs_gamma(spot, k, T, iv)
                gex = -1.0 * (gamma * oi * 100 * (spot ** 2) * 0.01)
                records.append({'strike': k, 'type': 'put', 'oi': oi, 'gex': gex})
                
    if not records:
        return None
        
    df = pd.DataFrame(records)
    net_gex = df['gex'].sum()
    
    # Strike aggregation for walls
    per_strike = df.groupby('strike')['gex'].sum().sort_index()
    call_df = df[df['type'] == 'call'].groupby('strike')['oi'].sum()
    put_df = df[df['type'] == 'put'].groupby('strike')['oi'].sum()
    
    call_wall = call_df.idxmax() if not call_df.empty else per_strike.idxmax()
    put_wall = put_df.idxmax() if not put_df.empty else per_strike.idxmin()
    
    # Gamma Flip (Zero Gamma approximation where cumulative net GEX shifts sign)
    cum_gex = per_strike.cumsum()
    zero_crossing = (cum_gex.shift(1) * cum_gex < 0)
    flip_candidates = per_strike.index[zero_crossing]
    gamma_flip = flip_candidates[0] if len(flip_candidates) > 0 else spot * 0.98

    return {
        'spot': spot,
        'net_gex': net_gex,
        'call_wall': call_wall,
        'put_wall': put_wall,
        'gamma_flip': gamma_flip
    }

def run():
    in_window, days_to_opex, opex_date = is_opex_drift_window()
    
    # Check VIX compression (5-day return < 0)
    vix = yf.Ticker("^VIX")
    vix_hist = vix.history(period="6d")
    vix_cooling = False
    if len(vix_hist) >= 5:
        vix_cooling = (vix_hist['Close'].iloc[-1] < vix_hist['Close'].iloc[0])
    
    signals = []
    
    for symbol in WATCHLIST:
        data = calculate_gex_and_levels(symbol)
        if not data:
            continue
            
        spot = data['spot']
        flip = data['gamma_flip']
        call_w = data['call_wall']
        put_w = data['put_wall']
        net_gex = data['net_gex']
        
        # 1. Positive Regime Filter
        is_positive_regime = (spot > flip) and (net_gex > 0)
        
        # 2. Hard Invalidation Warning (If already in a trade and flip is breached)
        if spot <= flip:
            signals.append(
                f"🚨 *REGIME WARNING: {symbol}*\n"
                f"• Spot (${spot:.2f}) closed *below Gamma Flip* (${flip:.2f})\n"
                f"• Dealers flipped negative. Downside risk elevated. Cut swing longs."
            )
            continue
            
        # 3. Take Profit Alert (Near Call Wall)
        if spot >= (call_w * 0.995):
            signals.append(
                f"🎯 *TARGET HIT: {symbol}*\n"
                f"• Spot (${spot:.2f}) is within 0.5% of Call Wall (${call_w:.2f})\n"
                f"• Lock in swing profits ahead of pinning flow."
            )
            continue
            
        # 4. Entry Compound Setup (Window + Pos GEX + VIX Cooling)
        if in_window and is_positive_regime and vix_cooling:
            upside_to_wall = ((call_w - spot) / spot) * 100
            if upside_to_wall >= 1.5:  # Only if there's enough room to trade
                signals.append(
                    f"🟢 *HIGH PROB SETUP: {symbol}*\n"
                    f"• *Days to OpEx:* {days_to_opex} days ({opex_date})\n"
                    f"• *Spot Price:* ${spot:.2f}\n"
                    f"• *Regime:* Positive GEX (+${net_gex/1e6:.1f}M)\n"
                    f"• *Gamma Flip (Stop):* ${flip:.2f}\n"
                    f"• *Call Wall (Target):* ${call_w:.2f} (+{upside_to_wall:.1f}%)\n"
                    f"• *Flow:* Vanna/Charm tailwind active."
                )

   if signals:
        full_msg = "\n\n━━━━━━━━━━━━━━━\n\n".join(signals)
        send_telegram(f"🏖️ *Options Quantitative Radar*\n\n{full_msg}")
    else:
        # Diagnostic ping so you know it's working while testing
        status_msg = (
            f"✅ *Scanner Active & Online*\n"
            f"• *Days to Next OpEx:* {days_to_opex} days ({opex_date})\n"
            f"• *In Pre-OpEx Window (7-14d):* {'Yes' if in_window else 'No'}\n"
            f"• *VIX 5D Cooling:* {'Yes' if vix_cooling else 'No'}\n"
            f"• *Monitored Tickers:* {', '.join(WATCHLIST)}\n"
            f"• *Status:* Standing by until the 7–14 day window opens."
        )
        send_telegram(status_msg)
        print(status_msg)
