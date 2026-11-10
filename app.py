import streamlit as st
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Options Earnings Screener",
    layout="wide",
    page_icon="📊",
)

# ── Universe ─────────────────────────────────────────────────────────────

UNIVERSE = [
    "AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","AVGO",
    "COST","NFLX","AMD","ARM","QCOM","INTC","MU","SMCI","TSM","ASML",
    "JPM","GS","MS","BAC","V","MA","PYPL","COIN",
    "UNH","JNJ","LLY","ABBV","MRK","PFE",
    "XOM","CVX","COP",
    "CAT","DE","HON","GE","BA","RTX",
    "WMT","TGT","HD","LOW","NKE","SBUX",
    "UBER","LYFT","DASH","ABNB","BKNG",
    "SNOW","CRM","NOW","WDAY","TEAM",
    "MSTR","RIOT","MARA",
    "PLTR","RBLX","SNAP","PINS","RDDT",
    "F","GM","RIVN","LCID",
    "LMT","NOC","COHR","CAVA","CRWV",
]
UNIVERSE = list(dict.fromkeys(UNIVERSE))

# ── Core math ─────────────────────────────────────────────────────────────

def bs_gamma(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def get_direction(t):
    hist = t.history(period="6mo")["Close"].dropna()
    spot = float(hist.iloc[-1])
    ma20 = float(hist.rolling(20).mean().iloc[-1])
    ma50 = float(hist.rolling(50).mean().iloc[-1])
    if spot > ma20 > ma50:
        direction = "UP"
    elif spot < ma20 < ma50:
        direction = "DOWN"
    else:
        direction = "MIXED"
    hv30 = float(np.log(hist / hist.shift(1)).dropna().rolling(21).std().iloc[-1] * np.sqrt(252))
    return spot, ma20, ma50, direction, hv30


def get_earnings(t, max_days):
    try:
        cal = t.calendar
        if cal and "Earnings Date" in cal:
            dates = cal["Earnings Date"]
            if not isinstance(dates, list):
                dates = [dates]
            for d in dates:
                if hasattr(d, "date"):
                    d = d.date()
                elif isinstance(d, str):
                    d = datetime.strptime(d, "%Y-%m-%d").date()
                days = (d - datetime.today().date()).days
                if 0 <= days <= max_days:
                    return d, days
    except:
        pass
    return None, None


def compute_gex(t, spot, max_days=45):
    r = 0.05
    net_gex = 0.0
    gex_by_strike = {}
    for exp in t.options:
        T = (datetime.strptime(exp, "%Y-%m-%d") - datetime.today()).days
        if T <= 0 or T > max_days:
            continue
        T_yr = T / 365
        try:
            ch = t.option_chain(exp)
        except:
            continue
        for side, sign in [(ch.calls, +1), (ch.puts, -1)]:
            for _, row in side.iterrows():
                K, iv, oi = row["strike"], row["impliedVolatility"], row["openInterest"]
                if np.isnan(iv) or np.isnan(oi) or iv <= 0 or oi <= 0:
                    continue
                g = bs_gamma(spot, K, T_yr, r, iv)
                if np.isnan(g):
                    continue
                c = sign * oi * g * 100 * spot**2 * 0.01
                net_gex += c
                gex_by_strike[K] = gex_by_strike.get(K, 0) + c
    return net_gex, gex_by_strike


def get_chain(t, expiry, spot, direction):
    try:
        chain = t.option_chain(expiry)
    except:
        return pd.DataFrame(), pd.DataFrame(), 0.0

    try:
        atm = chain.calls.iloc[(chain.calls["strike"] - spot).abs().argsort()[:1]]
        atm_iv = float(atm["impliedVolatility"].values[0])
    except:
        atm_iv = 0.0

    def clean(df, is_call):
        df = df.copy()
        df["mid"] = (df["bid"] + df["ask"]) / 2
        df["sprd"] = (df["ask"] - df["bid"]) / df["mid"].replace(0, np.nan)
        lo, hi = (spot * 0.86, spot * 1.01) if not is_call else (spot * 0.99, spot * 1.14)
        df = df[(df["strike"] >= lo) & (df["strike"] <= hi)]
        df = df[df["openInterest"] >= 100]
        df = df[df["sprd"] < 0.35]
        df["OTM%"] = ((df["strike"] - spot) / spot * 100).round(1)
        df["$/ct"] = (df["mid"] * 100).round(0).astype(int)
        df["IV%"] = (df["impliedVolatility"] * 100).round(1)
        df["Sprd%"] = (df["sprd"] * 100).round(1)
        return (
            df[["strike", "mid", "openInterest", "IV%", "OTM%", "$/ct", "Sprd%"]]
            .rename(columns={"strike": "Strike", "mid": "Mid", "openInterest": "OI"})
            .sort_values("Strike", ascending=not is_call)
            .head(6)
            .reset_index(drop=True)
        )

    puts = clean(chain.puts, False) if direction in ("DOWN", "MIXED") else pd.DataFrame()
    calls = clean(chain.calls, True) if direction in ("UP", "MIXED") else pd.DataFrame()
    return puts, calls, atm_iv


def pick_expiry(t, earn_days):
    for exp in t.options:
        d = (datetime.strptime(exp, "%Y-%m-%d") - datetime.today()).days
        if d >= earn_days + 1:
            return exp
    return None


def scan_ticker(sym, max_earn_days):
    t = yf.Ticker(sym)
    earn_date, earn_days = get_earnings(t, max_earn_days)
    if earn_date is None:
        return None

    spot, ma20, ma50, direction, hv30 = get_direction(t)
    net_gex, gex_by_strike = compute_gex(t, spot)
    expiry = pick_expiry(t, earn_days)
    if expiry is None:
        return None

    puts, calls, atm_iv = get_chain(t, expiry, spot, direction)
    exp_move = spot * atm_iv * np.sqrt(max(earn_days, 1) / 365) if atm_iv > 0 else 0.0
    iv_hv = atm_iv / hv30 if hv30 > 0 else 0.0

    if direction == "DOWN" and net_gex < 0:
        score = 3
    elif direction == "UP" and net_gex > 0:
        score = 2
    elif direction != "MIXED":
        score = 1
    else:
        score = 0

    return dict(
        sym=sym, spot=spot, ma20=ma20, ma50=ma50,
        direction=direction, hv30=hv30, atm_iv=atm_iv, iv_hv=iv_hv,
        net_gex=net_gex, gex_by_strike=gex_by_strike,
        earn_date=earn_date, earn_days=earn_days,
        expiry=expiry, exp_move=exp_move,
        puts=puts, calls=calls, score=score,
    )

# ── UI ────────────────────────────────────────────────────────────────────

st.title("📊 Options Earnings Screener")
st.caption("MA direction + GEX analysis · backtesting only · no real orders · no number used that wasn't fetched this run")

with st.sidebar:
    st.header("⚙️ Filters")
    max_earn_days = st.slider("Earnings within next N days", 1, 21, 14)
    min_iv_hv = st.slider("Min IV/HV ratio", 1.0, 4.0, 1.5, 0.1)
    show_mixed = st.checkbox("Include MIXED direction", value=False)
    st.divider()
    st.caption(f"Universe: {len(UNIVERSE)} tickers")
    st.caption("Data: yfinance (live)")

run_btn = st.button("🔍 Run Scan", type="primary", use_container_width=True)

if "results" not in st.session_state:
    st.session_state.results = []
    st.session_state.scanned_at = None

if run_btn:
    st.session_state.results = []
    progress = st.progress(0.0, text="Initializing...")
    status = st.empty()

    for i, sym in enumerate(UNIVERSE):
        pct = (i + 1) / len(UNIVERSE)
        progress.progress(pct, text=f"Scanning {sym}… ({i+1}/{len(UNIVERSE)})")
        try:
            r = scan_ticker(sym, max_earn_days)
            if r:
                st.session_state.results.append(r)
        except:
            pass

    progress.empty()
    status.empty()
    st.session_state.scanned_at = datetime.now().strftime("%b %d %Y %H:%M")

# ── Results ───────────────────────────────────────────────────────────────

if st.session_state.results:
    results = st.session_state.results

    if not show_mixed:
        results = [r for r in results if r["direction"] != "MIXED"]
    results = [r for r in results if r["iv_hv"] >= min_iv_hv]
    results = sorted(results, key=lambda x: (x["earn_days"], -x["score"]))

    if st.session_state.scanned_at:
        st.caption(f"Last scan: {st.session_state.scanned_at}")

    if not results:
        st.warning("No tickers matched your filters. Try lowering the IV/HV minimum or enabling MIXED direction.")
    else:
        st.success(f"**{len(results)} tickers** with earnings in the next {max_earn_days} days")

        # Summary table
        summary_rows = []
        for r in results:
            gex_str = f"{r['net_gex']/1e6:+.1f}M"
            summary_rows.append({
                "Ticker": r["sym"],
                "Spot": f"${r['spot']:.2f}",
                "Direction": r["direction"],
                "Earnings": f"in {r['earn_days']}d  ({r['earn_date']})",
                "Net GEX": gex_str,
                "GEX Type": "⚡ DESTAB" if r["net_gex"] < 0 else "🧲 stab",
                "IV/HV": f"{r['iv_hv']:.2f}x",
                "Exp Move": f"±${r['exp_move']:.2f}",
            })

        st.dataframe(
            pd.DataFrame(summary_rows),
            hide_index=True,
            use_container_width=True,
        )

        st.divider()

        # Detail cards
        for r in results:
            dir_icon = "🟢" if r["direction"] == "UP" else ("🔴" if r["direction"] == "DOWN" else "🟡")
            gex_icon = "⚡" if r["net_gex"] < 0 else "🧲"
            label = (
                f"{dir_icon} **{r['sym']}** · ${r['spot']:.2f} · "
                f"{r['direction']} · Earnings in {r['earn_days']}d · "
                f"GEX {r['net_gex']/1e6:.1f}M {gex_icon}"
            )

            with st.expander(label, expanded=(r["score"] == 3)):
                c1, c2, c3, c4, c5, c6 = st.columns(6)
                c1.metric("Spot", f"${r['spot']:.2f}")
                c2.metric("MA20", f"${r['ma20']:.2f}")
                c3.metric("MA50", f"${r['ma50']:.2f}")
                c4.metric("HV30", f"{r['hv30']*100:.1f}%")
                c5.metric("ATM IV", f"{r['atm_iv']*100:.1f}%")
                c6.metric("Exp Move", f"±${r['exp_move']:.2f}")

                st.caption(f"Expiry: **{r['expiry']}** · IV/HV: **{r['iv_hv']:.2f}x**")

                # GEX chart
                nearby = {
                    k: v / 1e6
                    for k, v in r["gex_by_strike"].items()
                    if abs(k - r["spot"]) / r["spot"] < 0.12
                }
                if nearby:
                    gex_df = (
                        pd.DataFrame(list(nearby.items()), columns=["Strike", "GEX (M)"])
                        .sort_values("Strike")
                        .set_index("Strike")
                    )
                    st.caption("GEX by strike (±12% of spot)")
                    st.bar_chart(gex_df, height=200)

                # Options tables
                col_p, col_c = st.columns(2)
                if not r["puts"].empty:
                    with col_p:
                        st.caption(f"**PUTS** — {r['expiry']}")
                        st.dataframe(r["puts"], hide_index=True, use_container_width=True)
                if not r["calls"].empty:
                    with col_c:
                        st.caption(f"**CALLS** — {r['expiry']}")
                        st.dataframe(r["calls"], hide_index=True, use_container_width=True)

elif not run_btn:
    st.info("👆 Hit **Run Scan** to fetch live data and screen the full universe.")
