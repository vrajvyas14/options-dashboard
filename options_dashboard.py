"""
╔══════════════════════════════════════════════════════════════╗
║         Real-Time Options Pricing & Risk Dashboard           ║
║         Black-Scholes · Monte Carlo · The Greeks             ║
╚══════════════════════════════════════════════════════════════╝

Install:  pip install streamlit yfinance numpy scipy plotly
Run:      streamlit run options_dashboard.py
"""

import streamlit as st
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings, datetime

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QuantDesk · Options Pricer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS  (terminal-green-on-black quant aesthetic)
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

:root {
    --bg:       #0a0d0f;
    --bg2:      #111518;
    --bg3:      #181d21;
    --border:   #1e2a30;
    --green:    #00ff88;
    --green2:   #00cc6a;
    --amber:    #ffb300;
    --red:      #ff4d6d;
    --text:     #c8d8e0;
    --muted:    #5a7080;
    --accent:   #00d4ff;
}

html, body, [data-testid="stApp"] {
    background-color: var(--bg) !important;
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: var(--bg2) !important;
    border-right: 1px solid var(--border);
}

/* Inputs */
.stSelectbox > div > div,
.stTextInput > div > div > input {
    background-color: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: 'Space Mono', monospace !important;
    border-radius: 4px !important;
}

/* Sliders */
.stSlider [data-baseweb="slider"] {
    padding: 0 !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 16px 20px;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-family: 'Space Mono', monospace; font-size: 11px; letter-spacing: 1.5px; text-transform: uppercase; }
[data-testid="stMetricValue"] { color: var(--green) !important; font-family: 'Space Mono', monospace; font-size: 1.6rem; }
[data-testid="stMetricDelta"] { font-family: 'Space Mono', monospace; font-size: 11px; }

/* Tabs */
[data-baseweb="tab-list"] { background: var(--bg2) !important; border-bottom: 1px solid var(--border); gap: 0; }
[data-baseweb="tab"] { color: var(--muted) !important; font-family: 'Space Mono', monospace !important; font-size: 12px !important; padding: 10px 20px !important; border-radius: 0 !important; }
[data-baseweb="tab"][aria-selected="true"] { color: var(--green) !important; border-bottom: 2px solid var(--green) !important; background: transparent !important; }

/* Divider */
hr { border-color: var(--border) !important; }

/* Headers */
h1 { font-family: 'Space Mono', monospace !important; color: var(--green) !important; font-size: 1.4rem !important; letter-spacing: 2px; }
h2 { font-family: 'Space Mono', monospace !important; color: var(--text) !important; font-size: 1rem !important; letter-spacing: 1px; border-bottom: 1px solid var(--border); padding-bottom: 6px; }
h3 { font-family: 'Space Mono', monospace !important; color: var(--accent) !important; font-size: 0.85rem !important; letter-spacing: 1px; }

/* Dataframe */
[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: 6px; }

/* Buttons */
.stButton > button {
    background: var(--green) !important;
    color: #000 !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 12px !important;
    letter-spacing: 1px !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 8px 24px !important;
}
.stButton > button:hover { background: var(--green2) !important; }

/* Info / warning boxes */
.info-box {
    background: rgba(0,212,255,0.07);
    border: 1px solid rgba(0,212,255,0.25);
    border-radius: 6px;
    padding: 12px 16px;
    font-size: 13px;
    color: var(--accent);
    font-family: 'Space Mono', monospace;
    margin-bottom: 12px;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# PLOTLY DARK THEME  (shared across all charts)
# ─────────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="#111518",
    plot_bgcolor="#0a0d0f",
    font=dict(family="Space Mono", color="#c8d8e0", size=11),
    xaxis=dict(gridcolor="#1e2a30", linecolor="#1e2a30", zerolinecolor="#1e2a30"),
    yaxis=dict(gridcolor="#1e2a30", linecolor="#1e2a30", zerolinecolor="#1e2a30"),
    margin=dict(l=40, r=20, t=40, b=40),
)

GREEN  = "#00ff88"
AMBER  = "#ffb300"
RED    = "#ff4d6d"
ACCENT = "#00d4ff"
MUTED  = "#5a7080"


# ═════════════════════════════════════════════════════════════
# CORE MATH
# ═════════════════════════════════════════════════════════════

def d1_d2(S, K, T, r, sigma):
    """Black-Scholes d1 and d2 terms."""
    if T <= 0 or sigma <= 0:
        return None, None
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def black_scholes_price(S, K, T, r, sigma, option_type="call"):
    """
    Black-Scholes option pricing formula.
    S     = current stock price
    K     = strike price
    T     = time to expiration (years)
    r     = risk-free interest rate (decimal)
    sigma = annualised volatility (decimal)
    """
    d1, d2 = d1_d2(S, K, T, r, sigma)
    if d1 is None:
        return max(S - K, 0) if option_type == "call" else max(K - S, 0)

    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return max(price, 0)


def greeks(S, K, T, r, sigma, option_type="call"):
    """
    The Greeks: Delta, Gamma, Theta, Vega, Rho.
    Returns a dict.
    """
    d1, d2 = d1_d2(S, K, T, r, sigma)
    if d1 is None:
        return {g: 0.0 for g in ("delta","gamma","theta","vega","rho")}

    phi_d1 = norm.pdf(d1)   # standard normal PDF at d1
    sqrt_T  = np.sqrt(T)
    exp_rT  = np.exp(-r * T)

    if option_type == "call":
        delta = norm.cdf(d1)
        theta = (
            -(S * phi_d1 * sigma) / (2 * sqrt_T)
            - r * K * exp_rT * norm.cdf(d2)
        ) / 365          # per calendar day
        rho   = K * T * exp_rT * norm.cdf(d2) / 100
    else:
        delta = norm.cdf(d1) - 1
        theta = (
            -(S * phi_d1 * sigma) / (2 * sqrt_T)
            + r * K * exp_rT * norm.cdf(-d2)
        ) / 365
        rho   = -K * T * exp_rT * norm.cdf(-d2) / 100

    gamma = phi_d1 / (S * sigma * sqrt_T)
    vega  = S * phi_d1 * sqrt_T / 100          # per 1% vol move

    return dict(delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)


def monte_carlo_price(S, K, T, r, sigma, option_type="call",
                      n_simulations=10_000, n_steps=252, seed=42):
    """
    Monte Carlo simulation using Geometric Brownian Motion.
    Returns (price, std_error, all_final_prices_array).
    """
    rng = np.random.default_rng(seed)
    dt  = T / n_steps

    # GBM: simulate log returns
    Z         = rng.standard_normal((n_simulations, n_steps))
    log_ret   = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    price_paths = S * np.exp(np.cumsum(log_ret, axis=1))   # (sims, steps)

    final_prices = price_paths[:, -1]

    if option_type == "call":
        payoffs = np.maximum(final_prices - K, 0)
    else:
        payoffs = np.maximum(K - final_prices, 0)

    discount  = np.exp(-r * T)
    mc_price  = discount * payoffs.mean()
    std_error = discount * payoffs.std() / np.sqrt(n_simulations)

    return mc_price, std_error, final_prices, price_paths


def implied_volatility(market_price, S, K, T, r, option_type="call"):
    """
    Finds IV via Brent's method. Returns NaN if unsolvable.
    """
    try:
        intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
        if market_price <= intrinsic:
            return float("nan")
        f = lambda v: black_scholes_price(S, K, T, r, v, option_type) - market_price
        return brentq(f, 1e-6, 10.0, xtol=1e-6, maxiter=500)
    except Exception:
        return float("nan")


# ═════════════════════════════════════════════════════════════
# DATA HELPERS
# ═════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def fetch_stock_data(ticker: str):
    """Pull live stock data via yfinance; gracefully degrade if unavailable."""
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        info  = t.info
        hist  = t.history(period="1y")
        price = info.get("regularMarketPrice") or info.get("currentPrice") or (
                    hist["Close"].iloc[-1] if not hist.empty else None)
        # Historical volatility (annualised from daily log returns)
        if not hist.empty and len(hist) > 20:
            log_ret = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
            hist_vol = float(log_ret.std() * np.sqrt(252))
        else:
            hist_vol = 0.25
        name = info.get("longName", ticker)
        return dict(price=price, hist_vol=hist_vol, name=name,
                    hist=hist, error=None)
    except Exception as e:
        return dict(price=None, hist_vol=0.25, name=ticker,
                    hist=None, error=str(e))


# ═════════════════════════════════════════════════════════════
# CHART BUILDERS
# ═════════════════════════════════════════════════════════════

def chart_payoff_diagram(S, K, T, r, sigma, option_type, bs_price):
    """P&L payoff at expiry + today's theoretical curve."""
    spot_range = np.linspace(S * 0.6, S * 1.4, 200)

    # At expiry
    if option_type == "call":
        payoff_expiry = np.maximum(spot_range - K, 0) - bs_price
    else:
        payoff_expiry = np.maximum(K - spot_range, 0) - bs_price

    # Today's BS curve
    today_vals = np.array([
        black_scholes_price(s, K, T, r, sigma, option_type) - bs_price
        for s in spot_range
    ])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=spot_range, y=today_vals,
        name="Today (theoretical)",
        line=dict(color=ACCENT, width=2, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=spot_range, y=payoff_expiry,
        name="At expiry",
        line=dict(color=GREEN, width=2.5),
        fill="tozeroy",
        fillcolor="rgba(0,255,136,0.06)",
    ))
    fig.add_vline(x=K, line=dict(color=AMBER, dash="dash", width=1),
                  annotation_text=f"Strike ${K:.0f}", annotation_font_color=AMBER)
    fig.add_vline(x=S, line=dict(color=MUTED, dash="dash", width=1),
                  annotation_text=f"Spot ${S:.2f}", annotation_font_color=MUTED)
    fig.add_hline(y=0, line=dict(color=RED, width=1, dash="dot"))

    fig.update_layout(**PLOT_LAYOUT,
                      title="P&L Diagram",
                      xaxis_title="Stock Price at Expiry ($)",
                      yaxis_title="Profit / Loss ($)",
                      legend=dict(bgcolor="rgba(0,0,0,0)", x=0.01, y=0.99))
    return fig


def chart_mc_paths(price_paths, S, K, option_type, n_display=200):
    """Plot a sample of Monte Carlo simulation paths."""
    n_steps = price_paths.shape[1]
    time_ax = np.linspace(0, 1, n_steps)

    fig = go.Figure()
    idx = np.random.choice(price_paths.shape[0], min(n_display, price_paths.shape[0]),
                           replace=False)
    for i in idx:
        fig.add_trace(go.Scatter(
            x=time_ax, y=price_paths[i],
            mode="lines",
            line=dict(width=0.4, color="rgba(0,212,255,0.12)"),
            showlegend=False,
        ))

    # Median path
    fig.add_trace(go.Scatter(
        x=time_ax, y=np.median(price_paths, axis=0),
        name="Median path",
        line=dict(color=GREEN, width=2),
    ))
    fig.add_hline(y=K, line=dict(color=AMBER, dash="dash", width=1.5),
                  annotation_text=f"Strike ${K:.0f}", annotation_font_color=AMBER)
    fig.add_hline(y=S, line=dict(color=MUTED, dash="dot", width=1),
                  annotation_text=f"Spot S₀", annotation_font_color=MUTED)

    fig.update_layout(**PLOT_LAYOUT,
                      title=f"Monte Carlo Simulation ({price_paths.shape[0]:,} paths shown: {n_display})",
                      xaxis_title="Time (normalised to expiry)",
                      yaxis_title="Simulated Stock Price ($)",
                      legend=dict(bgcolor="rgba(0,0,0,0)"))
    return fig


def chart_mc_distribution(final_prices, K, option_type, mc_price):
    """Histogram of terminal stock prices from MC."""
    if option_type == "call":
        itm = final_prices[final_prices >= K]
        otm = final_prices[final_prices < K]
        itm_label, otm_label = "ITM (S > K)", "OTM (S < K)"
    else:
        itm = final_prices[final_prices <= K]
        otm = final_prices[final_prices > K]
        itm_label, otm_label = "ITM (S < K)", "OTM (S > K)"

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=itm, name=itm_label,
        marker_color="rgba(0,255,136,0.55)", nbinsx=60,
    ))
    fig.add_trace(go.Histogram(
        x=otm, name=otm_label,
        marker_color="rgba(255,77,109,0.35)", nbinsx=60,
    ))
    fig.add_vline(x=K, line=dict(color=AMBER, dash="dash", width=2),
                  annotation_text=f"K=${K:.0f}", annotation_font_color=AMBER)
    fig.update_layout(**PLOT_LAYOUT,
                      barmode="overlay",
                      title="Terminal Price Distribution",
                      xaxis_title="Stock Price at Expiry ($)",
                      yaxis_title="Frequency",
                      legend=dict(bgcolor="rgba(0,0,0,0)"))
    return fig


def chart_greeks_surface(S, K, T, r, sigma, option_type, greek_name):
    """3D surface: chosen Greek vs (Spot, Volatility)."""
    spot_range = np.linspace(S * 0.7, S * 1.3, 40)
    vol_range  = np.linspace(0.05, 0.80, 40)
    SS, VV = np.meshgrid(spot_range, vol_range)
    ZZ = np.vectorize(
        lambda s, v: greeks(s, K, T, r, v, option_type)[greek_name]
    )(SS, VV)

    fig = go.Figure(data=[go.Surface(
        x=SS, y=VV, z=ZZ,
        colorscale=[[0, "#0a0d0f"], [0.3, "#00cc6a"], [0.7, "#00d4ff"], [1, "#ffb300"]],
        opacity=0.9,
        contours=dict(z=dict(show=True, usecolormap=True, highlightcolor=GREEN, project_z=True)),
    )])
    fig.update_layout(
        paper_bgcolor="#111518",
        plot_bgcolor="#0a0d0f",
        font=dict(family="Space Mono", color="#c8d8e0", size=10),
        title=f"{greek_name.capitalize()} Surface",
        scene=dict(
            xaxis_title="Stock Price",
            yaxis_title="Volatility",
            zaxis_title=greek_name.capitalize(),
            xaxis=dict(backgroundcolor="#0a0d0f", gridcolor="#1e2a30"),
            yaxis=dict(backgroundcolor="#0a0d0f", gridcolor="#1e2a30"),
            zaxis=dict(backgroundcolor="#111518", gridcolor="#1e2a30"),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def chart_vol_smile(S, K, T, r, option_type):
    """Implied volatility smile (flat here; shows concept)."""
    strikes = np.linspace(S * 0.75, S * 1.25, 30)
    # Simulate a realistic vol smile shape around ATM
    moneyness = np.log(strikes / S)
    # Simple quadratic smile
    base_vol   = 0.25
    smile_vols = base_vol + 0.08 * moneyness**2 - 0.02 * moneyness

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=strikes, y=smile_vols * 100,
        mode="lines+markers",
        name="Implied Vol",
        line=dict(color=GREEN, width=2),
        marker=dict(size=4, color=GREEN),
    ))
    fig.add_vline(x=S, line=dict(color=MUTED, dash="dot"),
                  annotation_text="ATM", annotation_font_color=MUTED)
    fig.add_vline(x=K, line=dict(color=AMBER, dash="dash"),
                  annotation_text=f"K=${K:.0f}", annotation_font_color=AMBER)
    fig.update_layout(**PLOT_LAYOUT,
                      title="Implied Volatility Smile (Stylised)",
                      xaxis_title="Strike Price ($)",
                      yaxis_title="Implied Volatility (%)",
                      legend=dict(bgcolor="rgba(0,0,0,0)"))
    return fig


def chart_historical_prices(hist, ticker):
    """Candlestick chart of historical prices."""
    if hist is None or hist.empty:
        return None
    fig = go.Figure(data=[go.Candlestick(
        x=hist.index,
        open=hist["Open"], high=hist["High"],
        low=hist["Low"], close=hist["Close"],
        increasing_line_color=GREEN,
        decreasing_line_color=RED,
        increasing_fillcolor="rgba(0,255,136,0.4)",
        decreasing_fillcolor="rgba(255,77,109,0.4)",
    )])
    fig.update_layout(**PLOT_LAYOUT,
                      title=f"{ticker} — 1-Year Price History",
                      xaxis_title="Date", yaxis_title="Price ($)",
                      xaxis_rangeslider_visible=False)
    return fig


# ═════════════════════════════════════════════════════════════
# SIDEBAR  — inputs
# ═════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⬛ QUANTDESK")
    st.markdown('<div class="info-box">Options Pricing · Risk Analytics · Monte Carlo</div>',
                unsafe_allow_html=True)

    st.markdown("### UNDERLYING")
    ticker_input = st.text_input("Ticker Symbol", value="AAPL",
                                 help="Any Yahoo Finance ticker, e.g. MSFT, TSLA, SPY").upper()

    use_live = st.toggle("Fetch live market data", value=True,
                         help="Uses yfinance. Toggle off for manual entry.")

    stock_data = None
    if use_live:
        with st.spinner("Fetching..."):
            stock_data = fetch_stock_data(ticker_input)
        if stock_data["error"]:
            st.warning(f"Live data unavailable: {stock_data['error'][:60]}")

    live_price  = stock_data["price"] if (stock_data and stock_data["price"]) else 150.0
    live_hv     = stock_data["hist_vol"] if stock_data else 0.25

    S = st.number_input("Spot Price (S)", value=round(live_price, 2),
                        min_value=0.01, step=0.5, format="%.2f")

    st.markdown("### CONTRACT")
    option_type = st.selectbox("Option Type", ["call", "put"])
    K = st.number_input("Strike Price (K)", value=float(round(S / 5) * 5) or 150.0,
                        min_value=0.01, step=1.0, format="%.2f")
    expiry = st.date_input("Expiration Date",
                           value=datetime.date.today() + datetime.timedelta(days=90),
                           min_value=datetime.date.today() + datetime.timedelta(days=1))
    T = max((expiry - datetime.date.today()).days / 365.0, 1/365)

    st.markdown("### MODEL PARAMETERS")
    sigma = st.slider("Implied Volatility σ", 0.05, 1.50,
                      float(round(live_hv, 2)), 0.01,
                      format="%.2f",
                      help="Annualised volatility (decimal). Set by yfinance hist vol.")
    r = st.slider("Risk-Free Rate r", 0.00, 0.15, 0.053, 0.001,
                  format="%.3f", help="US 10-yr approx: ~5.3%")

    st.markdown("### SIMULATION")
    n_sims = st.select_slider("MC Simulations",
                              options=[1_000, 5_000, 10_000, 50_000, 100_000],
                              value=10_000)
    run_mc = st.button("▶  RUN MONTE CARLO")

    st.markdown("---")
    st.caption(f"T = {T*365:.0f} days ({T:.4f} yrs)")
    moneyness_pct = (S - K) / K * 100
    moneyness_str = "ATM" if abs(moneyness_pct) < 1 else (
        f"{'↑' if moneyness_pct > 0 else '↓'} {abs(moneyness_pct):.1f}% {'ITM' if (moneyness_pct>0)==(option_type=='call') else 'OTM'}")
    st.caption(f"Moneyness: {moneyness_str}")

# ═════════════════════════════════════════════════════════════
# MAIN — compute
# ═════════════════════════════════════════════════════════════

bs_price = black_scholes_price(S, K, T, r, sigma, option_type)
g        = greeks(S, K, T, r, sigma, option_type)
intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
time_val  = bs_price - intrinsic

# Cache MC result in session state
if "mc_result" not in st.session_state:
    st.session_state["mc_result"] = None

if run_mc or st.session_state["mc_result"] is None:
    with st.spinner(f"Simulating {n_sims:,} paths…"):
        mc_price, mc_se, final_prices, price_paths = monte_carlo_price(
            S, K, T, r, sigma, option_type, n_simulations=n_sims)
        st.session_state["mc_result"] = (mc_price, mc_se, final_prices, price_paths)
else:
    mc_price, mc_se, final_prices, price_paths = st.session_state["mc_result"]

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
ticker_display = stock_data["name"] if stock_data else ticker_input
st.markdown(f"# 📈 {ticker_input}  ·  {option_type.upper()} OPTION")
st.markdown(f"**{ticker_display}**  |  Spot **${S:.2f}**  ·  Strike **${K:.2f}**  ·  Expiry **{expiry}**  ·  σ **{sigma:.1%}**  ·  r **{r:.2%}**")
st.markdown("---")

# ─────────────────────────────────────────────────────────────
# TOP METRICS
# ─────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Black-Scholes", f"${bs_price:.4f}")
c2.metric("Monte Carlo",   f"${mc_price:.4f}",
          delta=f"±${mc_se:.4f} SE",
          delta_color="off")
c3.metric("Intrinsic Value", f"${intrinsic:.4f}")
c4.metric("Time Value", f"${time_val:.4f}")
bs_vs_mc = bs_price - mc_price
c5.metric("BS − MC Diff", f"${bs_vs_mc:+.4f}",
          delta="converges with ↑ sims", delta_color="off")

st.markdown("---")

# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "  THE GREEKS  ",
    "  P&L DIAGRAM  ",
    "  MONTE CARLO  ",
    "  SURFACES  ",
    "  MARKET DATA  ",
])

# ── Tab 1: Greeks ──────────────────────────────────────────
with tab1:
    st.markdown("## THE GREEKS")

    g1, g2, g3, g4, g5 = st.columns(5)
    g1.metric("Δ Delta",  f"{g['delta']:+.4f}", help="∂Price/∂Spot — directional exposure")
    g2.metric("Γ Gamma",  f"{g['gamma']:+.6f}", help="∂Delta/∂Spot — convexity")
    g3.metric("Θ Theta",  f"{g['theta']:+.4f}", help="∂Price/∂Time (per day)")
    g4.metric("𝜈 Vega",   f"{g['vega']:+.4f}",  help="∂Price/∂Vol (per 1% vol move)")
    g5.metric("ρ Rho",    f"{g['rho']:+.4f}",   help="∂Price/∂Rate (per 1% rate move)")

    st.markdown("---")

    # Greek sensitivity charts
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### DELTA vs SPOT")
        spot_r = np.linspace(S * 0.5, S * 1.5, 200)
        delta_r = [greeks(s, K, T, r, sigma, option_type)["delta"] for s in spot_r]
        fig_d = go.Figure()
        fig_d.add_trace(go.Scatter(x=spot_r, y=delta_r,
                                   line=dict(color=GREEN, width=2), name="Delta"))
        fig_d.add_vline(x=S, line=dict(color=MUTED, dash="dot"),
                        annotation_text="Current", annotation_font_color=MUTED)
        fig_d.add_vline(x=K, line=dict(color=AMBER, dash="dash"),
                        annotation_text="Strike", annotation_font_color=AMBER)
        fig_d.update_layout(**PLOT_LAYOUT, title="Delta",
                            xaxis_title="Spot ($)", yaxis_title="Delta")
        st.plotly_chart(fig_d, use_container_width=True)

    with col_right:
        st.markdown("### GAMMA vs SPOT")
        gamma_r = [greeks(s, K, T, r, sigma, option_type)["gamma"] for s in spot_r]
        fig_g = go.Figure()
        fig_g.add_trace(go.Scatter(x=spot_r, y=gamma_r,
                                   line=dict(color=ACCENT, width=2), name="Gamma"))
        fig_g.add_vline(x=K, line=dict(color=AMBER, dash="dash"),
                        annotation_text="Strike", annotation_font_color=AMBER)
        fig_g.update_layout(**PLOT_LAYOUT, title="Gamma",
                            xaxis_title="Spot ($)", yaxis_title="Gamma")
        st.plotly_chart(fig_g, use_container_width=True)

    col_left2, col_right2 = st.columns(2)
    with col_left2:
        st.markdown("### THETA vs DAYS TO EXPIRY")
        t_range  = np.linspace(1/365, T + 0.5, 150)
        theta_r  = [greeks(S, K, t, r, sigma, option_type)["theta"] for t in t_range]
        days_r   = t_range * 365
        fig_t = go.Figure()
        fig_t.add_trace(go.Scatter(x=days_r, y=theta_r,
                                   line=dict(color=RED, width=2), name="Theta"))
        fig_t.add_vline(x=T * 365, line=dict(color=MUTED, dash="dot"),
                        annotation_text="Today", annotation_font_color=MUTED)
        fig_t.update_layout(**PLOT_LAYOUT, title="Theta Decay",
                            xaxis_title="Days to Expiry", yaxis_title="Theta ($/day)")
        st.plotly_chart(fig_t, use_container_width=True)

    with col_right2:
        st.markdown("### VEGA vs VOLATILITY")
        vol_range = np.linspace(0.05, 1.0, 150)
        vega_r    = [greeks(S, K, T, r, v, option_type)["vega"] for v in vol_range]
        fig_v = go.Figure()
        fig_v.add_trace(go.Scatter(x=vol_range * 100, y=vega_r,
                                   line=dict(color=AMBER, width=2), name="Vega"))
        fig_v.add_vline(x=sigma * 100, line=dict(color=MUTED, dash="dot"),
                        annotation_text="Current σ", annotation_font_color=MUTED)
        fig_v.update_layout(**PLOT_LAYOUT, title="Vega",
                            xaxis_title="Volatility (%)", yaxis_title="Vega ($/1% vol)")
        st.plotly_chart(fig_v, use_container_width=True)

    # Greeks interpretation guide
    st.markdown("---")
    st.markdown("### GREEK INTERPRETATION")
    greek_data = {
        "Greek": ["Δ Delta", "Γ Gamma", "Θ Theta", "𝜈 Vega", "ρ Rho"],
        "Value": [f"{g['delta']:+.4f}", f"{g['gamma']:+.6f}", f"{g['theta']:+.4f}",
                  f"{g['vega']:+.4f}", f"{g['rho']:+.4f}"],
        "Interpretation": [
            f"Approx. probability of expiring ITM; hedge ratio vs {abs(g['delta']):.0%} of a share short",
            f"Delta changes by {g['gamma']:.5f} for each $1 move in spot",
            f"Loses ${abs(g['theta']):.4f}/day in time value (all else equal)",
            f"Gains ${g['vega']:.4f} for each 1% rise in implied volatility",
            f"Gains ${g['rho']:.4f} for each 1% rise in risk-free rates",
        ],
    }
    import pandas as pd
    st.dataframe(pd.DataFrame(greek_data), use_container_width=True, hide_index=True)


# ── Tab 2: P&L Diagram ────────────────────────────────────
with tab2:
    st.markdown("## P&L DIAGRAM")
    st.plotly_chart(chart_payoff_diagram(S, K, T, r, sigma, option_type, bs_price),
                    use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### BREAK-EVEN ANALYSIS")
        if option_type == "call":
            be = K + bs_price
            st.metric("Break-Even at Expiry", f"${be:.2f}",
                      delta=f"{(be/S-1)*100:+.2f}% from spot")
        else:
            be = K - bs_price
            st.metric("Break-Even at Expiry", f"${be:.2f}",
                      delta=f"{(be/S-1)*100:+.2f}% from spot")

    with col_b:
        st.markdown("### IMPLIED VOLATILITY SMILE")
        st.plotly_chart(chart_vol_smile(S, K, T, r, option_type),
                        use_container_width=True)


# ── Tab 3: Monte Carlo ────────────────────────────────────
with tab3:
    st.markdown("## MONTE CARLO SIMULATION")
    st.markdown(
        f'<div class="info-box">Simulated {n_sims:,} GBM paths · '
        f'MC Price: <b>${mc_price:.4f}</b> ± {mc_se:.4f} (1σ SE) · '
        f'ITM probability: <b>{(final_prices >= K if option_type=="call" else final_prices <= K).mean()*100:.1f}%</b>'
        f'</div>',
        unsafe_allow_html=True
    )

    st.plotly_chart(chart_mc_paths(price_paths, S, K, option_type),
                    use_container_width=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.plotly_chart(chart_mc_distribution(final_prices, K, option_type, mc_price),
                        use_container_width=True)
    with col_r:
        st.markdown("### SIMULATION STATISTICS")
        import pandas as pd
        if option_type == "call":
            payoffs_disp = np.maximum(final_prices - K, 0)
        else:
            payoffs_disp = np.maximum(K - final_prices, 0)

        stats = {
            "Metric": ["Mean Final Price", "Median Final Price", "Std Dev (Final)",
                       "5th Percentile", "95th Percentile",
                       "Mean Payoff (undiscounted)", "MC Price (discounted)",
                       "Std Error", "95% CI Lower", "95% CI Upper"],
            "Value": [
                f"${final_prices.mean():.2f}",
                f"${np.median(final_prices):.2f}",
                f"${final_prices.std():.2f}",
                f"${np.percentile(final_prices, 5):.2f}",
                f"${np.percentile(final_prices, 95):.2f}",
                f"${payoffs_disp.mean():.4f}",
                f"${mc_price:.4f}",
                f"${mc_se:.4f}",
                f"${mc_price - 1.96*mc_se:.4f}",
                f"${mc_price + 1.96*mc_se:.4f}",
            ],
        }
        st.dataframe(pd.DataFrame(stats), use_container_width=True, hide_index=True)


# ── Tab 4: Surfaces ───────────────────────────────────────
with tab4:
    st.markdown("## GREEK SURFACES")
    st.markdown("3-D surfaces show how each Greek varies across spot price and volatility.")

    sel_greek = st.selectbox("Select Greek to visualise",
                             ["delta", "gamma", "theta", "vega", "rho"])
    st.plotly_chart(chart_greeks_surface(S, K, T, r, sigma, option_type, sel_greek),
                    use_container_width=True, height=550)

    # Sensitivity table
    st.markdown("---")
    st.markdown("### SCENARIO ANALYSIS  —  Price across Spot × Volatility")
    import pandas as pd
    spot_scenarios = np.linspace(S * 0.85, S * 1.15, 7)
    vol_scenarios  = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
    tbl = pd.DataFrame(
        [[f"${black_scholes_price(s, K, T, r, v, option_type):.2f}"
          for s in spot_scenarios]
         for v in vol_scenarios],
        index=[f"σ={v:.0%}" for v in vol_scenarios],
        columns=[f"${s:.0f}" for s in spot_scenarios],
    )
    st.dataframe(tbl, use_container_width=True)


# ── Tab 5: Market Data ────────────────────────────────────
with tab5:
    st.markdown("## MARKET DATA")
    if stock_data and not stock_data.get("error") and stock_data["hist"] is not None:
        st.markdown(f"**{ticker_display}** historical data (1 year)")
        fig_hist = chart_historical_prices(stock_data["hist"], ticker_input)
        if fig_hist:
            st.plotly_chart(fig_hist, use_container_width=True)

        hist = stock_data["hist"]
        log_ret = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("### RETURNS DISTRIBUTION")
            fig_ret = go.Figure()
            fig_ret.add_trace(go.Histogram(
                x=log_ret * 100,
                name="Daily log returns",
                marker_color="rgba(0,255,136,0.55)",
                nbinsx=50,
            ))
            mu, std = log_ret.mean() * 100, log_ret.std() * 100
            x_norm = np.linspace(mu - 4*std, mu + 4*std, 200)
            y_norm = norm.pdf(x_norm, mu, std) * len(log_ret) * (log_ret.max() - log_ret.min()) * 100 / 50
            fig_ret.add_trace(go.Scatter(x=x_norm, y=y_norm,
                                         name="Normal fit",
                                         line=dict(color=AMBER, width=2)))
            fig_ret.update_layout(**PLOT_LAYOUT, title="Daily Log Returns",
                                  xaxis_title="Return (%)", yaxis_title="Frequency")
            st.plotly_chart(fig_ret, use_container_width=True)

        with col_b:
            st.markdown("### ROLLING VOLATILITY")
            rv_30 = log_ret.rolling(30).std()  * np.sqrt(252) * 100
            rv_60 = log_ret.rolling(60).std()  * np.sqrt(252) * 100
            rv_90 = log_ret.rolling(90).std()  * np.sqrt(252) * 100
            fig_rvol = go.Figure()
            for rv, label, colour in [(rv_30, "30d", GREEN), (rv_60, "60d", ACCENT), (rv_90, "90d", AMBER)]:
                fig_rvol.add_trace(go.Scatter(x=rv.index, y=rv,
                                              name=label, line=dict(color=colour, width=1.5)))
            fig_rvol.add_hline(y=sigma * 100, line=dict(color=RED, dash="dash"),
                               annotation_text="Current σ", annotation_font_color=RED)
            fig_rvol.update_layout(**PLOT_LAYOUT, title="Realised Volatility (annualised %)",
                                   xaxis_title="Date", yaxis_title="Volatility (%)")
            st.plotly_chart(fig_rvol, use_container_width=True)
    else:
        st.info("Enable 'Fetch live market data' in the sidebar and ensure yfinance is installed.")
        st.code("pip install yfinance", language="bash")

    # Pricing model summary
    st.markdown("---")
    st.markdown("### MODEL SUMMARY")
    import pandas as pd
    summary = {
        "Parameter": ["Spot (S)", "Strike (K)", "Expiry", "T (years)", "σ (vol)",
                       "r (rate)", "Option Type", "BS Price", "MC Price", "Intrinsic", "Time Value",
                       "Delta", "Gamma", "Theta/day", "Vega/1%vol", "Rho/1%rate"],
        "Value": [f"${S:.2f}", f"${K:.2f}", str(expiry), f"{T:.4f}",
                  f"{sigma:.2%}", f"{r:.3%}", option_type.upper(),
                  f"${bs_price:.4f}", f"${mc_price:.4f} ±{mc_se:.4f}",
                  f"${intrinsic:.4f}", f"${time_val:.4f}",
                  f"{g['delta']:+.4f}", f"{g['gamma']:+.6f}",
                  f"{g['theta']:+.4f}", f"{g['vega']:+.4f}", f"{g['rho']:+.4f}"],
    }
    st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="color:#5a7080; font-size:11px; font-family:\'Space Mono\',monospace;">'
    '⚠️ Educational purposes only. Not financial advice. '
    'Black-Scholes assumes log-normal returns, constant volatility & rates, and European-style exercise. '
    'Real options pricing involves additional complexity.'
    '</p>',
    unsafe_allow_html=True
)
