import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# ── Configuration ───────────────────────────────────────────────
st.set_page_config(
    page_title="Brent-WTI Arbitrage Monitor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Local dev: set these in .streamlit/secrets.toml (gitignored).
# Deployed: set these in Settings > Secrets on share.streamlit.io.
try:
    EIA_KEY  = st.secrets["EIA_KEY"]
    FRED_KEY = st.secrets["FRED_KEY"]
except Exception:
    st.error(
        "Missing EIA_KEY / FRED_KEY. Add them to .streamlit/secrets.toml locally, "
        "or in Settings > Secrets when deployed on Streamlit Community Cloud."
    )
    st.stop()

# ── Colour palette ──────────────────────────────────────────────
C = {
    "brent":     "#2563EB",
    "wti":       "#F59E0B",
    "spread":    "#6366F1",
    "open":      "rgba(16,185,129,0.25)",
    "open_line": "#10B981",
    "closed":    "rgba(239,68,68,0.18)",
    "closed_line":"#EF4444",
    "neutral":   "#94A3B8",
    "grid":      "rgba(203,213,225,0.4)",
    "bg":        "#FFFFFF",
    "usd":       "#8B5CF6",
    "rig":       "#EC4899",
    "inv":       "#0EA5E9",
    "curve":     "#F97316",
}

PLOT_LAYOUT = dict(
    plot_bgcolor=C["bg"],
    paper_bgcolor=C["bg"],
    font=dict(family="Inter, Arial, sans-serif", size=12, color="#1E293B"),
    margin=dict(l=60, r=40, t=50, b=40),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

# ── CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { background-color: #F8FAFC; }

div[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 16px 20px;
}
div[data-testid="stMetricLabel"] { font-size: 11px; font-weight: 500;
    text-transform: uppercase; letter-spacing: 0.06em; color: #64748B; }
div[data-testid="stMetricValue"] { font-size: 22px; font-weight: 600; color: #0F172A; }

.signal-open {
    background: #F0FDF4; border: 1.5px solid #10B981;
    border-radius: 8px; padding: 20px 24px; margin: 8px 0;
}
.signal-closed {
    background: #FFF1F2; border: 1.5px solid #EF4444;
    border-radius: 8px; padding: 20px 24px; margin: 8px 0;
}
.signal-monitor {
    background: #FFFBEB; border: 1.5px solid #F59E0B;
    border-radius: 8px; padding: 20px 24px; margin: 8px 0;
}
.signal-title { font-size: 18px; font-weight: 600; color: #0F172A; margin-bottom: 6px; }
.signal-body  { font-size: 13px; color: #475569; line-height: 1.7; }

.factor-card {
    background: #FFFFFF; border: 1px solid #E2E8F0;
    border-radius: 8px; padding: 14px 18px; margin-bottom: 8px;
}
.factor-label { font-size: 11px; font-weight: 500; text-transform: uppercase;
    letter-spacing: 0.06em; color: #64748B; margin-bottom: 4px; }
.factor-value { font-size: 16px; font-weight: 600; color: #0F172A; }
.factor-dir   { font-size: 12px; color: #64748B; margin-top: 2px; }

.section-title {
    font-size: 13px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.07em; color: #64748B;
    border-bottom: 1px solid #E2E8F0; padding-bottom: 6px; margin-bottom: 12px;
}
.stTabs [data-baseweb="tab"] {
    font-size: 13px; font-weight: 500; color: #64748B;
    padding: 10px 18px;
}
.stTabs [aria-selected="true"] { color: #2563EB; border-bottom: 2px solid #2563EB; }
div[data-testid="stSidebar"] { background: #F1F5F9; border-right: 1px solid #E2E8F0; }
</style>
""", unsafe_allow_html=True)


# ── Data fetching ───────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_eia_series(series_ids: list, route: str, days: int = 548) -> pd.DataFrame:
    url   = f"https://api.eia.gov/v2/{route}/data/"
    start = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    end   = datetime.today().strftime("%Y-%m-%d")
    params = {
        "api_key": EIA_KEY, "frequency": "daily",
        "data[0]": "value",
        "start": start, "end": end,
        "sort[0][column]": "period", "sort[0][direction]": "asc",
        "length": 5000,
    }
    for i, s in enumerate(series_ids):
        params[f"facets[series][{i}]"] = s
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    rows = r.json()["response"]["data"]
    df = pd.DataFrame(rows)
    df["period"] = pd.to_datetime(df["period"])
    df["value"]  = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["value"])


@st.cache_data(ttl=3600)
def fetch_eia_weekly(series_id: str, route: str, days: int = 548) -> pd.DataFrame:
    url   = f"https://api.eia.gov/v2/{route}/data/"
    start = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    end   = datetime.today().strftime("%Y-%m-%d")
    params = {
        "api_key": EIA_KEY, "frequency": "weekly",
        "data[0]": "value",
        "facets[series][0]": series_id,
        "start": start, "end": end,
        "sort[0][column]": "period", "sort[0][direction]": "asc",
        "length": 5000,
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    rows = r.json()["response"]["data"]
    df = pd.DataFrame(rows)
    df["period"] = pd.to_datetime(df["period"])
    df["value"]  = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["value"])


@st.cache_data(ttl=3600)
def fetch_fred(series_id: str, days: int = 548) -> pd.DataFrame:
    start = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    url   = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id, "api_key": FRED_KEY,
        "observation_start": start, "file_type": "json",
        "sort_order": "asc",
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    obs = r.json()["observations"]
    df = pd.DataFrame(obs)[["date", "value"]]
    df["date"]  = pd.to_datetime(df["date"])
    # FRED uses "." for missing values — coerce to NaN and drop
    df["value"] = df["value"].replace(".", None)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["value"]).rename(columns={"date": "period"})


@st.cache_data(ttl=3600)
def fetch_us_production(days: int = 548) -> pd.DataFrame:
    """Fetch weekly US crude oil production from EIA (thousand barrels/day)."""
    url   = "https://api.eia.gov/v2/petroleum/sum/sndw/data/"
    start = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    end   = datetime.today().strftime("%Y-%m-%d")
    params = {
        "api_key": EIA_KEY, "frequency": "weekly",
        "data[0]": "value",
        "facets[series][0]": "WCRFPUS2",
        "start": start, "end": end,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": 5000,
    }
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        rows = r.json()["response"]["data"]
        df = pd.DataFrame(rows)
        df["period"] = pd.to_datetime(df["period"])
        df["value"]  = pd.to_numeric(df["value"], errors="coerce")
        result = df.dropna(subset=["value"])[["period","value"]].rename(columns={"value": "us_production_kbd"})
        if len(result) > 0:
            return result
    except Exception:
        pass
    # Fallback: use WPULEUS3 series (weekly US crude production, mbbl/day)
    url2 = "https://api.eia.gov/v2/petroleum/sum/sndw/data/"
    params["facets[series][0]"] = "WCRFPUS2"
    params["frequency"] = "weekly"
    # Return empty df with correct columns if all fails
    return pd.DataFrame(columns=["period", "us_production_kbd"])


@st.cache_data(ttl=3600)
def build_master() -> dict:
    # Spot prices
    spot_raw = fetch_eia_series(["RBRTE", "RWTC"], "petroleum/pri/spt")
    brent = spot_raw[spot_raw["series"] == "RBRTE"][["period","value"]].rename(columns={"value":"brent"})
    wti   = spot_raw[spot_raw["series"] == "RWTC"][["period","value"]].rename(columns={"value":"wti"})
    prices = pd.merge(brent, wti, on="period").sort_values("period").reset_index(drop=True)
    prices["gross_spread"] = prices["brent"] - prices["wti"]

    # Inventory (national — shown as the general macro/headline number)
    inv_raw = fetch_eia_weekly("WCRSTUS1", "petroleum/stoc/wstk")
    inv = inv_raw.rename(columns={"value": "inventory_kb"})
    inv["inv_change"] = inv["inventory_kb"].diff()

    # Cushing, OK stocks specifically — WTI is priced AT Cushing, so this is a
    # much more direct driver of WTI-Brent spread dynamics than national
    # aggregate stocks. Empirically it roughly triples the regression's R²
    # versus using national stocks (0.52 -> 0.72 over a 12-month window).
    cushing_raw = fetch_eia_weekly("W_EPC0_SAX_YCUOK_MBBL", "petroleum/stoc/wstk")
    cushing = cushing_raw.rename(columns={"value": "cushing_kb"})

    # USD trade-weighted index (FRED: DTWEXBGS)
    usd = fetch_fred("DTWEXBGS").rename(columns={"value": "usd"})

    # US crude production (EIA weekly, thousand barrels/day)
    rig = fetch_us_production()
    rig = rig.rename(columns={"us_production_kbd": "rig_count"})

    # CBOE Crude Oil ETF Volatility Index (OVX) — the oil market's implied-vol
    # "fear gauge", derived from options prices. Unlike inventory data it's
    # forward-looking and reacts to geopolitical/supply-shock risk immediately
    # (it spiked from ~30 to ~120 within two weeks of the Feb-2026 Strait of
    # Hormuz crisis), which inventory-based measures structurally cannot do.
    ovx = fetch_fred("OVXCLS").rename(columns={"value": "ovx"})

    # Market structure proxy. EIA discontinued its WTI futures-curve series
    # (RCLC1-6) after Apr-2024, so no real forward curve is available. Per
    # Working's theory of storage, Cushing inventory running below/above its
    # own trailing average is the standard practical proxy for backwardation
    # vs contango, so we estimate directionally from it rather than fabricate
    # contract prices. This is a standardised score, not a $/bbl spread. It
    # only captures slow, structural tightness — it has no way to see a
    # geopolitical shock coming, which is what OVX (above) is for.
    structure = cushing[["period", "cushing_kb"]].copy()
    roll = structure["cushing_kb"].rolling(26, min_periods=8)
    structure["structure_proxy"] = -(structure["cushing_kb"] - roll.mean()) / roll.std()

    return {
        "prices": prices,
        "structure": structure[["period", "structure_proxy"]],
        "inventory": inv,
        "cushing": cushing,
        "usd": usd,
        "rig": rig,
        "ovx": ovx,
    }


# ── Regression model ────────────────────────────────────────────
def build_driver_model(data: dict, lookback_days: int = 365) -> dict:
    prices  = data["prices"].copy()
    cushing = data["cushing"].copy()
    usd     = data["usd"].copy()
    rig     = data["rig"].copy()
    ovx     = data["ovx"].copy()

    cutoff = datetime.today() - timedelta(days=lookback_days)

    # Forward-fill weekly/monthly series to daily
    base = prices[prices["period"] >= cutoff][["period","gross_spread"]].copy().reset_index(drop=True)

    def merge_ff(df, col):
        m = pd.merge_asof(base.sort_values("period"),
                          df[["period", col]].sort_values("period"),
                          on="period", direction="backward")
        return m[col]

    base["cushing_kb"]  = merge_ff(cushing, "cushing_kb")
    base["usd"]         = merge_ff(usd, "usd")
    base["ovx"]         = merge_ff(ovx, "ovx")
    # US production — use if available, else fill with mean (neutral)
    if len(rig) > 10:
        base["rig_count"] = merge_ff(rig, "rig_count")
    else:
        base["rig_count"] = 13000.0  # approximate recent avg in kbd
    base = base.dropna()

    if len(base) < 30:
        return None

    X_raw = base[["cushing_kb","usd","rig_count","ovx"]].values
    y     = base["gross_spread"].values

    scaler = StandardScaler()
    X      = scaler.fit_transform(X_raw)

    model  = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2     = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    coeffs = dict(zip(
        ["Cushing Crude Inventory", "USD Index", "US Production (kbd)", "Oil Volatility (OVX)"],
        model.coef_
    ))

    # Rolling 90-day correlations with spread
    base_s = base.set_index("period").sort_index()
    roll_corr = {}
    for col, label in [("cushing_kb","Cushing Inventory"),("usd","USD"),
                       ("rig_count","US Production"),("ovx","OVX")]:
        roll_corr[label] = base_s["gross_spread"].rolling(90).corr(base_s[col])

    # Current driver readings (latest values)
    latest = base.iloc[-1]
    current_drivers = {
        "Cushing Crude Inventory": {
            "value": f"{latest['cushing_kb']:,.0f} kb",
            "coeff": coeffs["Cushing Crude Inventory"],
            "direction": "Bearish for spread (builds → WTI weakens)" if latest["cushing_kb"] > base["cushing_kb"].mean() else "Neutral",
        },
        "USD Index": {
            "value": f"{latest['usd']:.1f}",
            "coeff": coeffs["USD Index"],
            "direction": "Above avg, headwind for crude prices" if latest["usd"] > base["usd"].mean() else "Below avg, supportive for crude",
        },
        "US Production (kbd)": {
            "value": f"{latest['rig_count']:,.0f} kbd",
            "coeff": coeffs["US Production (kbd)"],
            "direction": "Supply pressure elevated" if latest["rig_count"] > base["rig_count"].mean() else "Below average, supply tightening",
        },
        "Oil Volatility (OVX)": {
            "value": f"{latest['ovx']:.1f}",
            "coeff": coeffs["Oil Volatility (OVX)"],
            # Not asserting a fixed causal direction here: OVX overlaps heavily
            # with Cushing inventory (corr ~0.77 over this window), so its
            # fitted sign can flip between windows depending on which variable
            # "wins" the shared variance, see Methodology for detail.
            "direction": "Elevated, above-normal risk premium priced in" if latest["ovx"] > base["ovx"].mean() else "Below average, no unusual risk premium",
        },
    }

    return {
        "r2": r2,
        "coeffs": coeffs,
        "roll_corr": roll_corr,
        "current_drivers": current_drivers,
        "base": base,
        "y_pred": y_pred,
        "model": model,
        "scaler": scaler,
    }


# ── Opportunity signal ──────────────────────────────────────────
def compute_signal(data: dict, net_margin: float, total_cost: float) -> dict:
    cushing = data["cushing"]
    usd     = data["usd"]
    rig     = data["rig"]
    ovx     = data["ovx"]
    prices  = data["prices"]

    # Latest values
    latest_cushing = cushing["cushing_kb"].iloc[-1]
    prev_cushing   = cushing["cushing_kb"].iloc[-5] if len(cushing) > 5 else latest_cushing
    cushing_building = latest_cushing > prev_cushing
    cushing_pct    = cushing["cushing_kb"].rank(pct=True).iloc[-1]

    latest_usd    = usd["usd"].iloc[-1]
    usd_pct       = usd["usd"].rank(pct=True).iloc[-1]
    usd_elevated  = usd_pct > 0.60

    latest_rig    = rig["rig_count"].iloc[-1]
    rig_pct       = rig["rig_count"].rank(pct=True).iloc[-1]

    latest_ovx    = ovx["ovx"].iloc[-1]
    ovx_pct       = ovx["ovx"].rank(pct=True).iloc[-1]

    spread_series = prices["gross_spread"]
    spread_pct    = spread_series.rank(pct=True).iloc[-1]

    # Score each factor: +1 bullish for arb open, -1 bearish, 0 neutral
    scores = {}

    # Cushing inventory: building stocks at the WTI delivery point widen the spread (bullish for arb)
    if cushing_building and cushing_pct > 0.60:
        scores["Cushing Inventory"] = (+1, "Building, above historical average, widens Brent premium")
    elif not cushing_building and cushing_pct < 0.40:
        scores["Cushing Inventory"] = (-1, "Drawing, below average, supportive for WTI, narrows spread")
    else:
        scores["Cushing Inventory"] = (0, "Neutral, no strong directional inventory signal")

    # USD: strong dollar compresses spread (bearish for arb)
    if usd_elevated:
        scores["USD Strength"] = (-1, "Dollar elevated, broad headwind for crude spreads")
    else:
        scores["USD Strength"] = (+1, "Dollar not elevated, neutral to supportive")

    # US Production: rising output = WTI weakens = spread widens (mildly bullish for arb)
    if rig_pct > 0.65:
        scores["US Production"] = (+1, f"{latest_rig:,.0f} kbd, elevated, supports WTI discount to Brent")
    elif rig_pct < 0.35:
        scores["US Production"] = (-1, f"{latest_rig:,.0f} kbd, below average, US output tightening")
    else:
        scores["US Production"] = (0, f"{latest_rig:,.0f} kbd, near historical average")

    # Supply risk (OVX): elevated implied vol reflects a geopolitical/supply-shock
    # risk premium that historically shows up as a wider (not narrower) Brent
    # premium, since Brent reacts harder than WTI to international supply
    # disruption. Independent, market-priced data, so it doesn't double-count
    # the Cushing Inventory factor above.
    if ovx_pct > 0.70:
        scores["Supply Risk"] = (+1, f"OVX {latest_ovx:.0f}, elevated, crisis-level risk premium likely widening the spread")
    elif ovx_pct < 0.30:
        scores["Supply Risk"] = (-1, f"OVX {latest_ovx:.0f}, calm, no supply-shock risk premium in the spread")
    else:
        scores["Supply Risk"] = (0, f"OVX {latest_ovx:.0f}, near historical average")

    # Net margin: the arbitage itself
    if net_margin > 1.00:
        scores["Net Margin"] = (+1, f"${net_margin:.2f}/bbl above logistics cost, arb clearly open")
    elif net_margin > 0:
        scores["Net Margin"] = (0, f"${net_margin:.2f}/bbl, arb marginally open, execution risk high")
    else:
        scores["Net Margin"] = (-1, f"${net_margin:.2f}/bbl, arb closed at current logistics assumptions")

    total_score = sum(v[0] for v in scores.values())

    if total_score >= 3:
        signal    = "OPPORTUNITY"
        css_class = "signal-open"
        rationale = (
            f"Multiple factors align to support an open arbitrage. "
            f"The net margin stands at ${net_margin:.2f}/bbl after a ${total_cost:.2f}/bbl logistics assumption, "
            f"and the broader market structure, inventory dynamics and supply indicators, "
            f"is consistent with a sustained Brent premium. A transatlantic cargo warrants active evaluation."
        )
    elif total_score <= -2:
        signal    = "CLOSED"
        css_class = "signal-closed"
        rationale = (
            f"Market structure does not support the arbitrage at current cost assumptions. "
            f"The net margin of ${net_margin:.2f}/bbl is insufficient given prevailing headwinds. "
            f"Monitoring spread and cost dynamics is recommended; no cargo action indicated."
        )
    else:
        signal    = "MONITOR"
        css_class = "signal-monitor"
        rationale = (
            f"Conditions are mixed. The net margin of ${net_margin:.2f}/bbl sits near the threshold, "
            f"and underlying market signals are not yet strongly aligned in either direction. "
            f"Close monitoring of inventory releases and supply indicators is warranted before committing a cargo."
        )

    return {
        "signal":      signal,
        "css_class":   css_class,
        "rationale":   rationale,
        "scores":      scores,
        "total_score": total_score,
    }


# ── Sidebar ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Cost Assumptions")
    st.markdown("<div style='font-size:12px;color:#64748B;margin-bottom:12px'>Transatlantic cargo: USGC to NW Europe</div>", unsafe_allow_html=True)

    freight   = st.slider("Freight ($/bbl)",              0.50, 5.00, 2.00, 0.10)
    insurance = st.slider("Insurance ($/bbl)",             0.01, 0.50, 0.10, 0.01)
    port_fees = st.slider("Port fees & demurrage ($/bbl)", 0.10, 1.00, 0.30, 0.05)
    financing = st.slider("Financing & other ($/bbl)",     0.05, 1.00, 0.20, 0.05)
    cargo_sel = st.selectbox("Cargo size", ["VLCC (2,000,000 bbl)", "Suezmax (1,000,000 bbl)", "Aframax (600,000 bbl)"])

    total_cost = freight + insurance + port_fees + financing
    cargo_bbls = {"VLCC (2,000,000 bbl)": 2_000_000,
                  "Suezmax (1,000,000 bbl)": 1_000_000,
                  "Aframax (600,000 bbl)": 600_000}[cargo_sel]

    st.divider()
    st.markdown(f"**All-in cost: ${total_cost:.2f}/bbl**")
    st.markdown(f"Cargo: {cargo_bbls:,} bbl")

    st.divider()
    st.markdown("### Display")
    lookback = st.selectbox("Lookback", ["3 months","6 months","12 months","18 months"], index=2)
    lb_days  = {"3 months":90,"6 months":180,"12 months":365,"18 months":548}[lookback]
    model_lb = st.selectbox("Driver model window", ["6 months","12 months","18 months"], index=1)
    mlb_days = {"6 months":180,"12 months":365,"18 months":548}[model_lb]


# ── Load data ────────────────────────────────────────────────────
with st.spinner("Loading market data..."):
    try:
        data   = build_master()
        mdl    = build_driver_model(data, mlb_days)
        loaded = True
    except Exception as e:
        st.error(f"Data load failed: {e}")
        loaded = False

if not loaded:
    st.stop()

# ── Filter to lookback ───────────────────────────────────────────
cutoff  = datetime.today() - timedelta(days=lb_days)
prices  = data["prices"][data["prices"]["period"] >= cutoff].copy()
inv     = data["inventory"][data["inventory"]["period"] >= cutoff].copy()
cushing_df = data["cushing"][data["cushing"]["period"] >= cutoff].copy()
usd_df  = data["usd"][data["usd"]["period"] >= cutoff].copy()
rig_df  = data["rig"][data["rig"]["period"] >= cutoff].copy()
ovx_df  = data["ovx"][data["ovx"]["period"] >= cutoff].copy()
structure_df = data["structure"][data["structure"]["period"] >= cutoff].copy()

prices["net_margin"] = prices["gross_spread"] - total_cost
prices["arb_open"]   = prices["net_margin"] > 0

latest       = prices.iloc[-1]
cur_spread   = latest["gross_spread"]
cur_margin   = latest["net_margin"]
cur_brent    = latest["brent"]
cur_wti      = latest["wti"]
cur_date     = latest["period"].strftime("%d %b %Y")
arb_open_pct = prices["arb_open"].mean() * 100
cargo_pnl    = cur_margin * cargo_bbls

sig = compute_signal(data, cur_margin, total_cost)

# ── Page title ───────────────────────────────────────────────────
st.markdown("""
<div style='padding:4px 0 20px 0'>
  <div style='font-size:24px;font-weight:600;color:#0F172A'>Brent-WTI Physical Arbitrage Monitor</div>
  <div style='font-size:13px;color:#64748B;margin-top:4px'>
    Transatlantic crude arbitrage analysis: spread drivers, market structure and opportunity signal
  </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Snapshot",
    "Spread & Inventory",
    "Market Structure",
    "Driver Model & Signal"
])


# ════════════════════════════════════════════════════════════════
# TAB 1 — SNAPSHOT
# ════════════════════════════════════════════════════════════════
with tab1:
    st.markdown(f"<div style='font-size:12px;color:#94A3B8;margin-bottom:16px'>Last price data: {cur_date}</div>",
                unsafe_allow_html=True)

    # KPI row
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Brent", f"${cur_brent:.2f}")
    c2.metric("WTI",   f"${cur_wti:.2f}")
    c3.metric("Gross Spread", f"${cur_spread:.2f}")
    c4.metric("Logistics Cost", f"${total_cost:.2f}")
    arb_label = "Open" if cur_margin > 0 else "Closed"
    c5.metric("Net Margin", f"${cur_margin:.2f}", delta=arb_label,
              delta_color="normal" if cur_margin > 0 else "inverse")
    pnl_str = f"${cargo_pnl:,.0f}" if cargo_pnl >= 0 else f"-${abs(cargo_pnl):,.0f}"
    c6.metric(f"Cargo P&L ({cargo_sel.split()[0]})", pnl_str)

    st.divider()

    # Signal
    st.markdown(f"""
    <div class="{sig['css_class']}">
        <div class="signal-title">{sig['signal']}</div>
        <div class="signal-body">{sig['rationale']}</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Stats
    st.markdown("<div class='section-title'>Statistics, " + lookback + "</div>", unsafe_allow_html=True)
    s1,s2,s3,s4 = st.columns(4)
    s1.metric("Avg Gross Spread",  f"${prices['gross_spread'].mean():.2f}/bbl")
    s2.metric("Avg Net Margin",    f"${prices['net_margin'].mean():.2f}/bbl")
    s3.metric("Arb Open",          f"{arb_open_pct:.0f}% of trading days")
    s4.metric("Spread Volatility", f"${prices['gross_spread'].std():.2f}/bbl (1sd)")

    st.divider()

    # Cost breakdown
    st.markdown("<div class='section-title'>Logistics Cost Breakdown</div>", unsafe_allow_html=True)
    cost_fig = go.Figure(go.Bar(
        x=["Freight", "Insurance", "Port & Demurrage", "Financing"],
        y=[freight, insurance, port_fees, financing],
        marker_color=[C["brent"], C["spread"], C["curve"], C["usd"]],
        text=[f"${v:.2f}" for v in [freight, insurance, port_fees, financing]],
        textposition="outside",
    ))
    # "outside" bar labels need headroom above the tallest bar, otherwise the
    # default axis range (which tops out at the bar's own value) clips them.
    cost_max = max(freight, insurance, port_fees, financing)
    cost_fig.update_layout(
        **PLOT_LAYOUT,
        height=280,
        yaxis_title="$/bbl",
        showlegend=False,
        yaxis=dict(showgrid=True, gridcolor=C["grid"], range=[0, cost_max * 1.20]),
        xaxis=dict(showgrid=False),
    )
    st.plotly_chart(cost_fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# TAB 2 — SPREAD & INVENTORY
# ════════════════════════════════════════════════════════════════
with tab2:
    fig2 = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=("Brent & WTI Spot Prices ($/bbl)",
                        f"Net Arbitrage Margin after ${total_cost:.2f}/bbl logistics ($/bbl)",
                        "US Crude Inventory: National Aggregate (thousand barrels)"),
        vertical_spacing=0.07,
        row_heights=[0.35, 0.35, 0.30]
    )

    # Prices
    fig2.add_trace(go.Scatter(x=prices["period"], y=prices["brent"],
        name="Brent", line=dict(color=C["brent"], width=2)), row=1, col=1)
    fig2.add_trace(go.Scatter(x=prices["period"], y=prices["wti"],
        name="WTI", line=dict(color=C["wti"], width=2)), row=1, col=1)

    # Net margin — filled by sign
    pos = prices["net_margin"].clip(lower=0)
    neg = prices["net_margin"].clip(upper=0)
    fig2.add_trace(go.Scatter(x=prices["period"], y=pos, name="Arb Open",
        fill="tozeroy", fillcolor=C["open"],
        line=dict(color=C["open_line"], width=1.5)), row=2, col=1)
    fig2.add_trace(go.Scatter(x=prices["period"], y=neg, name="Arb Closed",
        fill="tozeroy", fillcolor=C["closed"],
        line=dict(color=C["closed_line"], width=1.5)), row=2, col=1)
    fig2.add_hline(y=0, line_dash="dash", line_color="#94A3B8", line_width=1, row=2, col=1)

    # Inventory
    fig2.add_trace(go.Scatter(x=inv["period"], y=inv["inventory_kb"],
        name="US Crude Stocks", line=dict(color=C["inv"], width=2),
        fill="tozeroy", fillcolor="rgba(14,165,233,0.12)"), row=3, col=1)

    fig2.update_layout(**PLOT_LAYOUT, height=740)
    fig2.update_yaxes(showgrid=True, gridcolor=C["grid"])
    fig2.update_xaxes(showgrid=False)
    # The "tozeroy" fill above forces the y-axis to include 0 by default, which
    # flattens genuine variation in a series that never gets near zero (stocks
    # sit around 700-850M bbl) — pin the visible range to the actual data span.
    inv_pad = (inv["inventory_kb"].max() - inv["inventory_kb"].min()) * 0.15 or 1
    fig2.update_yaxes(range=[inv["inventory_kb"].min() - inv_pad, inv["inventory_kb"].max() + inv_pad], row=3, col=1)
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(
        "National headline number (all PADDs). The driver model and market structure proxy use Cushing, OK "
        "stocks specifically, see the Market Structure and Driver Model & Signal tabs."
    )

    # Spread distribution
    st.markdown("<div class='section-title'>Spread Distribution</div>", unsafe_allow_html=True)
    da, db = st.columns(2)
    with da:
        hist = go.Figure(go.Histogram(
            x=prices["gross_spread"], nbinsx=40,
            marker_color=C["spread"], opacity=0.75
        ))
        # Stack these vertically (yshift) rather than relying on left/right
        # anchors, which collapse onto each other when the two lines land
        # close together on the x-axis.
        hist.add_vline(x=total_cost, line_dash="dash", line_color=C["closed_line"],
                       annotation_text=f"Breakeven ${total_cost:.2f}",
                       annotation_position="top",
                       annotation_font=dict(size=11, color=C["closed_line"]),
                       annotation_yshift=8)
        hist.add_vline(x=prices["gross_spread"].mean(), line_dash="dot", line_color=C["neutral"],
                       annotation_text=f"Mean ${prices['gross_spread'].mean():.2f}",
                       annotation_position="top",
                       annotation_font=dict(size=11, color=C["neutral"]),
                       annotation_yshift=26)
        # Extra top margin so the two stacked vline annotations have room
        # above the tallest bar without colliding with the chart title.
        hist_layout = {**PLOT_LAYOUT, "margin": dict(l=60, r=40, t=75, b=40)}
        hist.update_layout(**hist_layout, height=310,
                           xaxis_title="Gross Spread ($/bbl)", yaxis_title="Days",
                           showlegend=False, title="Gross Spread Distribution")
        st.plotly_chart(hist, use_container_width=True)

    with db:
        # Inventory vs spread scatter
        merged = pd.merge_asof(
            prices[["period","gross_spread"]].sort_values("period"),
            inv[["period","inventory_kb"]].sort_values("period"),
            on="period", direction="nearest"
        )
        scat = go.Figure(go.Scatter(
            x=merged["inventory_kb"], y=merged["gross_spread"],
            mode="markers",
            marker=dict(color=merged["gross_spread"], colorscale="RdYlGn",
                        size=4, opacity=0.55, showscale=True,
                        colorbar=dict(title="Spread", thickness=12)),
            text=merged["period"].dt.strftime("%d %b %Y"),
            hovertemplate="<b>%{text}</b><br>Stocks: %{x:,.0f} kb<br>Spread: $%{y:.2f}<extra></extra>"
        ))
        scat.update_layout(**PLOT_LAYOUT, height=310,
                           xaxis_title="US Crude Stocks (thousand bbl)",
                           yaxis_title="Gross Spread ($/bbl)",
                           showlegend=False,
                           title="US Inventories vs Brent-WTI Spread")
        st.plotly_chart(scat, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# TAB 3 — MARKET STRUCTURE
# ════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-title'>Market Structure Proxy (Estimated)</div>", unsafe_allow_html=True)
    st.info(
        "No real WTI futures curve is available (EIA discontinued the series in 2024). This chart is a "
        "**modeled estimate**: Cushing, OK crude stocks relative to their 26-week average, using theory of "
        "storage. Cushing stocks correlate with the actual spread at 0.84, versus 0.5 for national stocks.\n\n"
        "Limitation: it's inventory-only, so it can't anticipate a geopolitical shock. During the Feb-Apr 2026 "
        "Hormuz crisis, real WTI futures moved into steep backwardation while this proxy showed its loosest "
        "reading of the window. The Supply-Risk Premium chart below (OVX, real market data) covers that gap."
    )

    struct_fig = go.Figure()
    pos_struct = structure_df["structure_proxy"].clip(lower=0)
    neg_struct = structure_df["structure_proxy"].clip(upper=0)
    struct_fig.add_trace(go.Scatter(x=structure_df["period"], y=pos_struct,
        name="Tight (proxy)", fill="tozeroy",
        fillcolor="rgba(16,185,129,0.2)",
        line=dict(color=C["open_line"], width=1.5)))
    struct_fig.add_trace(go.Scatter(x=structure_df["period"], y=neg_struct,
        name="Loose (proxy)", fill="tozeroy",
        fillcolor="rgba(239,68,68,0.15)",
        line=dict(color=C["closed_line"], width=1.5)))
    struct_fig.add_hline(y=0, line_dash="dash", line_color="#94A3B8", line_width=1)

    struct_fig.update_layout(**PLOT_LAYOUT, height=280,
                            yaxis_title="Structure proxy (std. score)",
                            title="Estimated Market Tightness: Cushing Inventory Proxy")
    struct_fig.update_yaxes(showgrid=True, gridcolor=C["grid"])
    struct_fig.update_xaxes(showgrid=False)
    st.plotly_chart(struct_fig, use_container_width=True)

    # Supply-risk premium, real market data (OVX), not modeled
    st.markdown("<div class='section-title'>Supply-Risk Premium (OVX, Real Market Data)</div>", unsafe_allow_html=True)
    st.caption(
        "CBOE Crude Oil Volatility Index: options-implied volatility on crude. Real market data, not an "
        "estimate. Reacts to supply-shock risk immediately, unlike inventory data."
    )
    ovx_fig = go.Figure()
    ovx_fig.add_trace(go.Scatter(x=ovx_df["period"], y=ovx_df["ovx"],
        name="OVX", line=dict(color=C["closed_line"], width=2),
        fill="tozeroy", fillcolor="rgba(239,68,68,0.12)"))
    ovx_fig.add_hline(y=ovx_df["ovx"].quantile(0.70), line_dash="dash", line_color="#94A3B8", line_width=1,
                       annotation_text="70th pct (elevated)", annotation_font=dict(size=10))
    ovx_fig.update_layout(**PLOT_LAYOUT, height=260,
                          title="Oil Volatility Index (OVX)",
                          yaxis_title="Index", showlegend=False)
    ovx_fig.update_yaxes(showgrid=True, gridcolor=C["grid"])
    ovx_fig.update_xaxes(showgrid=False)
    st.plotly_chart(ovx_fig, use_container_width=True)

    # USD and rig count
    st.markdown("<div class='section-title'>Macro & Supply Context</div>", unsafe_allow_html=True)
    m1, m2 = st.columns(2)

    with m1:
        usd_fig = go.Figure()
        usd_fig.add_trace(go.Scatter(x=usd_df["period"], y=usd_df["usd"],
            name="USD Index", line=dict(color=C["usd"], width=2),
            fill="tozeroy", fillcolor="rgba(139,92,246,0.10)"))
        usd_fig.update_layout(**PLOT_LAYOUT, height=260,
                              title="Trade-Weighted USD Index",
                              yaxis_title="Index", showlegend=False)
        # "tozeroy" fill above forces the axis to include 0 by default; the
        # index only trades in the ~100-125 band, so that flattens real
        # movement. Pin the range to the actual data span instead.
        usd_pad = (usd_df["usd"].max() - usd_df["usd"].min()) * 0.15 or 1
        usd_fig.update_yaxes(showgrid=True, gridcolor=C["grid"],
                             range=[usd_df["usd"].min() - usd_pad, usd_df["usd"].max() + usd_pad])
        usd_fig.update_xaxes(showgrid=False)
        st.plotly_chart(usd_fig, use_container_width=True)

    with m2:
        rig_fig = go.Figure()
        rig_fig.add_trace(go.Scatter(x=rig_df["period"], y=rig_df["rig_count"],
            name="US Crude Production", line=dict(color=C["rig"], width=2),
            fill="tozeroy", fillcolor="rgba(236,72,153,0.10)"))
        rig_fig.update_layout(**PLOT_LAYOUT, height=260,
                              title="US Crude Production (thousand barrels/day)",
                              yaxis_title="kbd", showlegend=False)
        # Same "tozeroy" axis-flattening fix as the USD chart above.
        rig_pad = (rig_df["rig_count"].max() - rig_df["rig_count"].min()) * 0.15 or 1
        rig_fig.update_yaxes(showgrid=True, gridcolor=C["grid"],
                             range=[rig_df["rig_count"].min() - rig_pad, rig_df["rig_count"].max() + rig_pad])
        rig_fig.update_xaxes(showgrid=False)
        st.plotly_chart(rig_fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# TAB 4 — DRIVER MODEL & SIGNAL
# ════════════════════════════════════════════════════════════════
with tab4:
    if mdl is None:
        st.warning("Insufficient data for regression model. Try a longer model window.")
        st.stop()

    # Current driver readings
    st.markdown("<div class='section-title'>Current Driver Readings</div>", unsafe_allow_html=True)
    dr_cols = st.columns(len(mdl["current_drivers"]))
    for i, (name, info) in enumerate(mdl["current_drivers"].items()):
        coeff = info["coeff"]
        arrow = "+" if coeff > 0 else ""
        with dr_cols[i]:
            st.markdown(f"""
            <div class="factor-card">
                <div class="factor-label">{name}</div>
                <div class="factor-value">{info['value']}</div>
                <div class="factor-dir">{info['direction']}</div>
                <div style='font-size:11px;color:#94A3B8;margin-top:6px'>
                    Model coefficient: {arrow}{coeff:.3f}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # Regression results
    st.markdown(f"<div class='section-title'>Spread Driver Model, R² = {mdl['r2']:.3f} ({model_lb} window)</div>",
                unsafe_allow_html=True)

    r1, r2_col = st.columns([2, 1])

    with r1:
        # Actual vs fitted
        base_df = mdl["base"].copy()
        base_df["fitted"] = mdl["y_pred"]
        avf = go.Figure()
        avf.add_trace(go.Scatter(x=base_df["period"], y=base_df["gross_spread"],
            name="Actual Spread", line=dict(color=C["spread"], width=1.5)))
        avf.add_trace(go.Scatter(x=base_df["period"], y=base_df["fitted"],
            name="Model Fitted", line=dict(color=C["curve"], width=1.5, dash="dash")))
        avf.update_layout(**PLOT_LAYOUT, height=300,
                          yaxis_title="Spread ($/bbl)",
                          title="Actual Brent-WTI Spread vs Model Fit")
        avf.update_yaxes(showgrid=True, gridcolor=C["grid"])
        avf.update_xaxes(showgrid=False)
        st.plotly_chart(avf, use_container_width=True)

    with r2_col:
        # Coefficient bar chart
        names  = list(mdl["coeffs"].keys())
        vals   = list(mdl["coeffs"].values())
        colors = [C["open_line"] if v > 0 else C["closed_line"] for v in vals]
        coef_fig = go.Figure(go.Bar(
            x=vals, y=names, orientation="h",
            marker_color=colors,
            text=[f"{v:+.3f}" for v in vals], textposition="outside"
        ))
        coef_layout = {**PLOT_LAYOUT, "margin": dict(l=20, r=60, t=50, b=40)}
        coef_fig.update_layout(**coef_layout, height=300,
                               xaxis_title="Standardised Coefficient",
                               title="Factor Coefficients",
                               showlegend=False)
        # Give "outside" bar labels room on both sides — narrow negative bars
        # (e.g. USD) otherwise get their text clipped by the plot edge in this
        # narrow column.
        coef_pad = max(abs(min(vals)), abs(max(vals))) * 0.5 or 1
        coef_fig.update_xaxes(showgrid=True, gridcolor=C["grid"],
                              range=[min(vals) - coef_pad, max(vals) + coef_pad])
        coef_fig.update_yaxes(showgrid=False)
        st.plotly_chart(coef_fig, use_container_width=True)

    # Rolling correlations
    st.markdown("<div class='section-title'>Rolling 90-Day Correlations with Brent-WTI Spread</div>",
                unsafe_allow_html=True)
    corr_colors = {"Cushing Inventory": C["inv"], "USD": C["usd"], "US Production": C["rig"], "OVX": C["closed_line"]}
    corr_fig = go.Figure()
    for label, series in mdl["roll_corr"].items():
        idx = mdl["base"]["period"].values
        corr_fig.add_trace(go.Scatter(
            x=idx, y=series.values, name=label,
            line=dict(color=corr_colors.get(label, C["neutral"]), width=1.8)
        ))
    corr_fig.add_hline(y=0, line_dash="dash", line_color="#CBD5E1", line_width=1)
    corr_fig.update_layout(**PLOT_LAYOUT, height=300,
                           yaxis_title="Correlation coefficient",
                           yaxis=dict(range=[-1,1], showgrid=True, gridcolor=C["grid"]),
                           xaxis=dict(showgrid=False))
    st.plotly_chart(corr_fig, use_container_width=True)

    st.divider()

    # Signal breakdown
    st.markdown("<div class='section-title'>Opportunity Signal: Factor Breakdown</div>",
                unsafe_allow_html=True)

    sig_cols = st.columns(len(sig["scores"]))
    score_colors = {1: "#10B981", 0: "#F59E0B", -1: "#EF4444"}
    score_labels = {1: "Bullish", 0: "Neutral", -1: "Bearish"}

    for i, (factor, (score, desc)) in enumerate(sig["scores"].items()):
        with sig_cols[i]:
            color = score_colors[score]
            label = score_labels[score]
            st.markdown(f"""
            <div style='background:#fff;border:1px solid #E2E8F0;border-top:3px solid {color};
                        border-radius:8px;padding:12px 14px;'>
                <div style='font-size:10px;font-weight:600;text-transform:uppercase;
                            letter-spacing:.06em;color:#64748B;margin-bottom:6px'>{factor}</div>
                <div style='font-size:14px;font-weight:600;color:{color};margin-bottom:4px'>{label}</div>
                <div style='font-size:11px;color:#64748B;line-height:1.5'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # Overall score gauge. Range must track the number of scored factors —
    # each factor contributes -1/0/+1, so the max possible magnitude is
    # len(scores), not a fixed constant.
    max_score = len(sig["scores"])
    st.markdown("<br>", unsafe_allow_html=True)
    # Rendered as HTML instead of Plotly's native indicator title, so it
    # matches the factor-card label style exactly (Plotly's title font API
    # has no uppercase/letter-spacing/weight controls).
    st.markdown(
        "<div style='font-size:14px;font-weight:600;text-transform:uppercase;"
        "letter-spacing:.06em;color:#64748B;text-align:center'>Composite Signal Score</div>",
        unsafe_allow_html=True
    )
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=sig["total_score"],
        number={"font": {"size": 28}},
        gauge={
            "axis": {"range": [-max_score, max_score], "tickvals": list(range(-max_score, max_score + 1))},
            "bar": {"color": score_colors.get(
                1 if sig["total_score"] > 0 else (-1 if sig["total_score"] < 0 else 0), C["neutral"])},
            "steps": [
                {"range": [-max_score, -2], "color": "rgba(239,68,68,0.15)"},
                {"range": [-2, 3],          "color": "rgba(245,158,11,0.12)"},
                {"range": [3, max_score],   "color": "rgba(16,185,129,0.15)"},
            ],
            "threshold": {"line": {"color": "#0F172A", "width": 2}, "value": 0}
        }
    ))
    gauge_fig.update_layout(height=280, margin=dict(l=40,r=40,t=30,b=20),
                            paper_bgcolor="#fff", font=dict(family="Inter, Arial, sans-serif"))
    # Reuses the same N-column grid as the factor cards above (rather than a
    # CSS-centering trick, which Streamlit's own styles were overriding) and
    # places the gauge in the middle slot with use_container_width=True, so
    # it's aligned under the middle card by construction, not by guesswork.
    # The middle slot is widened (symmetric ratios, so the center point is
    # unchanged) purely to give the gauge more room to render at a larger size.
    n = len(sig["scores"])
    mid = n // 2
    ratios = [1] * n
    ratios[mid] = 3
    gauge_cols = st.columns(ratios)
    with gauge_cols[mid]:
        st.plotly_chart(gauge_fig, use_container_width=True)

    st.divider()

    # Explainer
    with st.expander("Methodology"):
        st.markdown(f"""
**Spread Driver Model**

OLS regression over the selected window ({model_lb}) on four standardized drivers:

| Driver | Reason |
|--------|-----------|
| Cushing Crude Inventory | Correlates with the spread at 0.84 (national stocks: 0.5) |
| Trade-Weighted USD | Stronger dollar compresses commodity spreads |
| US Production (kbd) | Rising output weighs on WTI relative to Brent |
| Oil Volatility (OVX) | Market-priced supply-risk premium, independent of inventory |

R² = **{mdl['r2']:.3f}** ({mdl['r2']*100:.0f}% of spread variance explained).

OVX and Cushing inventory correlate at 0.77, so their individual coefficients can shift, or flip sign,
between windows. Read the two together as one combined signal, not fully separable effects.

Tested and excluded, no R² gain once Cushing inventory and OVX are in the model: US crude exports
(WCREXUS2), refinery utilization (WPULEUS3), freight PPI. Tanker freight rates and Brent CFDs aren't
available through any free API.

**Opportunity Signal**

Five factors scored independently (+1 bullish, 0 neutral, -1 bearish): Net Margin, Cushing Inventory,
USD Strength, US Production, Supply Risk (OVX). Summed:

- Total ≥ +3: **Opportunity**
- Total ≤ -2: **Closed**
- Otherwise: **Monitor**

**Market Structure Proxy**

Estimates backwardation/contango risk from Cushing stocks relative to their trailing 26-week average.
Not real futures data, and inventory-only, so it misses geopolitical shocks. The OVX chart alongside it
is real market data and covers that gap.

**Data Sources**
- Brent & WTI spot prices: EIA (daily)
- US crude inventory (national & Cushing, OK): EIA Weekly Petroleum Status Report
- Trade-weighted USD index (DTWEXBGS): Federal Reserve / FRED
- US crude production (WCRFPUS2): EIA Weekly Petroleum Status Report
- Crude Oil Volatility Index (OVXCLS): CBOE via Federal Reserve / FRED
        """)

st.caption(f"Data: EIA, Federal Reserve (FRED)  |  Built by Nicholas Becnel  |  {datetime.today().strftime('%d %b %Y')}")
