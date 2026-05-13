"""
STEP 5 — Advanced Retail Intelligence Dashboard
Run: streamlit run app/streamlit_app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os, sys
from datetime import date

# ── path setup ──────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ── rerun compatibility ──────────────────────────────────────────────────────────
def _rerun():
    if hasattr(st, "rerun"):      st.rerun()
    else:                          st.experimental_rerun()

# ── page config ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RetailPulse AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@400;600;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; background: #0a0a0f; color: #e8e4dc; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.2rem 1.8rem; max-width: 100%; }
.hero { background: linear-gradient(135deg,#0d1117,#131b2e,#0d1117); border: 1px solid #1e2d45;
        border-radius: 14px; padding: 1.4rem 2rem; margin-bottom: 1.2rem;
        display: flex; justify-content: space-between; align-items: center; }
.hero-title { font-size: 2rem; font-weight: 800; color: #fff; }
.hero-sub { font-family: 'Space Mono',monospace; font-size: 0.68rem; color: #4a9eff;
            letter-spacing: 3px; text-transform: uppercase; margin-top: 4px; }
.hero-badge { background: #4a9eff22; border: 1px solid #4a9eff55; color: #4a9eff;
              border-radius: 8px; padding: 0.4rem 0.9rem;
              font-family: 'Space Mono',monospace; font-size: 0.72rem; }
.kpi-card { background: #0d1117; border: 1px solid #1e2d45; border-radius: 12px;
            padding: 1.1rem 1.3rem; position: relative; overflow: hidden; }
.kpi-card::before { content:''; position:absolute; top:0; left:0; right:0; height:3px;
                    background: var(--accent); border-radius:12px 12px 0 0; }
.kpi-label { font-family:'Space Mono',monospace; font-size:0.62rem; color:#5a6a80;
             letter-spacing:2px; text-transform:uppercase; margin-bottom:4px; }
.kpi-value { font-size:1.75rem; font-weight:800; color:#e8e4dc; line-height:1; }
.kpi-delta { font-family:'Space Mono',monospace; font-size:0.68rem; margin-top:3px; }
.alert-crit { background:#1f0d0d; border:1px solid #ff4a4a55; border-left:4px solid #ff4a4a;
              border-radius:8px; padding:0.7rem 1rem; margin-bottom:0.4rem;
              font-family:'Space Mono',monospace; font-size:0.73rem; color:#ff9a9a; }
.alert-warn { background:#1f1600; border:1px solid #ffb84a55; border-left:4px solid #ffb84a;
              border-radius:8px; padding:0.7rem 1rem; margin-bottom:0.4rem;
              font-family:'Space Mono',monospace; font-size:0.73rem; color:#ffd88a; }
.alert-ok   { background:#0d1f12; border:1px solid #4aff8055; border-left:4px solid #4aff80;
              border-radius:8px; padding:0.7rem 1rem; margin-bottom:0.4rem;
              font-family:'Space Mono',monospace; font-size:0.73rem; color:#9affb8; }
.alert-dead { background:#111118; border:1px solid #8888ff44; border-left:4px solid #8888ff;
              border-radius:8px; padding:0.7rem 1rem; margin-bottom:0.4rem;
              font-family:'Space Mono',monospace; font-size:0.73rem; color:#bbbbff; }
.sec-label  { font-family:'Space Mono',monospace; font-size:0.62rem; letter-spacing:3px;
              text-transform:uppercase; color:#4a9eff; margin-bottom:0.3rem; }
.sec-title  { font-size:1.05rem; font-weight:700; color:#e8e4dc; margin-bottom:0.8rem; }
div[data-testid="stMetric"] label { font-family:'Space Mono',monospace !important; font-size:0.65rem !important; }
</style>
""", unsafe_allow_html=True)

PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0d1117",
    font=dict(color="#9abadc", size=11),
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis=dict(gridcolor="#1a2535", linecolor="#1e2d45"),
    yaxis=dict(gridcolor="#1a2535", linecolor="#1e2d45"),
)

# ── auto-run pipeline if data missing ───────────────────────────────────────────
def ensure_pipeline():
    needed = [
        "outputs/inventory_intelligence.csv",
        "data/forecasts/forecast_15days.csv",
        "outputs/trend_scores.csv",
    ]
    missing = [p for p in needed if not os.path.exists(os.path.join(ROOT, p))]
    if missing:
        with st.spinner("⚡ First launch — running full ML pipeline (~3 min)…"):
            os.chdir(ROOT)
            from src.data_generation import generate_all
            from src.data_cleaning import clean_sales, clean_inventory, clean_weather
            from src.feature_engineering import build_daily_product_features
            from src.forecasting import train_forecast_model, generate_15day_forecast
            from src.inventory_optimizer import compute_inventory_intelligence
            from src.intelligence_engine import (compute_trend_scores, compute_velocity_table,
                                                  compute_seasonal_summary, compute_weather_impact)
            for d in ["data/raw","data/processed","data/forecasts","models","outputs"]:
                os.makedirs(d, exist_ok=True)
            cat, wth, sal, inv = generate_all()
            sc = clean_sales(sal); ic = clean_inventory(inv); wc = clean_weather(wth)
            sc.to_csv("data/processed/cleaned_sales.csv", index=False)
            ic.to_csv("data/processed/cleaned_inventory.csv", index=False)
            wc.to_csv("data/processed/cleaned_weather.csv", index=False)
            feat = build_daily_product_features(sc, ic, wc, cat)
            feat.to_csv("data/processed/featured_data.csv", index=False)
            mdl, fcols, _, _ = train_forecast_model(feat)
            fc = generate_15day_forecast(feat, mdl, fcols)
            fc.to_csv("data/forecasts/forecast_15days.csv", index=False)
            rec = compute_inventory_intelligence(sc, ic, fc, cat)
            rec.to_csv("outputs/inventory_intelligence.csv", index=False)
            compute_trend_scores(feat).to_csv("outputs/trend_scores.csv", index=False)
            compute_velocity_table(feat, rec).to_csv("outputs/velocity_table.csv", index=False)
            compute_seasonal_summary(sc).to_csv("outputs/seasonal_summary.csv", index=False)
            compute_weather_impact(feat).to_csv("outputs/weather_impact.csv", index=False)
        st.success("Pipeline complete!")
        _rerun()

os.chdir(ROOT)
ensure_pipeline()

# ── load all data ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_all():
    sales    = pd.read_csv("data/processed/cleaned_sales.csv",       parse_dates=["date"])
    forecast = pd.read_csv("data/forecasts/forecast_15days.csv",      parse_dates=["forecast_date"])
    inv_rec  = pd.read_csv("outputs/inventory_intelligence.csv")
    trends   = pd.read_csv("outputs/trend_scores.csv")
    velocity = pd.read_csv("outputs/velocity_table.csv")
    seasonal = pd.read_csv("outputs/seasonal_summary.csv")
    weather  = pd.read_csv("outputs/weather_impact.csv")
    catalog  = pd.read_csv("data/raw/product_catalog.csv")
    inv_log  = pd.read_csv("data/processed/cleaned_inventory.csv",   parse_dates=["date"])
    feat     = pd.read_csv("data/processed/featured_data.csv",        parse_dates=["date"])
    return sales, forecast, inv_rec, trends, velocity, seasonal, weather, catalog, inv_log, feat

sales, forecast, inv_rec, trends, velocity, seasonal, weather_impact, catalog, inv_log, feat = load_all()
categories = ["All"] + sorted(sales["category"].unique().tolist())
products   = sorted(inv_rec["product_id"].unique().tolist())

# ── SIDEBAR ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Filters")
    sel_cat = st.selectbox("Category", categories)
    sel_prod = st.selectbox("Product (detail view)", products)
    st.markdown("---")
    st.markdown("### 🔧 Inventory Settings")
    service_level = st.slider("Service Level %", 80, 99, 95) / 100
    lead_time     = st.slider("Lead Time (days)", 1, 30, 7)
    st.markdown("---")
    st.caption("RetailPulse AI · XGBoost Engine")

# ── HERO ─────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <div>
    <div class="hero-title">RetailPulse AI ⚡</div>
    <div class="hero-sub">Forecasting · Inventory · Trend Intelligence</div>
  </div>
  <div class="hero-badge">LIVE · {date.today().strftime("%d %b %Y")}</div>
</div>
""", unsafe_allow_html=True)

# ── KPI STRIP ────────────────────────────────────────────────────────────────────
total_fc     = int(forecast["forecast_sales"].sum())
critical     = int((inv_rec["alert_status"].str.contains("REORDER|OUT OF")).sum())
dead_count   = int(inv_rec["is_dead_stock"].sum())
over_count   = int(inv_rec["is_overstock"].sum())
hot_count    = int((trends["trend_label"] == "Hot 🔥").sum())
total_stock  = int(inv_rec["current_stock"].sum())

k1,k2,k3,k4,k5,k6 = st.columns(6)
def kpi(col, label, val, delta, color):
    col.markdown(f"""<div class="kpi-card" style="--accent:{color}">
    <div class="kpi-label">{label}</div>
    <div class="kpi-value">{val}</div>
    <div class="kpi-delta" style="color:{color}">{delta}</div>
    </div>""", unsafe_allow_html=True)

kpi(k1, "15-Day Forecast",  f"{total_fc:,}", "All products combined",   "#4a9eff")
kpi(k2, "Critical Alerts",  critical,        "Reorder + out-of-stock",  "#ff4a4a")
kpi(k3, "Dead Stock SKUs",  dead_count,      "30+ days no sale",        "#8888ff")
kpi(k4, "Overstock SKUs",   over_count,      "90+ days of supply",      "#ffb84a")
kpi(k5, "Trending Hot 🔥",  hot_count,       "Top momentum products",   "#4aff80")
kpi(k6, "Total Units In Stock", f"{total_stock:,}", "Across all SKUs",  "#ff88cc")

st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════════
# TAB LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📦 Storage Overview",
    "🔮 15-Day Forecast",
    "📈 Trend & Velocity",
    "🌤 Season & Weather",
    "🏷 Category Analysis",
    "🚨 Alerts & Restock",
])

# ─────────────────────────────────────────────────────────────────────────────────
# TAB 1: STORAGE OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="sec-label">Current Inventory State</div>', unsafe_allow_html=True)

    # Filter by category
    disp = inv_rec.copy()
    if sel_cat != "All":
        disp = disp[disp["category"] == sel_cat]

    # Stock level bar chart
    fig = go.Figure()
    colors = disp["alert_status"].map({
        "🟢 OK": "#4aff80", "🔴 OUT OF STOCK": "#ff4a4a",
        "🟠 REORDER NOW": "#ffb84a", "⚫ DEAD STOCK": "#8888ff",
        "🔵 OVERSTOCK": "#4a9eff",
    }).fillna("#666")

    fig.add_trace(go.Bar(
        x=disp["product_id"], y=disp["current_stock"],
        marker_color=list(colors),
        text=disp["alert_status"], textposition="outside",
        hovertemplate="<b>%{x}</b><br>Stock: %{y}<br>%{text}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=disp["product_id"], y=disp["reorder_point"],
        mode="markers", marker=dict(color="#ff4a4a", symbol="line-ew", size=10, line=dict(width=2, color="#ff4a4a")),
        name="Reorder Point", hovertemplate="Reorder point: %{y:.0f}<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_THEME, title="Current Stock vs Reorder Point", height=380,
                      showlegend=True, bargap=0.3)
    st.plotly_chart(fig, use_container_width=True)

    # Stock detail table
    st.markdown('<div class="sec-label">Inventory Detail Table</div>', unsafe_allow_html=True)
    tbl = disp[[
        "product_id","product_name","category","current_stock",
        "reorder_point","safety_stock","days_to_stockout",
        "last_sold_date","days_since_sold","last_restock_date","alert_status"
    ]].copy()
    tbl["days_to_stockout"] = tbl["days_to_stockout"].apply(
        lambda x: "∞" if x >= 999 else f"{x:.0f}d"
    )
    tbl = tbl.rename(columns={
        "product_id": "SKU", "product_name": "Name", "category": "Category",
        "current_stock": "Stock", "reorder_point": "Reorder Pt",
        "safety_stock": "Safety Stk", "days_to_stockout": "Days Left",
        "last_sold_date": "Last Sold", "days_since_sold": "Days Unsold",
        "last_restock_date": "Last Restock", "alert_status": "Status",
    })

    def color_status(val):
        m = {"🟢 OK": "color:#4aff80", "🔴 OUT OF STOCK": "color:#ff4a4a",
             "🟠 REORDER NOW": "color:#ffb84a", "⚫ DEAD STOCK": "color:#8888ff",
             "🔵 OVERSTOCK": "color:#4a9eff"}
        return m.get(val, "")

    # FIX: .applymap() was renamed to .map() in pandas >= 2.1
    try:
        styled = tbl.style.map(color_status, subset=["Status"])
    except AttributeError:
        styled = tbl.style.applymap(color_status, subset=["Status"])

    st.dataframe(styled, use_container_width=True, height=350)

    # Days-to-stockout gauge strip
    st.markdown('<div class="sec-label" style="margin-top:1rem">Days to Stockout (top urgency)</div>', unsafe_allow_html=True)
    urgent = disp[disp["days_to_stockout"] < 30].sort_values("days_to_stockout").head(10)
    if len(urgent):
        fig2 = go.Figure(go.Bar(
            x=urgent["days_to_stockout"], y=urgent["product_id"],
            orientation="h",
            marker_color=urgent["days_to_stockout"].apply(
                lambda v: "#ff4a4a" if v < 7 else "#ffb84a" if v < 15 else "#4aff80"
            ),
            text=urgent["days_to_stockout"].apply(lambda v: f"{v:.0f} days"),
            textposition="outside",
        ))
        fig2.update_layout(**PLOTLY_THEME, height=300, xaxis_title="Days remaining")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.success("✅ No products at critical stockout risk in the next 30 days.")

# ─────────────────────────────────────────────────────────────────────────────────
# TAB 2: 15-DAY FORECAST
# ─────────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="sec-label">XGBoost 15-Day Sales Forecast</div>', unsafe_allow_html=True)

    fc = forecast.copy()
    if sel_cat != "All":
        fc = fc[fc["category"] == sel_cat]

    # Top 10 products by total forecast
    top_fc = fc.groupby(["product_id","category"])["forecast_sales"].sum().reset_index()
    top_fc = top_fc.sort_values("forecast_sales", ascending=False).head(15)

    fig = px.bar(top_fc, x="product_id", y="forecast_sales", color="category",
                 title="Top Products — Total Forecast Units (next 15 days)",
                 color_discrete_sequence=px.colors.qualitative.Bold,
                 labels={"product_id": "SKU", "forecast_sales": "Forecast Units"})
    fig.update_layout(**PLOTLY_THEME, height=350)
    st.plotly_chart(fig, use_container_width=True)

    # Per-product daily forecast line
    st.markdown('<div class="sec-label">Daily Forecast — Selected Product</div>', unsafe_allow_html=True)
    prod_fc = forecast[forecast["product_id"] == sel_prod].copy()

    # Historical for context (last 60 days)
    hist = (
        sales[sales["product_id"] == sel_prod]
        .groupby("date")["quantity"].sum().reset_index()
    )
    hist["date"] = pd.to_datetime(hist["date"])
    hist_tail = hist.tail(60)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=hist_tail["date"], y=hist_tail["quantity"],
        mode="lines", name="Historical",
        line=dict(color="#4a9eff", width=2),
        fill="tozeroy", fillcolor="rgba(74,158,255,0.08)"
    ))
    if len(prod_fc):
        jitter = prod_fc["forecast_sales"].std() * 0.2
        fig3.add_trace(go.Scatter(
            x=pd.concat([prod_fc["forecast_date"], prod_fc["forecast_date"][::-1]]),
            y=pd.concat([prod_fc["forecast_sales"] + jitter, (prod_fc["forecast_sales"] - jitter)[::-1]]),
            fill="toself", fillcolor="rgba(255,184,74,0.12)",
            line=dict(color="rgba(255,184,74,0)"), name="Confidence band"
        ))
        fig3.add_trace(go.Scatter(
            x=prod_fc["forecast_date"], y=prod_fc["forecast_sales"],
            mode="lines+markers", name="Forecast",
            line=dict(color="#ffb84a", width=2.5, dash="dot"),
            marker=dict(size=6)
        ))
    fig3.add_vline(x=str(hist_tail["date"].max()), line_color="#2a3a4a", line_dash="dash")
    fig3.update_layout(**PLOTLY_THEME, title=f"{sel_prod} — Historical + 15-Day Forecast", height=340)
    st.plotly_chart(fig3, use_container_width=True)

    # Forecast table
    st.dataframe(
        prod_fc[["forecast_date","forecast_sales","day_offset"]].rename(columns={
            "forecast_date": "Date", "forecast_sales": "Forecast Units", "day_offset": "Day"
        }),
        use_container_width=True, height=200
    )

# ─────────────────────────────────────────────────────────────────────────────────
# TAB 3: TREND & VELOCITY
# ─────────────────────────────────────────────────────────────────────────────────
with tab3:
    left, right = st.columns([1.2, 1])

    with left:
        st.markdown('<div class="sec-label">Product Momentum</div>', unsafe_allow_html=True)
        tr = trends.copy()
        if sel_cat != "All":
            tr = tr[tr["category"] == sel_cat]

        fig = px.bar(
            tr.head(20), x="momentum_pct", y="product_id", orientation="h",
            color="momentum_pct",
            color_continuous_scale=["#ff4a4a","#ffb84a","#4aff80"],
            title="Momentum % vs 30 days ago",
            labels={"momentum_pct": "Momentum %", "product_id": "SKU"},
        )
        fig.update_layout(**PLOTLY_THEME, height=420, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown('<div class="sec-label">Selling Velocity (units/day)</div>', unsafe_allow_html=True)
        vel = velocity.copy()
        if sel_cat != "All":
            vel = vel[vel["category"] == sel_cat]

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="7-day avg",  x=vel["product_id"], y=vel["velocity_7d"],  marker_color="#4a9eff"))
        fig2.add_trace(go.Bar(name="30-day avg", x=vel["product_id"], y=vel["velocity_30d"], marker_color="rgba(74, 255, 128, 0.53)"))
        fig2.update_layout(**PLOTLY_THEME, title="Units Sold per Day", barmode="group", height=420)
        st.plotly_chart(fig2, use_container_width=True)

    # Trend label distribution
    st.markdown("---")
    st.markdown('<div class="sec-label">Trend Label Distribution</div>', unsafe_allow_html=True)
    tl_counts = trends["trend_label"].value_counts().reset_index()
    tl_counts.columns = ["Trend", "Count"]
    fig3 = px.pie(tl_counts, names="Trend", values="Count",
                  color_discrete_sequence=["#ff4a4a","#ffb84a","#4aff80","#4a9eff","#ff88cc"],
                  title="Products by Trend Status")
    fig3.update_layout(**PLOTLY_THEME, height=300)
    st.plotly_chart(fig3, use_container_width=True)

    # Revenue per day table
    st.markdown('<div class="sec-label">Revenue per Day (30-day avg)</div>', unsafe_allow_html=True)
    rev_tbl = vel[["product_id","category","velocity_7d","velocity_30d","revenue_per_day","days_to_stockout"]].copy()
    rev_tbl["revenue_per_day"] = rev_tbl["revenue_per_day"].apply(lambda x: f"₹{x:,.0f}")
    st.dataframe(rev_tbl, use_container_width=True, height=250)

# ─────────────────────────────────────────────────────────────────────────────────
# TAB 4: SEASON & WEATHER
# ─────────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="sec-label">Seasonal Sales Heatmap (Category × Month)</div>', unsafe_allow_html=True)

    # Pivot for heatmap
    seas = seasonal.copy()
    pivot = seas.pivot_table(index="category", columns="month_name",
                              values="total_units", aggfunc="sum")
    month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    pivot = pivot.reindex(columns=[m for m in month_order if m in pivot.columns])

    fig = px.imshow(pivot, color_continuous_scale="Blues",
                    title="Total Units Sold by Category × Month",
                    labels={"color": "Units"}, aspect="auto")
    fig.update_layout(**PLOTLY_THEME, height=380)
    st.plotly_chart(fig, use_container_width=True)

    left2, right2 = st.columns(2)

    with left2:
        st.markdown('<div class="sec-label">Monthly Revenue Trend</div>', unsafe_allow_html=True)
        rev_cat = seas.groupby(["month_name","category"])["total_revenue"].sum().reset_index()
        if sel_cat != "All":
            rev_cat = rev_cat[rev_cat["category"] == sel_cat]
        rev_cat["month_num"] = rev_cat["month_name"].apply(lambda m: month_order.index(m) if m in month_order else 0)
        rev_cat = rev_cat.sort_values("month_num")
        fig2 = px.line(rev_cat, x="month_name", y="total_revenue", color="category",
                       title="Revenue by Month", markers=True,
                       color_discrete_sequence=px.colors.qualitative.Bold)
        fig2.update_layout(**PLOTLY_THEME, height=320)
        st.plotly_chart(fig2, use_container_width=True)

    with right2:
        st.markdown('<div class="sec-label">Weather Impact on Sales</div>', unsafe_allow_html=True)
        wi = weather_impact.copy()
        if sel_cat != "All":
            wi = wi[wi["category"] == sel_cat]
        fig3 = px.bar(wi, x="weather_feature", y="correlation", color="category",
                      barmode="group", title="Sales vs Weather Correlation",
                      color_discrete_sequence=px.colors.qualitative.Bold,
                      labels={"correlation": "Pearson r", "weather_feature": "Feature"})
        fig3.add_hline(y=0, line_color="#2a3a4a")
        fig3.update_layout(**PLOTLY_THEME, height=320)
        st.plotly_chart(fig3, use_container_width=True)

    # Temperature overlay
    st.markdown('<div class="sec-label">Daily Sales vs Temperature — Selected Product</div>', unsafe_allow_html=True)
    prod_feat = feat[feat["product_id"] == sel_prod].sort_values("date").tail(120)
    if len(prod_feat) > 0:
        fig4 = make_subplots(specs=[[{"secondary_y": True}]])
        fig4.add_trace(go.Scatter(x=prod_feat["date"], y=prod_feat["daily_sales"],
                                   name="Daily Sales", line=dict(color="#4a9eff", width=2)), secondary_y=False)
        fig4.add_trace(go.Scatter(x=prod_feat["date"], y=prod_feat["temp_c"],
                                   name="Temp °C", line=dict(color="#ff6b6b", width=1.5, dash="dot")), secondary_y=True)
        fig4.update_layout(**PLOTLY_THEME, title=f"{sel_prod} — Sales vs Temperature", height=300)
        fig4.update_yaxes(title_text="Sales", secondary_y=False)
        fig4.update_yaxes(title_text="Temp °C", secondary_y=True)
        st.plotly_chart(fig4, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────────
# TAB 5: CATEGORY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────────
with tab5:
    # Category share
    cat_sales = sales.groupby("category").agg(
        total_units=("quantity","sum"),
        total_revenue=("total_spent","sum"),
        n_transactions=("transaction_id","count"),
    ).reset_index()

    c1, c2, c3 = st.columns(3)
    with c1:
        fig = px.pie(cat_sales, names="category", values="total_revenue",
                     title="Revenue Share by Category",
                     color_discrete_sequence=px.colors.qualitative.Bold)
        fig.update_layout(**PLOTLY_THEME, height=320)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = px.bar(cat_sales.sort_values("total_units", ascending=True),
                      x="total_units", y="category", orientation="h",
                      title="Total Units Sold by Category",
                      color="total_units", color_continuous_scale="Blues")
        fig2.update_layout(**PLOTLY_THEME, height=320, coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)
    with c3:
        fig3 = px.bar(cat_sales.sort_values("n_transactions", ascending=True),
                      x="n_transactions", y="category", orientation="h",
                      title="Transaction Count by Category",
                      color="n_transactions", color_continuous_scale="Greens")
        fig3.update_layout(**PLOTLY_THEME, height=320, coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

    # Weekly pattern
    st.markdown("---")
    st.markdown('<div class="sec-label">Day-of-Week Sales Pattern</div>', unsafe_allow_html=True)
    dow = sales.copy()
    dow["dayofweek"] = pd.to_datetime(dow["date"]).dt.day_name()
    dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    dow_agg = dow.groupby(["dayofweek","category"])["quantity"].mean().reset_index()
    dow_agg["dayofweek"] = pd.Categorical(dow_agg["dayofweek"], categories=dow_order, ordered=True)
    dow_agg = dow_agg.sort_values("dayofweek")
    if sel_cat != "All":
        dow_agg = dow_agg[dow_agg["category"] == sel_cat]

    fig4 = px.line(dow_agg, x="dayofweek", y="quantity", color="category",
                   title="Avg Units Sold per Day of Week", markers=True,
                   color_discrete_sequence=px.colors.qualitative.Bold)
    fig4.update_layout(**PLOTLY_THEME, height=320)
    st.plotly_chart(fig4, use_container_width=True)

    # Payment method
    st.markdown('<div class="sec-label">Payment Method Distribution</div>', unsafe_allow_html=True)
    pay = sales.groupby(["payment_method","category"])["quantity"].sum().reset_index()
    if sel_cat != "All":
        pay = pay[pay["category"] == sel_cat]
    fig5 = px.bar(pay, x="payment_method", y="quantity", color="category",
                  barmode="stack", title="Units by Payment Method",
                  color_discrete_sequence=px.colors.qualitative.Bold)
    fig5.update_layout(**PLOTLY_THEME, height=300)
    st.plotly_chart(fig5, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────────
# TAB 6: ALERTS & RESTOCK PLANNER
# ─────────────────────────────────────────────────────────────────────────────────
with tab6:
    st.markdown('<div class="sec-label">🚨 Active Alerts</div>', unsafe_allow_html=True)

    disp_rec = inv_rec.copy()
    if sel_cat != "All":
        disp_rec = disp_rec[disp_rec["category"] == sel_cat]

    for _, row in disp_rec.iterrows():
        status = row["alert_status"]
        msg    = (f"<b>{row['product_id']}</b> — {row['product_name']} | "
                  f"Stock: {int(row['current_stock'])} | "
                  f"Days left: {row['days_to_stockout']:.0f} | "
                  f"Reorder: {int(row['suggested_reorder_qty'])} units")
        if "OUT OF STOCK" in status:
            st.markdown(f'<div class="alert-crit">🔴 {msg}</div>', unsafe_allow_html=True)
        elif "REORDER" in status:
            st.markdown(f'<div class="alert-warn">🟠 {msg}</div>', unsafe_allow_html=True)
        elif "DEAD" in status:
            st.markdown(f'<div class="alert-dead">⚫ {msg} | Days since last sale: {int(row["days_since_sold"])}</div>', unsafe_allow_html=True)
        elif "OVERSTOCK" in status:
            st.markdown(f'<div class="alert-dead">🔵 {msg}</div>', unsafe_allow_html=True)

    # Restock schedule
    st.markdown("---")
    st.markdown('<div class="sec-label">📋 Suggested Restock Schedule</div>', unsafe_allow_html=True)
    needs_order = disp_rec[disp_rec["needs_reorder"] == 1].copy()
    if len(needs_order):
        needs_order["order_by_date"] = pd.Timestamp(date.today()) + pd.to_timedelta(
            needs_order["reorder_lead_days"], unit="d"
        )
        needs_order["suggested_reorder_qty"] = needs_order["suggested_reorder_qty"].astype(int)
        needs_order["est_cost"] = (needs_order["suggested_reorder_qty"] * needs_order["base_price"]).round(2)
        order_tbl = needs_order[[
            "product_id","product_name","category","current_stock",
            "suggested_reorder_qty","reorder_lead_days","order_by_date","est_cost"
        ]].rename(columns={
            "product_id": "SKU", "product_name": "Name", "category": "Cat",
            "current_stock": "Curr Stock", "suggested_reorder_qty": "Order Qty",
            "reorder_lead_days": "Lead Days", "order_by_date": "Order By", "est_cost": "Est Cost (₹)",
        })
        st.dataframe(order_tbl, use_container_width=True, height=300)
        st.metric("Total Restock Cost", f"₹{needs_order['est_cost'].sum():,.0f}")
    else:
        st.success("✅ No products currently need reordering.")

    # Dead stock analysis
    st.markdown("---")
    st.markdown('<div class="sec-label">🪦 Dead Stock Analysis</div>', unsafe_allow_html=True)
    dead = disp_rec[disp_rec["is_dead_stock"] == 1].copy()
    if len(dead):
        dead["inventory_value"] = (dead["current_stock"] * dead["base_price"]).round(2)
        fig = px.bar(dead, x="product_id", y="inventory_value", color="days_since_sold",
                     color_continuous_scale="Reds",
                     title="Dead Stock — Tied-up Inventory Value",
                     labels={"product_id": "SKU", "inventory_value": "Value (₹)", "days_since_sold": "Days Unsold"})
        fig.update_layout(**PLOTLY_THEME, height=300)
        st.plotly_chart(fig, use_container_width=True)
        total_dead_val = dead["inventory_value"].sum()
        st.warning(f"⚠️ Total dead stock value: ₹{total_dead_val:,.0f} — consider discounting or liquidation")
    else:
        st.success("✅ No dead stock detected.")

st.markdown(
    "<br><div style='text-align:center;font-family:monospace;font-size:0.6rem;color:#2a3a4a'>"
    "RetailPulse AI · XGBoost Forecasting · 95% Service Level · Plotly Dashboards</div>",
    unsafe_allow_html=True
)