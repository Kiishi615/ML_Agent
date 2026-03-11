import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Aurora Dashboard",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────
css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

    :root {
        --aurora-1: #00d2ff;
        --aurora-2: #7b2ff7;
        --aurora-3: #c471f5;
        --aurora-4: #fa71cd;
        --aurora-5: #12c2e9;
    }

    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #1a0a2e 25%, #0d1b2a 50%, #1a0a2e 75%, #0a0a1a 100%);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        font-family: 'Inter', sans-serif;
    }

    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .stApp::before {
        content: '';
        position: fixed;
        top: -50%; left: -50%;
        width: 200%; height: 200%;
        background:
            radial-gradient(ellipse at 20% 50%, rgba(123,47,247,0.08) 0%, transparent 50%),
            radial-gradient(ellipse at 80% 50%, rgba(0,210,255,0.08) 0%, transparent 50%),
            radial-gradient(ellipse at 50% 20%, rgba(196,113,245,0.06) 0%, transparent 50%),
            radial-gradient(ellipse at 50% 80%, rgba(250,113,205,0.06) 0%, transparent 50%);
        animation: auroraMove 20s ease-in-out infinite;
        pointer-events: none; z-index: 0;
    }

    @keyframes auroraMove {
        0%, 100% { transform: translate(0,0) rotate(0deg); }
        33% { transform: translate(30px,-30px) rotate(1deg); }
        66% { transform: translate(-20px,20px) rotate(-1deg); }
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}

    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: rgba(255,255,255,0.02); }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, var(--aurora-1), var(--aurora-2));
        border-radius: 3px;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(10,10,26,0.95) 0%, rgba(26,10,46,0.95) 100%) !important;
        border-right: 1px solid rgba(123,47,247,0.2) !important;
        backdrop-filter: blur(20px);
    }

    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #ffffff !important;
        font-family: 'Space Grotesk', sans-serif !important;
    }

    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li {
        color: rgba(255,255,255,0.7) !important;
    }

    section[data-testid="stSidebar"] .stRadio > label {
        color: rgba(255,255,255,0.8) !important;
    }

    section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label {
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 12px !important;
        padding: 10px 16px !important;
        margin-bottom: 6px !important;
        transition: all 0.3s ease !important;
        color: rgba(255,255,255,0.7) !important;
    }

    section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label:hover {
        background: rgba(123,47,247,0.15) !important;
        border-color: rgba(123,47,247,0.4) !important;
        transform: translateX(4px);
        color: #ffffff !important;
    }

    .stSlider > div > div > div {
        background: linear-gradient(90deg, var(--aurora-2), var(--aurora-1)) !important;
    }

    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: #ffffff !important;
    }

    p, li, span { color: rgba(255,255,255,0.8) !important; }

    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 20px 24px;
        backdrop-filter: blur(10px);
        transition: all 0.4s cubic-bezier(0.4,0,0.2,1);
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
    }

    [data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        border-color: rgba(123,47,247,0.4);
        box-shadow: 0 12px 40px rgba(123,47,247,0.15);
    }

    [data-testid="stMetric"] label {
        color: rgba(255,255,255,0.6) !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255,255,255,0.03);
        border-radius: 16px;
        padding: 6px;
        border: 1px solid rgba(255,255,255,0.06);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 12px;
        padding: 10px 24px;
        color: rgba(255,255,255,0.6) !important;
        font-weight: 500;
        transition: all 0.3s ease;
        border: none !important;
        background: transparent !important;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: #ffffff !important;
        background: rgba(255,255,255,0.06) !important;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(123,47,247,0.3), rgba(0,210,255,0.2)) !important;
        color: #ffffff !important;
        font-weight: 600;
    }

    .stTabs [data-baseweb="tab-highlight"] { display: none; }
    .stTabs [data-baseweb="tab-border"] { display: none; }

    .stMultiSelect [data-baseweb="select"],
    .stSelectbox [data-baseweb="select"] {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 12px !important;
        color: white !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, var(--aurora-2), var(--aurora-5)) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 10px 28px !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        letter-spacing: 0.5px;
        transition: all 0.4s cubic-bezier(0.4,0,0.2,1) !important;
        box-shadow: 0 4px 15px rgba(123,47,247,0.3) !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(123,47,247,0.5) !important;
    }

    .stPlotlyChart {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 20px;
        padding: 10px;
        backdrop-filter: blur(10px);
        transition: all 0.4s ease;
    }

    .stPlotlyChart:hover {
        border-color: rgba(123,47,247,0.3);
        box-shadow: 0 8px 32px rgba(123,47,247,0.1);
    }

    hr { border-color: rgba(255,255,255,0.06) !important; }

    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00d2ff, #7b2ff7, #c471f5, #fa71cd);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: textGradient 5s ease infinite;
        margin-bottom: 0;
        line-height: 1.2;
    }

    @keyframes textGradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .hero-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.15rem;
        color: rgba(255,255,255,0.5) !important;
        font-weight: 300;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-top: 8px;
    }

    .pulse-dot {
        display: inline-block;
        width: 8px; height: 8px;
        border-radius: 50%;
        background: #00d2ff;
        margin-right: 8px;
        animation: pulse 2s ease-in-out infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(0,210,255,0.4); }
        50% { opacity: 0.8; box-shadow: 0 0 0 10px rgba(0,210,255,0); }
    }

    .section-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 4px;
    }

    .section-desc {
        color: rgba(255,255,255,0.4) !important;
        font-size: 0.9rem;
        margin-bottom: 20px;
    }

    .custom-progress {
        background: rgba(255,255,255,0.06);
        border-radius: 10px;
        height: 8px;
        overflow: hidden;
        margin: 8px 0;
    }

    .custom-progress-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 1s cubic-bezier(0.4,0,0.2,1);
    }

    .particle {
        position: absolute;
        width: 3px; height: 3px;
        background: rgba(123,47,247,0.3);
        border-radius: 50%;
        animation: floatUp linear infinite;
    }

    @keyframes floatUp {
        0% { transform: translateY(100vh) rotate(0deg); opacity: 0; }
        10% { opacity: 1; }
        90% { opacity: 1; }
        100% { transform: translateY(-10vh) rotate(720deg); opacity: 0; }
    }

    .big-number {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00d2ff, #7b2ff7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1;
    }

    .glass-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.06) 0%, rgba(255,255,255,0.02) 100%);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 24px;
        padding: 32px;
        backdrop-filter: blur(20px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        margin-bottom: 24px;
        transition: all 0.4s cubic-bezier(0.4,0,0.2,1);
    }

    .glass-card:hover {
        transform: translateY(-2px);
        border-color: rgba(123,47,247,0.3);
        box-shadow: 0 16px 48px rgba(123,47,247,0.12);
    }

    .activity-item {
        display: flex;
        align-items: center;
        padding: 14px 18px;
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 14px;
        margin-bottom: 10px;
        transition: all 0.3s ease;
    }

    .activity-item:hover {
        background: rgba(255,255,255,0.04);
        border-color: rgba(123,47,247,0.2);
        transform: translateX(4px);
    }

    .activity-icon {
        width: 40px; height: 40px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        margin-right: 14px;
        flex-shrink: 0;
    }

    .activity-title {
        color: #ffffff !important;
        font-weight: 500;
        font-size: 0.9rem;
        margin: 0;
    }

    .activity-time {
        color: rgba(255,255,255,0.35) !important;
        font-size: 0.78rem;
        margin: 0;
    }

    .badge-green {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        background: rgba(0,255,136,0.12);
        color: #00ff88;
        border: 1px solid rgba(0,255,136,0.25);
    }

    .stTextInput input, .stNumberInput input {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 12px !important;
        color: white !important;
    }
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# ──────────────────────────────────────────────
# FLOATING PARTICLES
# ──────────────────────────────────────────────
particles_html = '<div style="position:fixed;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:0;overflow:hidden;">'
for i in range(30):
    size = random.uniform(2, 5)
    left = random.uniform(0, 100)
    delay = random.uniform(0, 20)
    duration = random.uniform(15, 30)
    opacity = random.uniform(0.1, 0.4)
    color_choices = [
        "rgba(123,47,247,{})".format(opacity),
        "rgba(0,210,255,{})".format(opacity),
        "rgba(196,113,245,{})".format(opacity),
        "rgba(250,113,205,{})".format(opacity),
    ]
    color = random.choice(color_choices)
    particles_html += (
        '<div class="particle" style="left:{}%;width:{}px;height:{}px;'
        'background:{};animation-delay:{}s;animation-duration:{}s;"></div>'
    ).format(left, size, size, color, delay, duration)
particles_html += '</div>'
st.markdown(particles_html, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def plotly_layout(title="", height=400):
    return dict(
        title=dict(
            text=title,
            font=dict(family="Space Grotesk", size=18, color="white"),
            x=0.02, y=0.95
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="rgba(255,255,255,0.7)", size=12),
        height=height,
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.04)",
            zerolinecolor="rgba(255,255,255,0.06)",
            tickfont=dict(color="rgba(255,255,255,0.5)"),
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.04)",
            zerolinecolor="rgba(255,255,255,0.06)",
            tickfont=dict(color="rgba(255,255,255,0.5)"),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(255,255,255,0.1)",
            font=dict(color="rgba(255,255,255,0.7)"),
        ),
        hoverlabel=dict(
            bgcolor="rgba(15,15,35,0.95)",
            bordercolor="rgba(123,47,247,0.5)",
            font=dict(color="white", family="Inter"),
        ),
    )


aurora_colors = [
    "#00d2ff", "#7b2ff7", "#c471f5", "#fa71cd", "#12c2e9",
    "#00ff88", "#ff6b6b", "#ffd93d", "#6bcb77", "#4d96ff"
]


# ──────────────────────────────────────────────
# GENERATE DATA
# ──────────────────────────────────────────────
np.random.seed(42)
days = 365
dates = [datetime.now() - timedelta(days=days - i) for i in range(days)]

revenue = np.cumsum(np.random.randn(days) * 1200 + 800) + 50000
users = np.cumsum(np.abs(np.random.randn(days) * 50 + 30)).astype(int) + 10000
sessions = np.random.randint(500, 3000, days)
conversion = np.clip(np.random.randn(days) * 0.5 + 4.5, 1.5, 8.5)

df = pd.DataFrame({
    "Date": dates,
    "Revenue": revenue,
    "Users": users,
    "Sessions": sessions,
    "Conversion Rate": conversion,
})


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div style="text-align:center;padding:20px 0;">'
        '<div style="font-size:3rem;margin-bottom:8px;">🌌</div>'
        '<div style="font-family:Space Grotesk,sans-serif;font-size:1.5rem;font-weight:700;'
        'background:linear-gradient(135deg,#00d2ff,#7b2ff7);'
        '-webkit-background-clip:text;-webkit-text-fill-color:transparent;">Aurora</div>'
        '<div style="color:rgba(255,255,255,0.4);font-size:0.8rem;letter-spacing:3px;'
        'text-transform:uppercase;">Analytics Dashboard</div>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠 Overview", "📊 Analytics", "🌍 Geography", "🎨 Playground"],
        index=0
    )

    st.markdown("---")
    st.markdown("##### ⚙️ Settings")

    theme_accent = st.select_slider(
        "Accent Color",
        options=["🔵 Cyan", "🟣 Purple", "🩷 Pink", "🟢 Green"],
        value="🟣 Purple"
    )

    date_range = st.slider("Date Range (days)", 30, 365, 180, 30)

    st.markdown("---")

    st.markdown(
        '<div style="text-align:center;padding:16px;background:rgba(255,255,255,0.02);'
        'border:1px solid rgba(255,255,255,0.06);border-radius:16px;margin-top:10px;">'
        '<div style="color:rgba(255,255,255,0.3);font-size:0.75rem;letter-spacing:1px;'
        'text-transform:uppercase;">Status</div>'
        '<div style="margin-top:8px;">'
        '<span class="pulse-dot"></span>'
        '<span style="color:#00ff88;font-weight:600;font-size:0.85rem;">All Systems Operational</span>'
        '</div>'
        '<div style="color:rgba(255,255,255,0.25);font-size:0.72rem;margin-top:6px;">'
        'Last updated: just now</div>'
        '</div>',
        unsafe_allow_html=True
    )


# Filter data
df_filtered = df.tail(date_range)


# ──────────────────────────────────────────────
# PAGE: OVERVIEW
# ──────────────────────────────────────────────
if page == "🏠 Overview":

    st.markdown(
        '<div style="padding:20px 0 10px 0;">'
        '<div class="hero-title">Good evening, Commander</div>'
        '<div class="hero-subtitle">Here\'s what\'s happening across your universe</div>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Metrics ──
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Revenue", "${:,.0f}".format(revenue[-1]),
                   "+{}%".format(random.randint(5, 15)))
    with c2:
        st.metric("Active Users", "{:,}".format(users[-1]),
                   "+{:,}".format(random.randint(800, 2000)))
    with c3:
        st.metric("Avg Sessions", "{:,.0f}".format(sessions[-30:].mean()),
                   "+{}%".format(random.randint(3, 12)))
    with c4:
        st.metric("Conversion", "{:.1f}%".format(conversion[-1]),
                   "+{:.1f}%".format(random.uniform(0.2, 1.2)))

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Revenue Chart + Goals ──
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown(
            '<div class="section-title">📈 Revenue Trajectory</div>'
            '<div class="section-desc">Revenue growth over the selected period</div>',
            unsafe_allow_html=True
        )

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_filtered["Date"], y=df_filtered["Revenue"],
            mode="lines",
            line=dict(color="#7b2ff7", width=3, shape="spline"),
            fill="tozeroy",
            fillcolor="rgba(123,47,247,0.08)",
            name="Revenue",
            hovertemplate="<b>%{x|%b %d}</b><br>Revenue: $%{y:,.0f}<extra></extra>"
        ))

        window = 30
        if len(df_filtered) > window:
            ma = df_filtered["Revenue"].rolling(window=window).mean()
            fig.add_trace(go.Scatter(
                x=df_filtered["Date"], y=ma,
                mode="lines",
                line=dict(color="#00d2ff", width=2, dash="dot"),
                name="30D Moving Avg",
                hovertemplate="<b>%{x|%b %d}</b><br>MA: $%{y:,.0f}<extra></extra>"
            ))

        fig.update_layout(**plotly_layout("", 420))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown(
            '<div class="section-title">🎯 Goals Progress</div>'
            '<div class="section-desc">Quarterly objectives tracking</div>',
            unsafe_allow_html=True
        )

        goals = [
            {"name": "Revenue Target", "current": 847, "target": 1000,
             "color": "#7b2ff7", "icon": "💰"},
            {"name": "New Users", "current": 12400, "target": 15000,
             "color": "#00d2ff", "icon": "👥"},
            {"name": "Retention", "current": 89, "target": 95,
             "color": "#00ff88", "icon": "🔄"},
            {"name": "NPS Score", "current": 72, "target": 80,
             "color": "#fa71cd", "icon": "⭐"},
        ]

        for g in goals:
            pct = min(g["current"] / g["target"] * 100, 100)
            html = (
                '<div style="padding:14px 18px;background:rgba(255,255,255,0.02);'
                'border:1px solid rgba(255,255,255,0.06);border-radius:14px;margin-bottom:12px;">'
                '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">'
                '<span style="color:rgba(255,255,255,0.8);font-weight:500;font-size:0.85rem;">'
                '{icon} {name}</span>'
                '<span style="color:{color};font-weight:700;font-size:0.85rem;">{pct:.0f}%</span>'
                '</div>'
                '<div class="custom-progress">'
                '<div class="custom-progress-fill" style="width:{pct}%;'
                'background:linear-gradient(90deg,{color}88,{color});"></div>'
                '</div>'
                '<div style="display:flex;justify-content:space-between;margin-top:4px;">'
                '<span style="color:rgba(255,255,255,0.3);font-size:0.72rem;">Current: {current:,}</span>'
                '<span style="color:rgba(255,255,255,0.3);font-size:0.72rem;">Target: {target:,}</span>'
                '</div></div>'
            ).format(
                icon=g["icon"], name=g["name"], color=g["color"],
                pct=pct, current=g["current"], target=g["target"]
            )
            st.markdown(html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Bottom Row ──
    col_a, col_b, col_c = st.columns([1, 1, 1])

    with col_a:
        st.markdown(
            '<div class="section-title">📊 Traffic Sources</div>'
            '<div class="section-desc">Where your visitors come from</div>',
            unsafe_allow_html=True
        )

        sources = ["Organic", "Direct", "Referral", "Social", "Email"]
        values = [35, 25, 18, 14, 8]
        colors = ["#7b2ff7", "#00d2ff", "#c471f5", "#fa71cd", "#00ff88"]

        fig_donut = go.Figure(data=[go.Pie(
            labels=sources, values=values, hole=0.7,
            marker=dict(colors=colors,
                        line=dict(color="rgba(10,10,26,1)", width=3)),
            textinfo="none",
            hovertemplate="<b>%{label}</b><br>%{value}% of traffic<extra></extra>"
        )])

        layout_donut = plotly_layout("", 350)
        layout_donut["showlegend"] = True
        layout_donut["legend"] = dict(
            orientation="h", yanchor="bottom", y=-0.15,
            xanchor="center", x=0.5,
            font=dict(size=11, color="rgba(255,255,255,0.6)")
        )
        layout_donut["annotations"] = [dict(
            text='<b style="font-size:24px;color:white;">100K</b><br>'
                 '<span style="color:rgba(255,255,255,0.4);font-size:11px;">VISITORS</span>',
            x=0.5, y=0.5, font_size=14, showarrow=False,
            font=dict(color="white")
        )]
        fig_donut.update_layout(**layout_donut)
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_b:
        st.markdown(
            '<div class="section-title">🌡️ Performance Gauge</div>'
            '<div class="section-desc">Real-time system performance score</div>',
            unsafe_allow_html=True
        )

        score = 87
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=score,
            delta={"reference": 80, "increasing": {"color": "#00ff88"},
                   "font": {"size": 16}},
            number={"font": {"size": 48, "color": "white",
                             "family": "Space Grotesk"}, "suffix": "%"},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 0,
                         "tickcolor": "rgba(0,0,0,0)",
                         "tickfont": {"color": "rgba(255,255,255,0.3)",
                                      "size": 10}},
                "bar": {"color": "rgba(0,0,0,0)"},
                "bgcolor": "rgba(255,255,255,0.03)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40], "color": "rgba(255,107,107,0.2)"},
                    {"range": [40, 70], "color": "rgba(255,217,61,0.2)"},
                    {"range": [70, 100], "color": "rgba(0,255,136,0.15)"},
                ],
                "threshold": {
                    "line": {"color": "#00ff88", "width": 4},
                    "thickness": 0.8, "value": score
                },
            }
        ))
        fig_gauge.update_layout(**plotly_layout("", 350))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_c:
        st.markdown(
            '<div class="section-title">⚡ Recent Activity</div>'
            '<div class="section-desc">Latest events in your system</div>',
            unsafe_allow_html=True
        )

        activities = [
            {"icon": "🚀", "title": "Deployment successful",
             "time": "2 min ago", "bg": "rgba(123,47,247,0.15)"},
            {"icon": "👤", "title": "New enterprise client signed",
             "time": "18 min ago", "bg": "rgba(0,210,255,0.15)"},
            {"icon": "📈", "title": "Revenue milestone reached",
             "time": "1 hour ago", "bg": "rgba(0,255,136,0.15)"},
            {"icon": "🔔", "title": "Alert threshold updated",
             "time": "3 hours ago", "bg": "rgba(255,217,61,0.15)"},
            {"icon": "🎯", "title": "Campaign goal achieved",
             "time": "5 hours ago", "bg": "rgba(250,113,205,0.15)"},
        ]

        for act in activities:
            html = (
                '<div class="activity-item">'
                '<div class="activity-icon" style="background:{bg};">{icon}</div>'
                '<div>'
                '<div class="activity-title">{title}</div>'
                '<div class="activity-time">{time}</div>'
                '</div></div>'
            ).format(
                bg=act["bg"], icon=act["icon"],
                title=act["title"], time=act["time"]
            )
            st.markdown(html, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# PAGE: ANALYTICS
# ──────────────────────────────────────────────
elif page == "📊 Analytics":

    st.markdown(
        '<div style="padding:20px 0 10px 0;">'
        '<div class="hero-title">Deep Analytics</div>'
        '<div class="hero-subtitle">Explore your data from every angle</div>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📈 Trends", "🔥 Heatmap", "📊 Distribution"])

    with tab1:
        st.markdown("<br>", unsafe_allow_html=True)

        metric_choice = st.selectbox(
            "Select Metric",
            ["Revenue", "Users", "Sessions", "Conversion Rate"],
            index=0
        )

        color_map = {
            "Revenue": ("#7b2ff7", "rgba(123,47,247,0.1)"),
            "Users": ("#00d2ff", "rgba(0,210,255,0.1)"),
            "Sessions": ("#fa71cd", "rgba(250,113,205,0.1)"),
            "Conversion Rate": ("#00ff88", "rgba(0,255,136,0.1)"),
        }

        line_color, fill_color = color_map[metric_choice]

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=df_filtered["Date"], y=df_filtered[metric_choice],
            mode="lines",
            line=dict(color=line_color, width=2.5, shape="spline"),
            fill="tozeroy", fillcolor=fill_color,
            name=metric_choice,
            hovertemplate=(
                "<b>%{x|%b %d, %Y}</b><br>"
                + metric_choice + ": %{y:,.2f}<extra></extra>"
            )
        ))

        max_idx = df_filtered[metric_choice].idxmax()
        fig_trend.add_annotation(
            x=df_filtered.loc[max_idx, "Date"],
            y=df_filtered.loc[max_idx, metric_choice],
            text="Peak", showarrow=True, arrowhead=0,
            arrowcolor=line_color,
            font=dict(color=line_color, size=11, family="Space Grotesk"),
            arrowwidth=1.5, ax=0, ay=-35,
            bgcolor="rgba(15,15,35,0.8)",
            bordercolor=line_color, borderwidth=1, borderpad=4,
        )

        fig_trend.update_layout(
            **plotly_layout(metric_choice + " Over Time", 480)
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        s1, s2, s3, s4 = st.columns(4)
        vals = df_filtered[metric_choice]
        with s1:
            st.metric("Mean", "{:,.2f}".format(vals.mean()))
        with s2:
            st.metric("Median", "{:,.2f}".format(vals.median()))
        with s3:
            st.metric("Std Dev", "{:,.2f}".format(vals.std()))
        with s4:
            st.metric("Peak", "{:,.2f}".format(vals.max()))

    with tab2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title">🔥 Weekly Activity Heatmap</div>'
            '<div class="section-desc">Session intensity by day and hour</div>',
            unsafe_allow_html=True
        )

        hours = list(range(24))
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        heatmap_data = np.random.randint(10, 100, size=(7, 24))
        heatmap_data[0:5, 9:18] += 60
        heatmap_data[0:5, 12:14] += 30

        fig_heat = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=["{:02d}:00".format(h) for h in hours],
            y=day_names,
            colorscale=[
                [0, "rgba(10,10,26,1)"],
                [0.2, "rgba(123,47,247,0.3)"],
                [0.4, "rgba(123,47,247,0.5)"],
                [0.6, "rgba(0,210,255,0.6)"],
                [0.8, "rgba(0,255,136,0.7)"],
                [1, "rgba(0,255,136,1)"],
            ],
            hovertemplate="<b>%{y} at %{x}</b><br>Sessions: %{z}<extra></extra>",
            showscale=True,
            colorbar=dict(
                tickfont=dict(color="rgba(255,255,255,0.5)"),
                title=dict(text="Sessions",
                           font=dict(color="rgba(255,255,255,0.5)")),
            )
        ))

        heat_layout = plotly_layout("", 420)
        heat_layout["xaxis"] = dict(
            side="top",
            tickfont=dict(color="rgba(255,255,255,0.5)", size=10),
            gridcolor="rgba(0,0,0,0)",
        )
        heat_layout["yaxis"] = dict(
            tickfont=dict(color="rgba(255,255,255,0.6)", size=12),
            gridcolor="rgba(0,0,0,0)",
            autorange="reversed"
        )
        fig_heat.update_layout(**heat_layout)
        st.plotly_chart(fig_heat, use_container_width=True)

    with tab3:
        st.markdown("<br>", unsafe_allow_html=True)
        col_d1, col_d2 = st.columns(2)

        with col_d1:
            st.markdown(
                '<div class="section-title">📊 Revenue Distribution</div>',
                unsafe_allow_html=True
            )

            fig_hist = go.Figure()
            daily_rev = np.diff(df_filtered["Revenue"].values)
            fig_hist.add_trace(go.Histogram(
                x=daily_rev, nbinsx=40,
                marker=dict(
                    color="rgba(123,47,247,0.5)",
                    line=dict(color="#7b2ff7", width=1),
                ),
                hovertemplate="Range: %{x:,.0f}<br>Count: %{y}<extra></extra>"
            ))
            fig_hist.update_layout(
                **plotly_layout("Daily Revenue Changes", 400)
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_d2:
            st.markdown(
                '<div class="section-title">🫧 Correlation Bubble</div>',
                unsafe_allow_html=True
            )

            fig_bubble = go.Figure()
            fig_bubble.add_trace(go.Scatter(
                x=df_filtered["Sessions"],
                y=df_filtered["Conversion Rate"],
                mode="markers",
                marker=dict(
                    size=np.clip(df_filtered["Users"] / 800, 5, 30),
                    color=df_filtered["Revenue"],
                    colorscale=[
                        [0, "#7b2ff7"], [0.5, "#00d2ff"], [1, "#00ff88"]
                    ],
                    opacity=0.7,
                    line=dict(width=1, color="rgba(255,255,255,0.1)"),
                    showscale=True,
                    colorbar=dict(
                        title="Revenue",
                        tickfont=dict(color="rgba(255,255,255,0.5)"),
                        titlefont=dict(color="rgba(255,255,255,0.5)"),
                    ),
                ),
                hovertemplate=(
                    "Sessions: %{x:,}<br>Conv: %{y:.1f}%<br>"
                    "Revenue: $%{marker.color:,.0f}<extra></extra>"
                )
            ))
            fig_bubble.update_layout(
                **plotly_layout("Sessions vs Conversion", 400)
            )
            st.plotly_chart(fig_bubble, use_container_width=True)


# ──────────────────────────────────────────────
# PAGE: GEOGRAPHY
# ──────────────────────────────────────────────
elif page == "🌍 Geography":

    st.markdown(
        '<div style="padding:20px 0 10px 0;">'
        '<div class="hero-title">Global Presence</div>'
        '<div class="hero-subtitle">Your worldwide footprint and reach</div>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    countries = [
        "USA", "GBR", "DEU", "FRA", "JPN", "AUS", "CAN", "BRA",
        "IND", "KOR", "SGP", "NLD", "SWE", "NOR", "ESP", "ITA",
        "MEX", "ARG", "CHN", "RUS"
    ]
    country_names = [
        "United States", "United Kingdom", "Germany", "France",
        "Japan", "Australia", "Canada", "Brazil", "India",
        "South Korea", "Singapore", "Netherlands", "Sweden",
        "Norway", "Spain", "Italy", "Mexico", "Argentina",
        "China", "Russia"
    ]
    user_counts = [
        45000, 18000, 15000, 12000, 22000, 8000, 11000, 9000,
        28000, 14000, 6000, 7000, 5000, 4000, 8500, 7500,
        6500, 4500, 35000, 10000
    ]

    df_geo = pd.DataFrame({
        "ISO": countries,
        "Country": country_names,
        "Users": user_counts,
    })

    fig_map = go.Figure(data=go.Choropleth(
        locations=df_geo["ISO"],
        z=df_geo["Users"],
        text=df_geo["Country"],
        colorscale=[
            [0, "rgba(10,10,30,1)"],
            [0.25, "rgba(123,47,247,0.4)"],
            [0.5, "rgba(123,47,247,0.7)"],
            [0.75, "rgba(0,210,255,0.8)"],
            [1, "rgba(0,255,136,0.9)"],
        ],
        marker_line_color="rgba(255,255,255,0.1)",
        marker_line_width=0.5,
        colorbar=dict(
            title="Users",
            tickfont=dict(color="rgba(255,255,255,0.5)"),
            titlefont=dict(color="rgba(255,255,255,0.6)"),
        ),
        hovertemplate="<b>%{text}</b><br>Users: %{z:,}<extra></extra>"
    ))

    fig_map.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=520,
        margin=dict(l=0, r=0, t=30, b=0),
        geo=dict(
            bgcolor="rgba(0,0,0,0)",
            lakecolor="rgba(10,10,30,1)",
            landcolor="rgba(15,15,35,1)",
            showocean=True,
            oceancolor="rgba(8,8,20,1)",
            showlakes=True, showland=True,
            showcountries=True,
            countrycolor="rgba(255,255,255,0.05)",
            coastlinecolor="rgba(255,255,255,0.08)",
            projection_type="natural earth",
            showframe=False,
        ),
        font=dict(family="Inter", color="rgba(255,255,255,0.7)"),
    )

    st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">🏆 Top Markets</div>'
        '<div class="section-desc">Highest performing regions by user count</div>',
        unsafe_allow_html=True
    )

    df_geo_sorted = df_geo.sort_values("Users", ascending=False).head(8)

    flags = {
        "United States": "🇺🇸", "China": "🇨🇳", "India": "🇮🇳",
        "Japan": "🇯🇵", "United Kingdom": "🇬🇧", "Germany": "🇩🇪",
        "South Korea": "🇰🇷", "France": "🇫🇷", "Canada": "🇨🇦",
        "Australia": "🇦🇺", "Brazil": "🇧🇷", "Russia": "🇷🇺"
    }

    col_geo = st.columns(4)
    for idx, (_, row) in enumerate(df_geo_sorted.iterrows()):
        with col_geo[idx % 4]:
            flag = flags.get(row["Country"], "🌐")
            pct_up = random.randint(5, 25)
            html = (
                '<div class="glass-card" style="text-align:center;padding:24px 16px;">'
                '<div style="font-size:2.5rem;margin-bottom:8px;">{flag}</div>'
                '<div style="color:white;font-weight:600;font-size:1rem;'
                'font-family:Space Grotesk;">{country}</div>'
                '<div class="big-number">{users:,}</div>'
                '<span style="color:rgba(255,255,255,0.4);font-size:0.78rem;">users</span>'
                '<div style="margin-top:12px;">'
                '<span class="badge-green">↑ {pct}%</span>'
                '</div></div>'
            ).format(
                flag=flag, country=row["Country"],
                users=row["Users"], pct=pct_up
            )
            st.markdown(html, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# PAGE: PLAYGROUND
# ──────────────────────────────────────────────
elif page == "🎨 Playground":

    st.markdown(
        '<div style="padding:20px 0 10px 0;">'
        '<div class="hero-title">Creative Playground</div>'
        '<div class="hero-subtitle">Interactive visualizations and generative art</div>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    col_play1, col_play2 = st.columns(2)

    with col_play1:
        st.markdown(
            '<div class="section-title">🌀 3D Surface</div>'
            '<div class="section-desc">Interactive mathematical surface</div>',
            unsafe_allow_html=True
        )

        surface_type = st.selectbox(
            "Surface Type",
            ["Sinusoidal Wave", "Ripple", "Saddle", "Spiral"]
        )

        x_surf = np.linspace(-5, 5, 80)
        y_surf = np.linspace(-5, 5, 80)
        X, Y = np.meshgrid(x_surf, y_surf)

        if surface_type == "Sinusoidal Wave":
            Z = np.sin(np.sqrt(X**2 + Y**2))
        elif surface_type == "Ripple":
            Z = np.sin(X * 2) * np.cos(Y * 2)
        elif surface_type == "Saddle":
            Z = X**2 - Y**2
        else:
            R = np.sqrt(X**2 + Y**2)
            Z = np.sin(R + np.arctan2(Y, X) * 3)

        fig_3d = go.Figure(data=[go.Surface(
            z=Z, x=X, y=Y,
            colorscale=[
                [0, "#0a0a2e"], [0.2, "#7b2ff7"],
                [0.4, "#c471f5"], [0.6, "#00d2ff"],
                [0.8, "#00ff88"], [1, "#ffffff"],
            ],
            opacity=0.9,
            contours=dict(
                z=dict(show=True, usecolormap=True,
                       highlightcolor="white", project_z=True)
            ),
            hovertemplate="x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<extra></extra>"
        )])

        fig_3d.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=500,
            margin=dict(l=0, r=0, t=30, b=0),
            scene=dict(
                bgcolor="rgba(0,0,0,0)",
                xaxis=dict(
                    backgroundcolor="rgba(0,0,0,0)",
                    gridcolor="rgba(255,255,255,0.05)",
                    showbackground=True,
                    zerolinecolor="rgba(255,255,255,0.1)",
                    tickfont=dict(color="rgba(255,255,255,0.3)")
                ),
                yaxis=dict(
                    backgroundcolor="rgba(0,0,0,0)",
                    gridcolor="rgba(255,255,255,0.05)",
                    showbackground=True,
                    zerolinecolor="rgba(255,255,255,0.1)",
                    tickfont=dict(color="rgba(255,255,255,0.3)")
                ),
                zaxis=dict(
                    backgroundcolor="rgba(0,0,0,0)",
                    gridcolor="rgba(255,255,255,0.05)",
                    showbackground=True,
                    zerolinecolor="rgba(255,255,255,0.1)",
                    tickfont=dict(color="rgba(255,255,255,0.3)")
                ),
                camera=dict(eye=dict(x=1.8, y=1.8, z=1.2)),
            ),
            font=dict(color="rgba(255,255,255,0.7)"),
        )

        st.plotly_chart(fig_3d, use_container_width=True)

    with col_play2:
        st.markdown(
            '<div class="section-title">🎆 Generative Polar</div>'
            '<div class="section-desc">Parametric art through mathematics</div>',
            unsafe_allow_html=True
        )

        petals = st.slider("Complexity", 2, 12, 6)
        layers = st.slider("Layers", 1, 5, 3)

        fig_polar = go.Figure()

        for layer in range(layers):
            theta = np.linspace(0, 2 * np.pi, 1000)
            r = (np.cos(petals * theta + layer * np.pi / layers)
                 + 0.5 * np.sin(3 * theta))
            r = np.abs(r) * (1 + layer * 0.3)

            opacity = 0.8 - layer * 0.12
            c = aurora_colors[layer % len(aurora_colors)]

            cr = int(c[1:3], 16)
            cg = int(c[3:5], 16)
            cb = int(c[5:7], 16)
            fill_c = "rgba({},{},{},0.03)".format(cr, cg, cb)

            fig_polar.add_trace(go.Scatterpolar(
                r=r, theta=np.degrees(theta),
                mode="lines",
                line=dict(color=c, width=2),
                opacity=opacity,
                fill="toself", fillcolor=fill_c,
                name="Layer {}".format(layer + 1),
                hoverinfo="skip",
            ))

        fig_polar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=500,
            margin=dict(l=40, r=40, t=40, b=40),
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=False),
                angularaxis=dict(
                    gridcolor="rgba(255,255,255,0.04)",
                    tickfont=dict(color="rgba(255,255,255,0.3)"),
                    linecolor="rgba(255,255,255,0.06)",
                ),
            ),
            showlegend=True,
            legend=dict(
                bgcolor="rgba(0,0,0,0)",
                font=dict(color="rgba(255,255,255,0.6)"),
            ),
            font=dict(family="Inter", color="rgba(255,255,255,0.7)"),
        )

        st.plotly_chart(fig_polar, use_container_width=True)

    # ── Radar Chart ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">🕸️ Skills Radar</div>'
        '<div class="section-desc">Multi-dimensional performance comparison</div>',
        unsafe_allow_html=True
    )

    col_r1, col_r2 = st.columns([2, 1])

    with col_r1:
        categories = [
            "Speed", "Reliability", "Scalability",
            "Security", "UX Design", "Innovation"
        ]

        fig_radar = go.Figure()

        fig_radar.add_trace(go.Scatterpolar(
            r=[92, 85, 88, 95, 78, 90],
            theta=categories,
            fill="toself",
            fillcolor="rgba(123,47,247,0.15)",
            line=dict(color="#7b2ff7", width=2),
            name="Current Quarter",
        ))

        fig_radar.add_trace(go.Scatterpolar(
            r=[78, 80, 72, 88, 85, 75],
            theta=categories,
            fill="toself",
            fillcolor="rgba(0,210,255,0.1)",
            line=dict(color="#00d2ff", width=2),
            name="Previous Quarter",
        ))

        fig_radar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=450,
            margin=dict(l=80, r=80, t=40, b=40),
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(
                    visible=True, range=[0, 100],
                    gridcolor="rgba(255,255,255,0.06)",
                    tickfont=dict(color="rgba(255,255,255,0.3)", size=10),
                    linecolor="rgba(255,255,255,0.08)",
                ),
                angularaxis=dict(
                    gridcolor="rgba(255,255,255,0.06)",
                    tickfont=dict(
                        color="rgba(255,255,255,0.7)", size=12,
                        family="Space Grotesk"
                    ),
                    linecolor="rgba(255,255,255,0.08)",
                ),
            ),
            legend=dict(
                bgcolor="rgba(0,0,0,0)",
                font=dict(color="rgba(255,255,255,0.7)", family="Inter"),
                bordercolor="rgba(255,255,255,0.1)", borderwidth=1,
            ),
            font=dict(family="Inter", color="rgba(255,255,255,0.7)"),
        )

        st.plotly_chart(fig_radar, use_container_width=True)

    with col_r2:
        st.markdown("<br><br>", unsafe_allow_html=True)

        improvements = [
            {"cat": "Speed", "val": 92, "delta": "+14", "color": "#7b2ff7"},
            {"cat": "Reliability", "val": 85, "delta": "+5", "color": "#00d2ff"},
            {"cat": "Scalability", "val": 88, "delta": "+16", "color": "#c471f5"},
            {"cat": "Security", "val": 95, "delta": "+7", "color": "#00ff88"},
            {"cat": "UX Design", "val": 78, "delta": "-7", "color": "#fa71cd"},
            {"cat": "Innovation", "val": 90, "delta": "+15", "color": "#ffd93d"},
        ]

        for imp in improvements:
            if imp["delta"].startswith("+"):
                sign_color = "#00ff88"
            else:
                sign_color = "#ff6b6b"

            html = (
                '<div style="display:flex;align-items:center;padding:12px 16px;'
                'background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.05);'
                'border-radius:12px;margin-bottom:8px;">'
                '<div style="width:8px;height:8px;border-radius:50%;'
                'background:{color};margin-right:12px;flex-shrink:0;'
                'box-shadow:0 0 8px {color}44;"></div>'
                '<div style="flex:1;">'
                '<span style="color:rgba(255,255,255,0.8);font-weight:500;'
                'font-size:0.85rem;">{cat}</span></div>'
                '<div style="text-align:right;">'
                '<span style="color:white;font-weight:700;'
                'font-family:Space Grotesk;font-size:1.1rem;">{val}</span>'
                '<span style="color:{sign_color};font-weight:600;'
                'font-size:0.78rem;margin-left:8px;">{delta}</span>'
                '</div></div>'
            ).format(
                color=imp["color"], cat=imp["cat"],
                val=imp["val"], sign_color=sign_color,
                delta=imp["delta"]
            )
            st.markdown(html, unsafe_allow_html=True)

    # ── Particle Field ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">✨ Particle Field</div>'
        '<div class="section-desc">Click the button to regenerate the cosmic field</div>',
        unsafe_allow_html=True
    )

    if st.button("🌟 Generate New Universe", use_container_width=True):
        st.balloons()

    n_particles = 500
    theta_p = np.random.uniform(0, 2 * np.pi, n_particles)
    r_p = np.random.exponential(2, n_particles)
    x_p = r_p * np.cos(theta_p) + np.random.randn(n_particles) * 0.3
    y_p = r_p * np.sin(theta_p) + np.random.randn(n_particles) * 0.3
    sizes = np.random.exponential(4, n_particles) + 2
    colors_p = np.random.choice(aurora_colors, n_particles)

    fig_particles = go.Figure()

    for c in set(colors_p):
        mask = colors_p == c
        fig_particles.add_trace(go.Scatter(
            x=x_p[mask], y=y_p[mask],
            mode="markers",
            marker=dict(size=sizes[mask], color=c, opacity=0.7,
                        line=dict(width=0)),
            hoverinfo="skip", showlegend=False,
        ))

    particle_layout = plotly_layout("", 400)
    particle_layout["xaxis"] = dict(visible=False)
    particle_layout["yaxis"] = dict(visible=False, scaleanchor="x")
    fig_particles.update_layout(**particle_layout)

    st.plotly_chart(fig_particles, use_container_width=True)


# ──────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    '<div style="text-align:center;padding:30px 0;'
    'border-top:1px solid rgba(255,255,255,0.06);">'
    '<div style="font-family:Space Grotesk,sans-serif;font-size:1rem;'
    'background:linear-gradient(135deg,#00d2ff,#7b2ff7);'
    '-webkit-background-clip:text;-webkit-text-fill-color:transparent;'
    'font-weight:600;margin-bottom:6px;">Aurora Dashboard</div>'
    '<div style="color:rgba(255,255,255,0.25);font-size:0.78rem;'
    'letter-spacing:1px;">Crafted with ✨ and Streamlit</div>'
    '</div>',
    unsafe_allow_html=True
)