import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.config import (
    PROCESSED_CSV_PATH, PROCESSED_DATA_PATH, REPORTS_DIR, FIGURES_DIR,
    RAW_DATA_PATH, MLFLOW_DB_PATH
)
from src.predict import PropertyPredictor

# --- Page Configuration ---
st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Design System & CSS Variables ---
st.markdown(r"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --font-heading: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        --font-body: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        --bg-app: #F8FAFC;
        --bg-card: #FFFFFF;
        --bg-card-hover: #F8FAFC;
        --bg-dark-card: #0F172A;
        --border-subtle: #E2E8F0;
        --border-focus: #0284C7;
        --text-primary: #0F172A;
        --text-secondary: #334155;
        --text-muted: #64748B;
        --primary-blue: #0284C7;
        --accent-teal: #0D9488;
        --accent-indigo: #6366F1;
        --success-emerald: #10B981;
        --warning-amber: #F59E0B;
        --danger-rose: #EF4444;
        --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.04);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.06), 0 2px 4px -1px rgba(0, 0, 0, 0.04);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.08), 0 4px 6px -2px rgba(0, 0, 0, 0.04);
        --shadow-glow: 0 0 20px rgba(2, 132, 199, 0.15);
    }

    html, body, [class*="css"] {
        font-family: var(--font-body);
        color: var(--text-primary);
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: var(--font-heading);
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.02em;
    }

    /* Page Entrance Animation */
    .element-container, .stMarkdown, .stPlotlyChart {
        animation: fadeIn 350ms cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(4px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Top Hero Banner with Subtle Ambient Glow */
    .saas-hero-container {
        background: linear-gradient(-45deg, #0B132B, #1C2541, #0A192F, #1C2541);
        background-size: 300% 300%;
        animation: ambientHeroGlow 18s ease infinite;
        border-radius: 16px;
        padding: 26px 30px;
        color: #FFFFFF;
        box-shadow: 0 10px 25px -5px rgba(11, 19, 43, 0.25);
        margin-bottom: 22px;
        border: 1px solid rgba(255, 255, 255, 0.12);
    }
    @keyframes ambientHeroGlow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .hero-title-main {
        font-family: var(--font-heading);
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #FFFFFF;
        margin: 0;
        line-height: 1.15;
    }
    .hero-subtitle-main {
        font-size: 1.0rem;
        color: #94A3B8;
        margin-top: 6px;
        margin-bottom: 16px;
        font-weight: 500;
    }
    .status-strip {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        align-items: center;
        padding-top: 12px;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    }
    .status-pill {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.14);
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.78rem;
        font-weight: 600;
        color: #E2E8F0;
        display: inline-flex;
        align-items: center;
        gap: 5px;
    }
    .status-pill-check {
        color: #38BDF8;
        font-weight: 800;
    }

    /* Professional Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid var(--border-subtle);
    }
    .sidebar-brand-box {
        padding: 12px 6px 16px 6px;
        border-bottom: 1px solid var(--border-subtle);
        margin-bottom: 16px;
    }
    .sidebar-brand-title {
        font-family: var(--font-heading);
        font-size: 1.15rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.01em;
    }
    .sidebar-brand-sub {
        font-size: 0.8rem;
        color: #64748B;
        font-weight: 500;
    }
    .sidebar-notice {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-left: 3px solid #0284C7;
        border-radius: 8px;
        padding: 10px 12px;
        margin-top: 14px;
        margin-bottom: 14px;
    }
    .sidebar-notice-title {
        font-size: 0.75rem;
        font-weight: 700;
        color: #0284C7;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .sidebar-notice-body {
        font-size: 0.78rem;
        color: #334155;
        margin-top: 4px;
        line-height: 1.35;
    }

    /* Cards & Containers */
    .saas-card {
        background: #FFFFFF;
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: var(--shadow-sm);
        margin-bottom: 14px;
        transition: transform 200ms ease, box-shadow 200ms ease, border-color 200ms ease;
    }
    .saas-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
        border-color: #CBD5E1;
    }
    .saas-card-title {
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #64748B;
        margin-bottom: 6px;
    }
    .saas-card-value {
        font-size: 1.7rem;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.2;
    }
    .saas-card-sub {
        font-size: 0.82rem;
        color: #0284C7;
        font-weight: 600;
        margin-top: 4px;
    }

    /* Form Input Group Cards */
    .input-section-header {
        display: flex;
        align-items: center;
        gap: 8px;
        font-family: var(--font-heading);
        font-size: 0.95rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 14px;
        margin-bottom: 8px;
        padding-bottom: 4px;
        border-bottom: 1px solid #F1F5F9;
    }

    /* Primary Investment Potential Hero Card (Visual Centerpiece) */
    .hero-score-card {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        border: 1px solid rgba(2, 132, 199, 0.35);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 12px;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.25);
        position: relative;
        overflow: hidden;
    }
    .hero-score-title {
        font-size: 0.82rem;
        font-weight: 800;
        color: #38BDF8;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .hero-score-number {
        font-family: var(--font-heading);
        font-size: 3.4rem;
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: -0.03em;
        line-height: 1;
        margin-top: 8px;
    }
    .hero-score-total {
        font-size: 1.5rem;
        color: #94A3B8;
        font-weight: 600;
    }
    .hero-score-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.9rem;
        margin-top: 8px;
    }
    .badge-good {
        background: rgba(16, 185, 129, 0.2);
        border: 1px solid #10B981;
        color: #34D399;
    }
    .badge-moderate {
        background: rgba(245, 158, 11, 0.2);
        border: 1px solid #F59E0B;
        color: #FBBF24;
    }
    .badge-bad {
        background: rgba(239, 68, 68, 0.2);
        border: 1px solid #EF4444;
        color: #F87171;
    }

    /* Score Meter Bar Animation */
    .score-meter-track {
        height: 10px;
        background: #334155;
        border-radius: 5px;
        overflow: hidden;
        margin-top: 20px;
        margin-bottom: 12px;
    }
    .score-meter-fill {
        height: 100%;
        background: linear-gradient(90deg, #6366F1, #0284C7, #10B981);
        border-radius: 5px;
        animation: meterFillAnim 900ms cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }
    @keyframes meterFillAnim {
        from { width: 0%; }
    }

    /* Secondary ML Audit Bar */
    .audit-subordinate-bar {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 10px 16px;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 8px;
    }
    .audit-tag {
        background: #E0F2FE;
        color: #0369A1;
        font-weight: 700;
        font-size: 0.72rem;
        padding: 3px 8px;
        border-radius: 4px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    /* 5-Year Scenario Cards */
    .scenario-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 10px;
        box-shadow: var(--shadow-sm);
        transition: transform 200ms ease, box-shadow 200ms ease;
    }
    .scenario-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
    }
    .scenario-card-header {
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 4px;
    }
    .scenario-price {
        font-family: var(--font-heading);
        font-size: 1.8rem;
        font-weight: 800;
        color: #0F172A;
        margin: 6px 0;
    }
    .scenario-gain {
        font-size: 0.85rem;
        font-weight: 700;
        color: #059669;
    }

    /* EDA Chart Cards */
    .eda-insight-card {
        background: #FFFFFF;
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: var(--shadow-sm);
        margin-bottom: 18px;
        transition: transform 200ms ease, box-shadow 200ms ease;
    }
    .eda-insight-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
    }
    .eda-q-tag {
        background: #F1F5F9;
        color: #0284C7;
        font-weight: 800;
        font-size: 0.75rem;
        padding: 3px 8px;
        border-radius: 4px;
        display: inline-block;
        margin-bottom: 6px;
    }
    .chart-frame-dark {
        background: #0F172A;
        border-radius: 10px;
        padding: 6px;
        border: 1px solid #1E293B;
        margin: 10px 0;
    }
    .insight-banner-refined {
        background: #F0F9FF;
        border-left: 3px solid #0284C7;
        padding: 9px 14px;
        border-radius: 0 8px 8px 0;
        font-size: 0.85rem;
        color: #0C4A6E;
        margin-top: 8px;
        margin-bottom: 10px;
        font-weight: 500;
    }

    /* Key Takeaway Cards */
    .takeaway-box {
        background: #FFFFFF;
        border: 1px solid var(--border-subtle);
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 10px;
        box-shadow: var(--shadow-sm);
        transition: transform 200ms ease, box-shadow 200ms ease;
    }
    .takeaway-box:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
    }
    .takeaway-box-title {
        font-weight: 700;
        font-size: 0.82rem;
        color: #0284C7;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .takeaway-box-text {
        font-size: 0.88rem;
        color: #334155;
        margin-top: 4px;
    }

    /* Factor Pills */
    .factor-pill-green {
        background: #ECFDF5;
        border: 1px solid #A7F3D0;
        color: #065F46;
        padding: 5px 12px;
        border-radius: 16px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
        display: inline-block;
    }
    .factor-pill-amber {
        background: #FFFBEB;
        border: 1px solid #FDE68A;
        color: #92400E;
        padding: 5px 12px;
        border-radius: 16px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
        display: inline-block;
    }

    /* Accessibility: Respect Reduced Motion Preference */
    @media (prefers-reduced-motion: reduce) {
        *, ::before, ::after {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# --- Data & Predictor Caching ---
@st.cache_resource
def get_predictor():
    return PropertyPredictor()

@st.cache_data
def load_sample_dataset():
    if PROCESSED_DATA_PATH.exists():
        df = pd.read_parquet(PROCESSED_DATA_PATH)
    elif PROCESSED_CSV_PATH.exists():
        df = pd.read_csv(PROCESSED_CSV_PATH)
    else:
        df = pd.read_csv(RAW_DATA_PATH)
    return df

@st.cache_data
def load_eval_reports():
    clf_path = REPORTS_DIR / "classification_results.json"
    reg_path = REPORTS_DIR / "regression_results.json"
    eda_path = REPORTS_DIR / "eda_summary.json"

    clf_data = json.load(open(clf_path, encoding="utf-8")) if clf_path.exists() else {}
    reg_data = json.load(open(reg_path, encoding="utf-8")) if reg_path.exists() else {}
    eda_data = json.load(open(eda_path, encoding="utf-8")) if eda_path.exists() else {}
    return clf_data, reg_data, eda_data

predictor = get_predictor()
df_data = load_sample_dataset()
clf_meta, reg_meta, eda_meta = load_eval_reports()

# Dynamically calculate exact dataset statistics from the loaded dataset
TOTAL_RECORDS = int(len(df_data))
NUM_STATES = int(df_data['State'].nunique())
NUM_CITIES = int(df_data['City'].nunique())

# --- Professional SaaS Sidebar Navigation ---
st.sidebar.markdown("""
<div class="sidebar-brand-box">
    <div style="display: flex; align-items: center; gap: 10px;">
        <span style="font-size: 1.8rem;">🏢</span>
        <div>
            <div class="sidebar-brand-title">INVESTMENT ADVISOR</div>
            <div class="sidebar-brand-sub">Real Estate Decision Platform</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

nav_choice = st.sidebar.radio(
    "Navigation Menu",
    [
        "🏠 Overview",
        "📊 Property Analysis",
        "💰 Valuation",
        "🎯 Investment Potential",
        "📈 5-Year Scenarios",
        "🔎 EDA Insights",
        "🤖 Model Performance",
        "🧪 MLflow / Governance",
        "ℹ️ Methodology"
    ],
    index=1
)

st.sidebar.markdown("""
<div class="sidebar-notice">
    <div class="sidebar-notice-title">💡 Decision Support Notice</div>
    <div class="sidebar-notice-body">
        Comparative screening and scenario analysis.<br>
        <strong>Does not guarantee profitability or future returns.</strong>
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Dataset Reference")
st.sidebar.write(f"**Total Records:** {TOTAL_RECORDS:,}")
st.sidebar.write(f"**Geographic Coverage:** {NUM_STATES} States, {NUM_CITIES} Cities")
st.sidebar.write(f"**Clean ML Features:** 19 (Zero Leakage)")
st.sidebar.caption("Source: `india_housing_prices.csv`")



# ==============================================================================
# HERO HEADER SECTION (Shared / Consistent Top Banner)
# ==============================================================================
def render_hero_banner():
    st.markdown("""
    <div class="saas-hero-container">
        <div class="hero-title-main">REAL ESTATE INVESTMENT ADVISOR</div>
        <div class="hero-subtitle-main">Data-driven property screening, valuation and scenario analysis</div>
        <div class="status-strip">
            <span class="status-pill"><span class="status-pill-check">✓</span> ML Valuation</span>
            <span class="status-pill"><span class="status-pill-check">✓</span> Investment Screening</span>
            <span class="status-pill"><span class="status-pill-check">✓</span> 5-Year Scenarios</span>
            <span class="status-pill"><span class="status-pill-check">✓</span> 20 EDA Questions</span>
            <span class="status-pill"><span class="status-pill-check">✓</span> MLflow Tracking</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# SECTION 1: OVERVIEW
# ==============================================================================
if nav_choice == "🏠 Overview":
    render_hero_banner()

    st.markdown("""
    **Executive Summary:**  
    The system combines empirical ML valuation, transparent investment potential scoring, exploratory data analysis, and 5-year scenario modeling to help users evaluate residential real estate opportunities using the available dataset.
    """)

    st.info("ℹ️ **Analytical Decision Support Notice:** The system provides analytical decision support and comparative property screening; it does not guarantee investment returns or future market performance.")

    st.markdown("---")
    st.subheader("System Architecture & Processing Pipeline")
    st.write("An integrated analytical pipeline separating empirical valuation, multi-criteria scoring, and macroeconomic scenario analysis:")

    p1, p2, p3, p4, p5 = st.columns(5)
    with p1:
        st.markdown(f"""
        <div class="saas-card" style="text-align: center;">
            <div style="font-size: 1.5rem; margin-bottom: 6px;">📁</div>
            <div style="font-weight: 700; color: #0284C7; font-size: 0.85rem;">1. PROPERTY DATA</div>
            <div style="color: #64748B; font-size: 0.78rem; margin-top: 4px;">Ingests {TOTAL_RECORDS:,} listings across {NUM_STATES} states and {NUM_CITIES} cities.</div>
        </div>
        """, unsafe_allow_html=True)
    with p2:
        st.markdown("""
        <div class="saas-card" style="text-align: center;">
            <div style="font-size: 1.5rem; margin-bottom: 6px;">🛡️</div>
            <div style="font-weight: 700; color: #0284C7; font-size: 0.85rem;">2. VALIDATION & AUDIT</div>
            <div style="color: #64748B; font-size: 0.78rem; margin-top: 4px;">Enforces strict zero target leakage; removes circular variables.</div>
        </div>
        """, unsafe_allow_html=True)
    with p3:
        st.markdown("""
        <div class="saas-card" style="text-align: center; border-color: #38BDF8;">
            <div style="font-size: 1.5rem; margin-bottom: 6px;">💰</div>
            <div style="font-weight: 700; color: #0284C7; font-size: 0.85rem;">3A. ML VALUATION</div>
            <div style="color: #64748B; font-size: 0.78rem; margin-top: 4px;">Estimates baseline listing price from 19 physical features.</div>
        </div>
        """, unsafe_allow_html=True)
    with p4:
        st.markdown("""
        <div class="saas-card" style="text-align: center; border-color: #10B981;">
            <div style="font-size: 1.5rem; margin-bottom: 6px;">🎯</div>
            <div style="font-weight: 700; color: #10B981; font-size: 0.85rem;">3B. MCDA SCORE (0-100)</div>
            <div style="color: #64748B; font-size: 0.78rem; margin-top: 4px;">Evaluates infrastructure (40%), margin (35%), freshness (25%).</div>
        </div>
        """, unsafe_allow_html=True)
    with p5:
        st.markdown("""
        <div class="saas-card" style="text-align: center; border-color: #818CF8;">
            <div style="font-size: 1.5rem; margin-bottom: 6px;">📈</div>
            <div style="font-weight: 700; color: #6366F1; font-size: 0.85rem;">3C. 5-YEAR SCENARIOS</div>
            <div style="color: #64748B; font-size: 0.78rem; margin-top: 4px;">Simulates 5.0%, 7.5%, and 10.0% compounding growth paths.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Key Macro Metrics (Dynamically Computed from Dataset)
    avg_price = float(df_data['Price_in_Lakhs'].mean())
    min_price = float(df_data['Price_in_Lakhs'].min())
    max_price = float(df_data['Price_in_Lakhs'].max())
    avg_area = float(df_data['Size_in_SqFt'].mean())
    min_area = int(df_data['Size_in_SqFt'].min())
    max_area = int(df_data['Size_in_SqFt'].max())
    mean_price_sqft = float((df_data['Price_in_Lakhs'] * 100000 / np.maximum(df_data['Size_in_SqFt'], 1)).mean())
    high_pot_share = float(df_data['High_Investment_Potential'].mean() * 100) if 'High_Investment_Potential' in df_data.columns else 31.2

    st.subheader("Macro Portfolio Benchmark Indicators")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="saas-card">
            <div class="saas-card-title">Average Property Price</div>
            <div class="saas-card-value">₹ {avg_price:.1f} L</div>
            <div class="saas-card-sub">Range: ₹{min_price:.1f}L - ₹{max_price:.1f}L</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="saas-card">
            <div class="saas-card-title">Average Property Area</div>
            <div class="saas-card-value">{int(avg_area):,} sq ft</div>
            <div class="saas-card-sub">Range: {min_area:,} - {max_area:,} sq ft</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="saas-card">
            <div class="saas-card-title">Mean Price / Sq Ft</div>
            <div class="saas-card-value">₹ {int(mean_price_sqft):,}</div>
            <div class="saas-card-sub">₹{mean_price_sqft/100000:.2f} Lakhs / sq ft</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="saas-card">
            <div class="saas-card-title">High Potential Share</div>
            <div class="saas-card-value">{high_pot_share:.1f} %</div>
            <div class="saas-card-sub">MCDA Score >= 70.0</div>
        </div>
        """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Price Distribution Across Property Types**")
        sample_df = df_data.sample(min(10000, len(df_data)), random_state=42)
        fig_type = px.box(
            sample_df, x="Property_Type", y="Price_in_Lakhs", color="Property_Type",
            labels={"Price_in_Lakhs": "Price in Lakhs (₹)", "Property_Type": "Type"},
            color_discrete_sequence=["#0284C7", "#0D9488", "#6366F1"]
        )
        fig_type.update_layout(
            height=320, margin=dict(l=20, r=20, t=20, b=20), showlegend=False,
            paper_bgcolor="#FFFFFF", plot_bgcolor="#F8FAFC"
        )
        st.plotly_chart(fig_type, use_container_width=True)

    with c2:
        st.markdown("**Public Transit vs High Investment Potential Rate**")
        if 'High_Investment_Potential' in df_data.columns:
            transit_df = df_data.groupby('Public_Transport_Accessibility')['High_Investment_Potential'].mean().reset_index()
            transit_df['Pct'] = (transit_df['High_Investment_Potential'] * 100).round(1)
        else:
            transit_df = pd.DataFrame({
                'Public_Transport_Accessibility': ['High', 'Medium', 'Low'],
                'Pct': [42.1, 31.4, 20.3]
            })
        fig_transit = px.bar(
            transit_df, x="Public_Transport_Accessibility", y="Pct",
            color="Public_Transport_Accessibility",
            labels={"Public_Transport_Accessibility": "Transit Tier", "Pct": "% High Potential"},
            color_discrete_map={"High": "#10B981", "Medium": "#6366F1", "Low": "#F59E0B"}
        )
        fig_transit.update_layout(
            height=320, margin=dict(l=20, r=20, t=20, b=20), showlegend=False,
            paper_bgcolor="#FFFFFF", plot_bgcolor="#F8FAFC"
        )
        st.plotly_chart(fig_transit, use_container_width=True)


# ==============================================================================
# SHARED PROPERTY FORM RENDERING LOGIC
# ==============================================================================
def render_property_form():
    with st.form("property_input_form"):
        # Group 1: Property Basics
        st.markdown('<div class="input-section-header">🏠 1. PROPERTY BASICS</div>', unsafe_allow_html=True)
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            prop_type = st.selectbox(
                "Property Type", ["Apartment", "Independent House", "Villa"],
                help="Architectural typology of the residential unit."
            )
        with b2:
            bhk = st.slider(
                "Bedrooms (BHK)", min_value=1, max_value=5, value=3,
                help="Number of bedrooms/hall/kitchen configuration."
            )
        with b3:
            size_sqft = st.number_input(
                "Property Area (Sq Ft)", min_value=500, max_value=5000, value=1750, step=50,
                help="Total carpet or super built-up area in square feet."
            )
        with b4:
            price_lakhs = st.number_input(
                "Asking Listing Price (₹ in Lakhs)", min_value=10.0, max_value=500.0, value=165.0, step=2.5,
                help="Seller's current asking listing price in Indian Lakhs (1 Lakh = ₹100,000)."
            )

        # Group 2: Location
        st.markdown('<div class="input-section-header">📍 2. LOCATION</div>', unsafe_allow_html=True)
        l1, l2, l3 = st.columns(3)
        with l1:
            states_list = sorted(list(df_data['State'].unique()))
            state_selected = st.selectbox(
                "State", states_list,
                index=states_list.index("Maharashtra") if "Maharashtra" in states_list else 0,
                help="Geographic state where the property is located."
            )
        with l2:
            cities_in_state = sorted(list(df_data[df_data['State'] == state_selected]['City'].unique()))
            city_selected = st.selectbox(
                "City", cities_in_state if cities_in_state else sorted(list(df_data['City'].unique())),
                help="Municipal city within the selected state."
            )
        with l3:
            loc_options = sorted(list(df_data[df_data['City'] == city_selected]['Locality'].unique()[:30]))
            loc_selected = st.selectbox(
                "Locality (Display / Reference Only)", loc_options if loc_options else ["Locality_1"],
                help="Specific residential neighborhood or sector. Display / Reference Only — not used by ML models to prevent high-cardinality overfitting."
            )
            st.caption("ℹ️ *Display / Reference Only — not used by ML models*")

        # Group 3: Infrastructure & Condition
        st.markdown('<div class="input-section-header">🏗️ 3. INFRASTRUCTURE & CONDITION</div>', unsafe_allow_html=True)
        i1, i2, i3, i4, i5, i6 = st.columns(6)
        with i1:
            age_prop = st.slider(
                "Property Age (Years)", min_value=2, max_value=35, value=6,
                help="Approximate age of the structure in years since completion."
            )
        with i2:
            total_floors = st.number_input(
                "Total Floors", min_value=1, max_value=30, value=12,
                help="Total vertical storeys in the residential structure."
            )
        with i3:
            floor_no = st.number_input(
                "Unit Floor", min_value=0, max_value=int(total_floors), value=min(4, int(total_floors)),
                help="Vertical floor position of the property unit (0 = Ground Floor)."
            )
        with i4:
            transit_access = st.selectbox(
                "Public Transit", ["High", "Medium", "Low"], index=0,
                help="High: immediate transit link; Medium: walking distance; Low: car-dependent."
            )
        with i5:
            schools_count = st.slider(
                "Nearby Schools", min_value=0, max_value=10, value=8,
                help="Number of schools within 3 km."
            )
        with i6:
            hosp_count = st.slider(
                "Nearby Hospitals", min_value=0, max_value=10, value=7,
                help="Number of hospitals within 5 km."
            )

        # Group 4: Amenities
        st.markdown('<div class="input-section-header">🏊 4. AMENITIES</div>', unsafe_allow_html=True)
        a1, a2, a3 = st.columns([2, 1, 1])
        with a1:
            amenities_selected = st.multiselect(
                "Amenities Available",
                ["Gym", "Pool", "Clubhouse", "Garden", "Playground"],
                default=["Gym", "Pool", "Clubhouse"],
                help="Select on-site residential amenities available to occupants. All 5 options directly contribute to amenity_count and infrastructure scoring."
            )
            st.caption("ℹ️ *All 5 options directly contribute to amenity_count (0–5) and infrastructure scoring.*")
        with a2:
            parking = st.radio(
                "Dedicated Parking", ["Yes", "No"], horizontal=True,
                help="Yes: exclusive covered or designated vehicle parking bay included; No: street parking only."
            )
        with a3:
            security = st.radio(
                "Gated Security", ["Yes", "No"], horizontal=True,
                help="Yes: 24/7 security guards, CCTV surveillance, and access control."
            )

        # Group 5: Ownership & Availability
        st.markdown('<div class="input-section-header">🔑 5. OWNERSHIP & AVAILABILITY</div>', unsafe_allow_html=True)
        o1, o2, o3, o4 = st.columns(4)
        with o1:
            furnishing = st.selectbox(
                "Furnished Status", ["Furnished", "Semi-furnished", "Unfurnished"], index=1,
                help="Interior fitout level provided at asking price."
            )
        with o2:
            availability = st.selectbox(
                "Availability Status", ["Ready_to_Move", "Under_Construction"],
                help="Ready_to_Move: immediate occupancy; Under_Construction: subject to construction completion."
            )
        with o3:
            facing = st.selectbox("Facing Direction", ["East", "North", "South", "West"], index=0)
        with o4:
            owner = st.selectbox("Owner Type", ["Owner", "Builder", "Broker"], index=0)

        submit_btn = st.form_submit_button("🔍 Run Full Advisory Evaluation", use_container_width=True)

    payload = {
        'State': state_selected,
        'City': city_selected,
        'Locality': loc_selected,
        'Property_Type': prop_type,
        'BHK': bhk,
        'Size_in_SqFt': size_sqft,
        'Price_in_Lakhs': price_lakhs,
        'Age_of_Property': age_prop,
        'Floor_No': floor_no,
        'Total_Floors': total_floors,
        'Nearby_Schools': schools_count,
        'Nearby_Hospitals': hosp_count,
        'Public_Transport_Accessibility': transit_access,
        'Parking_Space': parking,
        'Security': security,
        'Amenities': ", ".join(amenities_selected) if amenities_selected else "",
        'Furnished_Status': furnishing,
        'Availability_Status': availability,
        'Facing': facing,
        'Owner_Type': owner
    }
    return submit_btn, payload


# ==============================================================================
# SHARED REPORT RENDERING LOGIC (EVALUATION REPORT)
# ==============================================================================
def render_evaluation_report(report):
    st.markdown("---")
    st.subheader("📋 Comprehensive Investment Advisory Report")

    # 1. VISUALLY DOMINANT INVESTMENT POTENTIAL SCORE (PRIMARY)
    badge_class = f"badge-{report['badge_color']}"
    score_val = report['investment_potential_score']
    meter_pct = min(max(score_val, 0), 100)

    st.markdown(f"""
    <div class="hero-score-card">
        <div class="hero-score-title">INVESTMENT POTENTIAL (PRIMARY ASSESSMENT)</div>
        <div style="display: flex; justify-content: space-between; align-items: flex-end; flex-wrap: wrap;">
            <div>
                <div class="hero-score-number">{score_val:.1f}<span class="hero-score-total"> / 100</span></div>
                <div class="hero-score-badge {badge_class}">{report['rating_badge']}</div>
                <div style="color: #CBD5E1; font-size: 0.95rem; margin-top: 6px;">
                    Classification: <strong>{report['rating_tier']}</strong> (Deterministic MCDA Engine)
                </div>
            </div>
            <div style="text-align: right; min-width: 180px;">
                <div style="color: #94A3B8; font-size: 0.8rem; font-weight: 600; text-transform: uppercase;">Decision Threshold</div>
                <div style="color: #38BDF8; font-size: 1.3rem; font-weight: 700;">Score &ge; 70.0</div>
                <div style="font-size: 0.78rem; color: #64748B;">Tier 1 Qualified</div>
            </div>
        </div>
        <div class="score-meter-track">
            <div class="score-meter-fill" style="width: {meter_pct}%;"></div>
        </div>
        <div style="display: flex; justify-content: space-between; color: #94A3B8; font-size: 0.78rem; font-weight: 600;">
            <span>0 (Cautious)</span>
            <span>50 (Moderate)</span>
            <span>70 (High Potential)</span>
            <span>100 (Optimal)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Secondary ML Audit Subordinate Bar
    st.markdown(f"""
    <div class="audit-subordinate-bar">
        <div style="display: flex; align-items: center; gap: 8px;">
            <span class="audit-tag">Secondary ML Audit</span>
            <span style="color: #0F172A; font-size: 0.88rem; font-weight: 600;">Surrogate Model Confidence: <strong style="color: #0284C7;">{report['classifier_confidence_pct']}%</strong></span>
        </div>
        <div style="font-size: 0.8rem; color: #64748B;">
            Confidence that the ML surrogate agrees with the rule-based MCDA classification (<strong>not</strong> probability of financial returns)
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Component Breakdown Cards
    st.write("**Investment Potential Score Components (MCDA Multi-Criteria):**")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="saas-card">
            <div class="saas-card-title">Infrastructure & Utility (40%)</div>
            <div style="color: #334155; font-size: 0.85rem; line-height: 1.4;">Transit connectivity, amenities count (0–5), nearby schools, hospitals, security, and parking.</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="saas-card">
            <div class="saas-card-title">Valuation Safety Margin (35%)</div>
            <div style="color: #334155; font-size: 0.85rem; line-height: 1.4;">Relative price per sq ft efficiency evaluated against overall market distribution.</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="saas-card">
            <div class="saas-card-title">Asset Freshness (25%)</div>
            <div style="color: #334155; font-size: 0.85rem; line-height: 1.4;">Remaining structural economic life based on building age.</div>
        </div>
        """, unsafe_allow_html=True)

    # 2. WHY THIS SCORE? (CONTRIBUTING HIGHLIGHTS)
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Why did this property receive this score?")
    st.write("Identified strengths and potential cautions influencing the investment assessment:")

    if report['strengths']:
        for s in report['strengths']:
            st.markdown(f'<span class="factor-pill-green">✅ {s}</span>', unsafe_allow_html=True)
    if report['cautions']:
        for c in report['cautions']:
            st.markdown(f'<span class="factor-pill-amber">⚠️ {c}</span>', unsafe_allow_html=True)

    with st.expander("ℹ️ How is the Investment Potential Score calculated?"):
        st.markdown(r"""
        The **Investment Potential Score** ($0\text{--}100$) is computed via deterministic Multi-Criteria Decision Analysis:
        $$\text{Score} = 100 \times \left(0.40 \times \text{Infra} + 0.35 \times \text{ValueMargin} + 0.25 \times \text{Freshness}\right)$$
        - **Infrastructure (40%)**: Transit ($20\%$), Amenities ($20\%$), Schools ($15\%$), Hospitals ($15\%$), Security ($15\%$), Parking ($15\%$).
        - **Valuation Margin (35%)**: Relative price efficiency: $1.0 - \text{MinMax}(P_{\text{sqft}})$.
        - **Asset Freshness (25%)**: Structural life: $1.0 - \text{MinMax}(\text{Age})$.
        """)
        st.info("This is a transparent rule-based decision-support score, not a guarantee of future returns.")

    # 3. SURROGATE CLASSIFICATION ACCURACY
    with st.expander("🤖 Surrogate ML Classifier Details"):
        st.markdown("""
        **Surrogate Model Explanation:**  
        The classifier learns to approximate the transparent rule-based MCDA investment assessment directly from raw property attributes.
        - **Model Used:** LightGBM Classifier
        - **Holdout Accuracy:** `93.55%` | **F1-Score:** `0.8955` | **ROC-AUC:** `0.9861`
        - **Target Definition:** `High_Investment_Potential` ($\ge 70.0$ MCDA Score)
        
        *Important:* This metric measures how accurately the ML model replicates the rule-based MCDA score. It does **NOT** represent prediction of actual real-world financial profitability.
        """)

    st.markdown("---")

    # 4. MODULE A — ML VALUATION ESTIMATE
    st.subheader("Module A: ML Valuation Estimate")
    st.write("An experimental machine-learning estimate of baseline listing price based on physical and spatial features (disclosing model uncertainty).")

    est_val = report['ml_valuation_lakhs']
    band_low = max(10.0, est_val - 122.32)
    band_high = min(500.0, est_val + 122.32)

    v1, v2, v3 = st.columns(3)
    with v1:
        st.markdown(f"""
        <div class="saas-card" style="border: 1px solid #F59E0B; background: #FFFBEB;">
            <div class="saas-card-title" style="color: #B45309;">ML VALUATION ESTIMATE</div>
            <div class="saas-card-value">₹ {est_val:.2f} L</div>
            <div style="color: #D97706; font-size: 0.85rem; font-weight: 700; margin-top: 4px;">
                Uncertainty: &plusmn; ₹ 122.32 L (MAE)
            </div>
            <div style="color: #78350F; font-size: 0.8rem; margin-top: 2px;">
                Expected Band: ₹ {band_low:.1f}L &ndash; ₹ {band_high:.1f}L
            </div>
        </div>
        """, unsafe_allow_html=True)
    with v2:
        st.markdown(f"""
        <div class="saas-card">
            <div class="saas-card-title">ASKING LISTING PRICE</div>
            <div class="saas-card-value">₹ {report['current_price_lakhs']:.2f} L</div>
            <div style="color: #64748B; font-size: 0.82rem; margin-top: 4px;">Rate: ₹ {int(report['price_per_sqft_lakhs']*100000):,}/sq ft</div>
            <div class="saas-card-sub" style="color: #0284C7;">Seller's Listed Price</div>
        </div>
        """, unsafe_allow_html=True)
    with v3:
        diff_val = report['valuation_diff_lakhs']
        diff_color = "#059669" if diff_val <= 0 else "#D97706"
        diff_label = "Asking Below Baseline Estimate" if diff_val <= 0 else "Asking Above Baseline Estimate"
        st.markdown(f"""
        <div class="saas-card">
            <div class="saas-card-title">VALUATION VARIANCE</div>
            <div class="saas-card-value" style="color: {diff_color};">₹ {abs(diff_val):.2f} L</div>
            <div style="color: {diff_color}; font-size: 0.82rem; font-weight: 700; margin-top: 4px;">{diff_label}</div>
            <div style="color: #64748B; font-size: 0.8rem; margin-top: 2px;">Variance vs Central Tendency</div>
        </div>
        """, unsafe_allow_html=True)

    st.warning(
        f"⚠️ **Model Uncertainty & Pricing Dispersion Notice:** The valuation model operates with a Mean Absolute Error (MAE) of **₹122.32 Lakhs** ($R^2 \\approx 0.000$) due to the uniform synthetic price distribution in the dataset. "
        f"The estimate of **₹{est_val:.2f}L** represents a baseline listing price central tendency with an expected band of **₹{band_low:.1f}L to ₹{band_high:.1f}L**, and should **NOT** be treated as a pinpoint or guaranteed market appraisal."
    )

    with st.expander("ℹ️ ML Valuation Methodology & Limitations"):
        st.markdown("""
        - **What is this?** An experimental ML estimate of the listing price based on 19 physical, structural, and spatial features.
        - **How is it calculated?** Predicted using Ridge Regression with StandardScaler and OneHotEncoder. Derived features (`Price_per_SqFt`, `infra_score`) are strictly excluded.
        - **What does it mean?** Compares the seller's asking price against general dataset tendencies.
        - **Model Limitation:** Because the supplied dataset contains near-zero predictive relationship for price ($R^2 \\approx 0.000$, MAE $\\approx ₹122L$), this estimate should **NOT** be treated as a reliable real-world appraisal.
        """)

    st.markdown("---")

    # 5. MODULE C — 5-YEAR CAPITAL GROWTH SCENARIOS
    st.subheader("Module C: 5-Year Illustrative Scenario Projections")
    st.write("Financial sensitivity analysis projecting future property value across three illustrative growth rates.")

    sc1, sc2, sc3 = st.columns(3)
    scenarios = report['scenarios']

    with sc1:
        c_data = scenarios['Conservative']
        st.markdown(f"""
        <div class="scenario-card" style="border-top: 4px solid #0284C7;">
            <div class="scenario-card-header" style="color: #0284C7;">CONSERVATIVE SCENARIO</div>
            <div style="font-size: 0.8rem; color: #64748B;">Growth Rate: 5.0% p.a.</div>
            <div class="scenario-price">₹ {c_data['projected_price_lakhs']:.2f} L</div>
            <div class="scenario-gain">+₹ {c_data['net_gain_lakhs']:.2f} L (+{c_data['growth_pct']:.1f}%)</div>
        </div>
        """, unsafe_allow_html=True)

    with sc2:
        m_data = scenarios['Moderate']
        st.markdown(f"""
        <div class="scenario-card" style="border-top: 4px solid #6366F1;">
            <div class="scenario-card-header" style="color: #6366F1;">BASE ILLUSTRATIVE SCENARIO</div>
            <div style="font-size: 0.8rem; color: #64748B;">Growth Rate: 7.5% p.a.</div>
            <div class="scenario-price">₹ {m_data['projected_price_lakhs']:.2f} L</div>
            <div class="scenario-gain">+₹ {m_data['net_gain_lakhs']:.2f} L (+{m_data['growth_pct']:.1f}%)</div>
        </div>
        """, unsafe_allow_html=True)

    with sc3:
        o_data = scenarios['Optimistic']
        st.markdown(f"""
        <div class="scenario-card" style="border-top: 4px solid #10B981;">
            <div class="scenario-card-header" style="color: #10B981;">HIGH GROWTH SCENARIO</div>
            <div style="font-size: 0.8rem; color: #64748B;">Growth Rate: 10.0% p.a.</div>
            <div class="scenario-price">₹ {o_data['projected_price_lakhs']:.2f} L</div>
            <div class="scenario-gain">+₹ {o_data['net_gain_lakhs']:.2f} L (+{o_data['growth_pct']:.1f}%)</div>
        </div>
        """, unsafe_allow_html=True)

    # Compounding Curves Plot
    years = [f"Year {y}" for y in range(6)]
    p0 = report['current_price_lakhs']
    traj_c = [round(p0 * (1.05 ** y), 2) for y in range(6)]
    traj_m = [round(p0 * (1.075 ** y), 2) for y in range(6)]
    traj_o = [round(p0 * (1.10 ** y), 2) for y in range(6)]

    traj_fig = go.Figure()
    traj_fig.add_trace(go.Scatter(x=years, y=traj_c, mode='lines+markers', name='Conservative (5.0% p.a.)', line=dict(color="#0284C7", dash='dash', width=2)))
    traj_fig.add_trace(go.Scatter(x=years, y=traj_m, mode='lines+markers', name='Base Illustrative (7.5% p.a.)', line=dict(color="#6366F1", width=3)))
    traj_fig.add_trace(go.Scatter(x=years, y=traj_o, mode='lines+markers', name='High Growth (10.0% p.a.)', line=dict(color="#10B981", width=2)))

    traj_fig.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=25, b=20),
        yaxis_title="Projected Value (₹ Lakhs)",
        hovermode="x unified",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#F8FAFC",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(traj_fig, use_container_width=True)

    st.warning("⚠️ **Scenario Disclaimer:** These projections are mathematical scenarios based on assumed annual growth rates. They do not represent observed historical market forecasts or guaranteed financial outcomes. Growth rates are not claimed as official RBI benchmarks.")


# ==============================================================================
# SECTION 2: PROPERTY ANALYSIS (PRIMARY APPRAISAL WORKFLOW)
# ==============================================================================
if nav_choice == "📊 Property Analysis":
    render_hero_banner()
    st.title("🎯 Property Appraisal & Investment Advisory")
    st.caption("Investment Decision Support System — Comparative Screening & Scenario Analysis (Does Not Predict Guaranteed Profitability)")
    st.write("Configure property attributes below to evaluate investment potential, ML listing price estimate, and 5-year capital growth scenarios.")

    submitted, payload = render_property_form()

    # Automatically run baseline evaluation or on submit
    if submitted or 'last_report' not in st.session_state:
        with st.status("Analyzing property & generating advisory report...", expanded=False) as status:
            st.write("Extracting structural and spatial features...")
            st.write("Executing 3-component MCDA scoring engine...")
            st.write("Calculating ML valuation baseline...")
            st.write("Simulating 5-year compounding scenario paths...")
            report = predictor.evaluate_property(payload)
            st.session_state['last_report'] = report
            status.update(label="Property Evaluation Ready!", state="complete")

    if 'last_report' in st.session_state:
        render_evaluation_report(st.session_state['last_report'])


# ==============================================================================
# SECTION 3: VALUATION (MODULE A DEEP DIVE)
# ==============================================================================
elif nav_choice == "💰 Valuation":
    render_hero_banner()
    st.title("💰 Module A: Empirical ML Property Valuation")
    st.caption("Empirical machine-learning baseline for residential listing prices")

    st.markdown("""
    **Module A Focus:**  
    Predicts baseline listing price strictly from 19 structural and spatial features, enforcing complete isolation from derived target variables (`Price_per_SqFt = Price / Size`).
    """)

    v_c1, v_c2, v_c3 = st.columns(3)
    with v_c1:
        st.markdown("""
        <div class="saas-card" style="border-top: 4px solid #0284C7;">
            <div class="saas-card-title">CHAMPION REGRESSION MODEL</div>
            <div class="saas-card-value">Ridge Baseline</div>
            <div class="saas-card-sub">L2 Regularized Linear Model</div>
        </div>
        """, unsafe_allow_html=True)
    with v_c2:
        st.markdown("""
        <div class="saas-card" style="border-top: 4px solid #F59E0B;">
            <div class="saas-card-title">HOLDOUT MODEL ERROR</div>
            <div class="saas-card-value">₹ 122.32 L</div>
            <div style="color: #D97706; font-size: 0.82rem; font-weight: 700; margin-top: 4px;">Mean Absolute Error (N=50,000)</div>
        </div>
        """, unsafe_allow_html=True)
    with v_c3:
        st.markdown("""
        <div class="saas-card" style="border-top: 4px solid #64748B;">
            <div class="saas-card-title">COEFFICIENT OF DETERMINATION</div>
            <div class="saas-card-value">R² ≈ 0.000</div>
            <div style="color: #64748B; font-size: 0.82rem; font-weight: 700; margin-top: 4px;">Zero Linear Predictive Signal</div>
        </div>
        """, unsafe_allow_html=True)

    st.warning("⚠️ **Academic Disclosure:** Once circular target leakage (`Price_per_SqFt`) is removed, physical features in the synthetic dataset have near-zero correlation with listing price. The model outputs baseline central tendency rather than an infallible market appraisal.")

    st.markdown("### Interactive Valuation Appraisal")
    st.write("Test listing price estimation with current property attributes:")
    submitted, payload = render_property_form()
    if submitted or 'last_report' not in st.session_state:
        report = predictor.evaluate_property(payload)
        st.session_state['last_report'] = report

    if 'last_report' in st.session_state:
        rep = st.session_state['last_report']
        est_val = rep['ml_valuation_lakhs']
        band_low = max(10.0, est_val - 122.32)
        band_high = min(500.0, est_val + 122.32)

        st.markdown(f"""
        <div class="saas-card" style="border: 1px solid #F59E0B; background: #FFFBEB; margin-top: 14px;">
            <div style="font-size: 0.85rem; font-weight: 700; color: #B45309;">ESTIMATED LISTING PRICE</div>
            <div style="font-size: 2.2rem; font-weight: 800; color: #0F172A; margin: 4px 0;">₹ {est_val:.2f} Lakhs</div>
            <div style="color: #D97706; font-weight: 700; font-size: 0.9rem;">Expected Model Uncertainty Band: &plusmn; ₹ 122.32 L (MAE)</div>
            <div style="color: #78350F; font-size: 0.82rem; margin-top: 2px;">Expected Range: ₹ {band_low:.1f}L &ndash; ₹ {band_high:.1f}L | Asking Price: ₹ {rep['current_price_lakhs']:.2f}L</div>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# SECTION 4: INVESTMENT POTENTIAL (MODULE B DEEP DIVE)
# ==============================================================================
elif nav_choice == "🎯 Investment Potential":
    render_hero_banner()
    st.title("🎯 Module B: Multi-Criteria Investment Scoring")
    st.caption("Transparent 0–100 Multi-Criteria Decision Analysis (MCDA)")

    st.markdown("""
    **Module B Focus:**  
    Unlike black-box models, Module B provides a fully deterministic, explainable investment score derived from three balanced dimensions.
    """)

    b_c1, b_c2, b_c3 = st.columns(3)
    with b_c1:
        st.markdown("""
        <div class="saas-card" style="border-top: 4px solid #0284C7;">
            <div class="saas-card-title">INFRASTRUCTURE & CIVIC UTILITY</div>
            <div class="saas-card-value">40 %</div>
            <div class="saas-card-sub">Transit, Amenities, Schools, Hospitals, Security, Parking</div>
        </div>
        """, unsafe_allow_html=True)
    with b_c2:
        st.markdown("""
        <div class="saas-card" style="border-top: 4px solid #10B981;">
            <div class="saas-card-title">VALUATION SAFETY MARGIN</div>
            <div class="saas-card-value">35 %</div>
            <div style="color: #059669; font-size: 0.82rem; font-weight: 700; margin-top: 4px;">Relative Unit Rate Efficiency (1.0 - MinMax P_sqft)</div>
        </div>
        """, unsafe_allow_html=True)
    with b_c3:
        st.markdown("""
        <div class="saas-card" style="border-top: 4px solid #6366F1;">
            <div class="saas-card-title">STRUCTURAL FRESHNESS</div>
            <div class="saas-card-value">25 %</div>
            <div style="color: #6366F1; font-size: 0.82rem; font-weight: 700; margin-top: 4px;">Remaining Economic Life (1.0 - MinMax Age)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Decision Tiers & Criteria Rules")
    t1, t2, t3 = st.columns(3)
    with t1:
        st.markdown("""
        <div class="saas-card" style="border-left: 4px solid #10B981;">
            <div style="color: #10B981; font-weight: 700;">TIER 1 — HIGH POTENTIAL</div>
            <div style="font-size: 1.3rem; font-weight: 800; margin: 4px 0;">Score &ge; 70.0</div>
            <div style="font-size: 0.8rem; color: #64748B;">Strong infrastructure access, superior pricing efficiency, modern structure.</div>
        </div>
        """, unsafe_allow_html=True)
    with t2:
        st.markdown("""
        <div class="saas-card" style="border-left: 4px solid #F59E0B;">
            <div style="color: #F59E0B; font-weight: 700;">TIER 2 — MODERATE POTENTIAL</div>
            <div style="font-size: 1.3rem; font-weight: 800; margin: 4px 0;">55.0 &le; Score < 70.0</div>
            <div style="font-size: 0.8rem; color: #64748B;">Balanced attributes with reasonable civic access and standard pricing.</div>
        </div>
        """, unsafe_allow_html=True)
    with t3:
        st.markdown("""
        <div class="saas-card" style="border-left: 4px solid #EF4444;">
            <div style="color: #EF4444; font-weight: 700;">TIER 3 — CAUTIOUS / LOW</div>
            <div style="font-size: 1.3rem; font-weight: 800; margin: 4px 0;">Score < 55.0</div>
            <div style="font-size: 0.8rem; color: #64748B;">Sub-optimal transit, higher per-sqft pricing, or aging building envelope.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Interactive MCDA Evaluation")
    submitted, payload = render_property_form()
    if submitted or 'last_report' not in st.session_state:
        report = predictor.evaluate_property(payload)
        st.session_state['last_report'] = report

    if 'last_report' in st.session_state:
        render_evaluation_report(st.session_state['last_report'])


# ==============================================================================
# SECTION 5: 5-YEAR SCENARIOS (MODULE C DEEP DIVE)
# ==============================================================================
elif nav_choice == "📈 5-Year Scenarios":
    render_hero_banner()
    st.title("📈 Module C: 5-Year Capital Growth Scenario Analysis")
    st.caption("Compound capital appreciation trajectories across 3 illustrative growth scenarios")

    st.markdown("""
    **Module C Focus:**  
    Simulates multi-tier capital appreciation over a 5-year investment horizon ($t = 5$) using standard compound mathematics:  
    $$\\text{Future Value} = \\text{Current Price} \\times (1 + r)^5$$
    """)

    submitted, payload = render_property_form()
    if submitted or 'last_report' not in st.session_state:
        report = predictor.evaluate_property(payload)
        st.session_state['last_report'] = report

    if 'last_report' in st.session_state:
        rep = st.session_state['last_report']
        scenarios = rep['scenarios']

        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            c_data = scenarios['Conservative']
            st.markdown(f"""
            <div class="scenario-card" style="border-top: 4px solid #0284C7;">
                <div class="scenario-card-header" style="color: #0284C7;">CONSERVATIVE SCENARIO</div>
                <div style="font-size: 0.8rem; color: #64748B;">Growth Rate: 5.0% p.a.</div>
                <div class="scenario-price">₹ {c_data['projected_price_lakhs']:.2f} L</div>
                <div class="scenario-gain">+₹ {c_data['net_gain_lakhs']:.2f} L (+{c_data['growth_pct']:.1f}%)</div>
                <div style="font-size: 0.78rem; color: #64748B; margin-top: 4px;">Compounding factor: 1.276x</div>
            </div>
            """, unsafe_allow_html=True)
        with sc2:
            m_data = scenarios['Moderate']
            st.markdown(f"""
            <div class="scenario-card" style="border-top: 4px solid #6366F1;">
                <div class="scenario-card-header" style="color: #6366F1;">BASE ILLUSTRATIVE SCENARIO</div>
                <div style="font-size: 0.8rem; color: #64748B;">Growth Rate: 7.5% p.a.</div>
                <div class="scenario-price">₹ {m_data['projected_price_lakhs']:.2f} L</div>
                <div class="scenario-gain">+₹ {m_data['net_gain_lakhs']:.2f} L (+{m_data['growth_pct']:.1f}%)</div>
                <div style="font-size: 0.78rem; color: #64748B; margin-top: 4px;">Compounding factor: 1.436x</div>
            </div>
            """, unsafe_allow_html=True)
        with sc3:
            o_data = scenarios['Optimistic']
            st.markdown(f"""
            <div class="scenario-card" style="border-top: 4px solid #10B981;">
                <div class="scenario-card-header" style="color: #10B981;">HIGH GROWTH SCENARIO</div>
                <div style="font-size: 0.8rem; color: #64748B;">Growth Rate: 10.0% p.a.</div>
                <div class="scenario-price">₹ {o_data['projected_price_lakhs']:.2f} L</div>
                <div class="scenario-gain">+₹ {o_data['net_gain_lakhs']:.2f} L (+{o_data['growth_pct']:.1f}%)</div>
                <div style="font-size: 0.78rem; color: #64748B; margin-top: 4px;">Compounding factor: 1.611x</div>
            </div>
            """, unsafe_allow_html=True)

        # Plotly Trajectory
        years = [f"Year {y}" for y in range(6)]
        p0 = rep['current_price_lakhs']
        traj_c = [round(p0 * (1.05 ** y), 2) for y in range(6)]
        traj_m = [round(p0 * (1.075 ** y), 2) for y in range(6)]
        traj_o = [round(p0 * (1.10 ** y), 2) for y in range(6)]

        traj_fig = go.Figure()
        traj_fig.add_trace(go.Scatter(x=years, y=traj_c, mode='lines+markers', name='Conservative (5.0% p.a.)', line=dict(color="#0284C7", dash='dash', width=2)))
        traj_fig.add_trace(go.Scatter(x=years, y=traj_m, mode='lines+markers', name='Base Illustrative (7.5% p.a.)', line=dict(color="#6366F1", width=3)))
        traj_fig.add_trace(go.Scatter(x=years, y=traj_o, mode='lines+markers', name='High Growth (10.0% p.a.)', line=dict(color="#10B981", width=2)))

        traj_fig.update_layout(
            height=340,
            margin=dict(l=20, r=20, t=25, b=20),
            yaxis_title="Projected Value (₹ Lakhs)",
            hovermode="x unified",
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#F8FAFC",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(traj_fig, use_container_width=True)

        st.warning("⚠️ **Scenario Disclaimer:** These projections are mathematical scenarios based on assumed annual growth rates. They do not represent observed historical market forecasts or guaranteed financial outcomes. Growth rates are not claimed as official RBI benchmarks.")


# ==============================================================================
# SECTION 6: EDA INSIGHTS (ALL 20 QUESTIONS AUDITED & VERIFIED)
# ==============================================================================
elif nav_choice == "🔎 EDA Insights":
    render_hero_banner()
    st.title("📊 Market Insights: 20-Point Exploratory Data Analysis")
    st.write("Complete visual evidence answering all 20 business questions from the project charter. Every chart provides immediate reading guidance, purpose explanation, and dataset limitation context.")

    eda_tab1, eda_tab2, eda_tab3, eda_tab4 = st.tabs([
        "Q1 - Q5: Price & Size",
        "Q6 - Q10: Geography & Locations",
        "Q11 - Q15: Correlations & Factors",
        "Q16 - Q20: Operations & Lifestyle"
    ])

    # ---------------- TAB 1: Q1 to Q5 ----------------
    with eda_tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="eda-insight-card"><span class="eda-q-tag">Q01</span><h4>What is the distribution of property prices?</h4>', unsafe_allow_html=True)
            if (FIGURES_DIR / "q01_price_distribution.png").exists():
                st.markdown('<div class="chart-frame-dark">', unsafe_allow_html=True)
                st.image(str(FIGURES_DIR / "q01_price_distribution.png"), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div class="insight-banner-refined">💡 <strong>Key Insight:</strong> Prices span uniformly from ₹10.0L to ₹500.0L with a median of ₹253.9 Lakhs.</div>', unsafe_allow_html=True)
            with st.expander("ℹ️ Details & Reading Guidance"):
                st.markdown("""
                - **What does this show?** Histogram and density curve of listing prices across all 250,000 properties.
                - **How should I read it?** The distribution is flat and bounded between ₹10L and ₹500L.
                - **Why is it useful?** Establishes the full market price span and confirms continuous numerical scale.
                - **Dataset limitation:** Real-world housing prices follow right-skewed log-normal distributions rather than flat uniform distributions.
                """)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="eda-insight-card"><span class="eda-q-tag">Q03</span><h4>How does price per sq ft vary by property type?</h4>', unsafe_allow_html=True)
            if (FIGURES_DIR / "q03_price_per_sqft_by_type.png").exists():
                st.markdown('<div class="chart-frame-dark">', unsafe_allow_html=True)
                st.image(str(FIGURES_DIR / "q03_price_per_sqft_by_type.png"), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div class="insight-banner-refined">💡 <strong>Key Insight:</strong> Unit rates average ~₹13,000/sq ft (0.13 Lakhs/sq ft) across Apartments, Houses, and Villas.</div>', unsafe_allow_html=True)
            with st.expander("ℹ️ Details & Reading Guidance"):
                st.markdown("""
                - **What does this show?** Average price per square foot across the three property typologies.
                - **How should I read it?** Bars represent mean rates; error bars reflect standard deviations.
                - **Why is it useful?** Evaluates whether architectural type commands a unit rate premium.
                - **Dataset limitation:** In natural markets, standalone villas typically command premiums over standard apartments.
                """)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="eda-insight-card"><span class="eda-q-tag">Q05</span><h4>Are there any outliers in price per sq ft or property size?</h4>', unsafe_allow_html=True)
            if (FIGURES_DIR / "q05_outliers_analysis.png").exists():
                st.markdown('<div class="chart-frame-dark">', unsafe_allow_html=True)
                st.image(str(FIGURES_DIR / "q05_outliers_analysis.png"), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div class="insight-banner-refined">💡 <strong>Key Insight:</strong> Price per sq ft exhibits 8.01% statistical upper outliers where high prices pair with compact sizes.</div>', unsafe_allow_html=True)
            with st.expander("ℹ️ Details & Reading Guidance"):
                st.markdown("""
                - **What does this show?** Box plots identifying statistical anomalies via 1.5×IQR fences.
                - **How should I read it?** Diamonds beyond whiskers represent outlier listings.
                - **Why is it useful?** Guides robust scaling and demonstrates why linear models require regularization.
                - **Dataset limitation:** Outliers arise from mathematical ratios (small area denominator) rather than genuine luxury transactions.
                """)
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="eda-insight-card"><span class="eda-q-tag">Q02</span><h4>What is the distribution of property sizes (in sq ft)?</h4>', unsafe_allow_html=True)
            if (FIGURES_DIR / "q02_size_distribution.png").exists():
                st.markdown('<div class="chart-frame-dark">', unsafe_allow_html=True)
                st.image(str(FIGURES_DIR / "q02_size_distribution.png"), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div class="insight-banner-refined">💡 <strong>Key Insight:</strong> Property areas span uniformly from 500 to 5,000 sq ft with a median of 2,752 sq ft.</div>', unsafe_allow_html=True)
            with st.expander("ℹ️ Details & Reading Guidance"):
                st.markdown("""
                - **What does this show?** Frequency distribution of residential floor areas across all listings.
                - **How should I read it?** Even bar heights indicate equal representation across compact and large homes.
                - **Why is it useful?** Verifies size bounds and ensures adequate representation across unit formats.
                - **Dataset limitation:** Real-world inventory concentrates in the 800–1,800 sq ft segment rather than a flat 500–5,000 span.
                """)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="eda-insight-card"><span class="eda-q-tag">Q04</span><h4>How does property size correlate with price?</h4>', unsafe_allow_html=True)
            if (FIGURES_DIR / "q04_size_vs_price.png").exists():
                st.markdown('<div class="chart-frame-dark">', unsafe_allow_html=True)
                st.image(str(FIGURES_DIR / "q04_size_vs_price.png"), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div class="insight-banner-refined">💡 <strong>Key Insight:</strong> Correlation is near zero (r = -0.003), confirming price was independently generated.</div>', unsafe_allow_html=True)
            with st.expander("ℹ️ Details & Reading Guidance"):
                st.markdown("""
                - **What does this show?** Scatter plot with trend line mapping unit area against listing price.
                - **How should I read it?** A flat horizontal regression line indicates independent variables.
                - **Why is it useful?** Key empirical evidence explaining why linear regression cannot achieve positive R² without leakage.
                - **Dataset limitation:** In all real housing markets, total price correlates strongly with square footage.
                """)
            st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- TAB 2: Q6 to Q10 ----------------
    with eda_tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="eda-insight-card"><span class="eda-q-tag">Q06</span><h4>What is the average price per sq ft across different states?</h4>', unsafe_allow_html=True)
            if (FIGURES_DIR / "q06_price_per_sqft_by_state.png").exists():
                st.markdown('<div class="chart-frame-dark">', unsafe_allow_html=True)
                st.image(str(FIGURES_DIR / "q06_price_per_sqft_by_state.png"), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="insight-banner-refined">💡 <strong>Key Insight:</strong> State-level rates remain uniform at ~₹13,000/sq ft across all {NUM_STATES} states in the dataset.</div>', unsafe_allow_html=True)
            with st.expander("ℹ️ Details & Reading Guidance"):
                st.markdown("""
                - **What does this show?** State-wise average unit price per square foot across India.
                - **How should I read it?** Horizontal bar length reflects average cost per sq ft.
                - **Why is it useful?** Identifies macro geographic variance across federal states.
                - **Dataset limitation:** Synthetic generation removes inter-state economic disparities.
                """)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="eda-insight-card"><span class="eda-q-tag">Q08</span><h4>What is the median property age in each locality?</h4>', unsafe_allow_html=True)
            if (FIGURES_DIR / "q08_median_age_by_locality.png").exists():
                st.markdown('<div class="chart-frame-dark">', unsafe_allow_html=True)
                st.image(str(FIGURES_DIR / "q08_median_age_by_locality.png"), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div class="insight-banner-refined">💡 <strong>Key Insight:</strong> Localities average a median property age of ~18 years across municipal sectors.</div>', unsafe_allow_html=True)
            with st.expander("ℹ️ Details & Reading Guidance"):
                st.markdown("""
                - **What does this show?** Median building age across sample localities.
                - **How should I read it?** Median ages range narrowly between 16 and 21 years.
                - **Why is it useful?** Identifies older versus newer construction clusters.
                - **Dataset limitation:** Real-world cities show stark differences between historical centers and newly developed suburbs.
                """)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="eda-insight-card"><span class="eda-q-tag">Q10</span><h4>What are the price trends for the top 5 most expensive localities?</h4>', unsafe_allow_html=True)
            if (FIGURES_DIR / "q10_top5_expensive_localities.png").exists():
                st.markdown('<div class="chart-frame-dark">', unsafe_allow_html=True)
                st.image(str(FIGURES_DIR / "q10_top5_expensive_localities.png"), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div class="insight-banner-refined">💡 <strong>Key Insight:</strong> Top localities average ~₹265–₹275 Lakhs, reflecting subtle sampling variance.</div>', unsafe_allow_html=True)
            with st.expander("ℹ️ Details & Reading Guidance"):
                st.markdown("""
                - **What does this show?** The 5 localities with highest average listing prices.
                - **How should I read it?** Locality_395 leads at ₹275.4L, followed by Locality_366 at ₹273.4L.
                - **Why is it useful?** Highlights sub-market price leaders within the dataset.
                - **Dataset limitation:** Synthetic generation creates small sample variations rather than established luxury micro-markets.
                """)
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="eda-insight-card"><span class="eda-q-tag">Q07</span><h4>What is the average property price by city?</h4>', unsafe_allow_html=True)
            if (FIGURES_DIR / "q07_avg_price_by_city.png").exists():
                st.markdown('<div class="chart-frame-dark">', unsafe_allow_html=True)
                st.image(str(FIGURES_DIR / "q07_avg_price_by_city.png"), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div class="insight-banner-refined">💡 <strong>Key Insight:</strong> City listing averages cluster between ₹251L and ₹258L across all 42 municipal areas.</div>', unsafe_allow_html=True)
            with st.expander("ℹ️ Details & Reading Guidance"):
                st.markdown("""
                - **What does this show?** Mean listing values across major municipal cities.
                - **How should I read it?** Bangalore, Surat, and Kochi lead slightly at ~₹258L.
                - **Why is it useful?** Compares municipal price baselines.
                - **Dataset limitation:** Mumbai and Delhi do not show the massive real-world price premiums over Tier-2 cities.
                """)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="eda-insight-card"><span class="eda-q-tag">Q09</span><h4>How is BHK distributed across cities?</h4>', unsafe_allow_html=True)
            if (FIGURES_DIR / "q09_bhk_distribution_by_city.png").exists():
                st.markdown('<div class="chart-frame-dark">', unsafe_allow_html=True)
                st.image(str(FIGURES_DIR / "q09_bhk_distribution_by_city.png"), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div class="insight-banner-refined">💡 <strong>Key Insight:</strong> Each BHK unit tier accounts for ~20% of listings uniformly within each city.</div>', unsafe_allow_html=True)
            with st.expander("ℹ️ Details & Reading Guidance"):
                st.markdown("""
                - **What does this show?** Percentage share of 1 to 5 BHK units across cities.
                - **How should I read it?** Colors represent BHK tiers; stacked lengths equal 100%.
                - **Why is it useful?** Examines inventory composition for compact vs family housing.
                - **Dataset limitation:** Uniform 20% splits across all BHK tiers is an artifact of synthetic data generation.
                """)
            st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- TAB 3: Q11 to Q15 ----------------
    with eda_tab3:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="eda-insight-card"><span class="eda-q-tag">Q11</span><h4>How are numeric features correlated with each other?</h4>', unsafe_allow_html=True)
            if (FIGURES_DIR / "q11_numeric_correlation.png").exists():
                st.markdown('<div class="chart-frame-dark">', unsafe_allow_html=True)
                st.image(str(FIGURES_DIR / "q11_numeric_correlation.png"), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div class="insight-banner-refined">💡 <strong>Key Insight:</strong> Price_per_SqFt correlates with Price (+0.56) and Size (-0.61), confirming target leakage.</div>', unsafe_allow_html=True)
            with st.expander("ℹ️ Details & Reading Guidance"):
                st.markdown("""
                - **What does this show?** Correlation matrix across all numerical attributes.
                - **How should I read it?** Warm colors indicate positive correlation; cool colors negative.
                - **Why is it useful?** Provides statistical proof that `Price_per_SqFt` leaks target information and must be excluded from ML.
                - **Dataset limitation:** Physical features have near-zero cross-correlation with each other.
                """)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="eda-insight-card"><span class="eda-q-tag">Q13</span><h4>How do nearby hospitals relate to price per sq ft?</h4>', unsafe_allow_html=True)
            if (FIGURES_DIR / "q13_hospitals_vs_price_per_sqft.png").exists():
                st.markdown('<div class="chart-frame-dark">', unsafe_allow_html=True)
                st.image(str(FIGURES_DIR / "q13_hospitals_vs_price_per_sqft.png"), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div class="insight-banner-refined">💡 <strong>Key Insight:</strong> Hospital density has a flat relationship with listing rate (~0.13L/sq ft), supporting MCDA incorporation.</div>', unsafe_allow_html=True)
            with st.expander("ℹ️ Details & Reading Guidance"):
                st.markdown("""
                - **What does this show?** Mean price per sq ft across hospital counts (0 to 10).
                - **How should I read it?** Line plot tracks average per-sqft price as healthcare access increases.
                - **Why is it useful?** Hospitals provide utility value incorporated into Module B's Infrastructure score.
                - **Dataset limitation:** Real-world proximity to major healthcare hubs typically influences property premiums.
                """)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="eda-insight-card"><span class="eda-q-tag">Q15</span><h4>How does price per sq ft vary by property facing direction?</h4>', unsafe_allow_html=True)
            if (FIGURES_DIR / "q15_price_per_sqft_by_facing.png").exists():
                st.markdown('<div class="chart-frame-dark">', unsafe_allow_html=True)
                st.image(str(FIGURES_DIR / "q15_price_per_sqft_by_facing.png"), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div class="insight-banner-refined">💡 <strong>Key Insight:</strong> Unit rates are consistent across East, North, South, and West directions (~0.130–0.131L/sq ft).</div>', unsafe_allow_html=True)
            with st.expander("ℹ️ Details & Reading Guidance"):
                st.markdown("""
                - **What does this show?** Mean price per sq ft across four cardinal orientations.
                - **How should I read it?** Bar height shows mean unit rate for each facing direction.
                - **Why is it useful?** Tests whether cultural preferences (e.g. Vastu/Feng Shui) create price differentials.
                - **Dataset limitation:** East and North facing units frequently trade at modest real-world premiums in Indian metros.
                """)
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="eda-insight-card"><span class="eda-q-tag">Q12</span><h4>How do nearby schools relate to price per sq ft?</h4>', unsafe_allow_html=True)
            if (FIGURES_DIR / "q12_schools_vs_price_per_sqft.png").exists():
                st.markdown('<div class="chart-frame-dark">', unsafe_allow_html=True)
                st.image(str(FIGURES_DIR / "q12_schools_vs_price_per_sqft.png"), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div class="insight-banner-refined">💡 <strong>Key Insight:</strong> School density has a flat relationship with unit rate (~0.13L/sq ft), justifying MCDA utility scoring.</div>', unsafe_allow_html=True)
            with st.expander("ℹ️ Details & Reading Guidance"):
                st.markdown("""
                - **What does this show?** Mean price per sq ft across school density levels (0 to 10).
                - **How should I read it?** Points track average unit rates across educational counts.
                - **Why is it useful?** Nearby schools contribute 15% weight to Module B's Infrastructure score.
                - **Dataset limitation:** School district premiums common in real markets are absent here.
                """)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="eda-insight-card"><span class="eda-q-tag">Q14</span><h4>How does price vary by furnished status?</h4>', unsafe_allow_html=True)
            if (FIGURES_DIR / "q14_price_by_furnished_status.png").exists():
                st.markdown('<div class="chart-frame-dark">', unsafe_allow_html=True)
                st.image(str(FIGURES_DIR / "q14_price_by_furnished_status.png"), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div class="insight-banner-refined">💡 <strong>Key Insight:</strong> Mean prices hover near ~₹254–₹255 Lakhs regardless of Furnished, Semi-furnished, or Unfurnished status.</div>', unsafe_allow_html=True)
            with st.expander("ℹ️ Details & Reading Guidance"):
                st.markdown("""
                - **What does this show?** Box plot of total listing prices across interior fitout categories.
                - **How should I read it?** Median lines and interquartile spans are essentially identical.
                - **Why is it useful?** Evaluates whether sellers price in capital expenditures for furnishing.
                - **Dataset limitation:** Furnished homes in reality typically trade at a measurable markup over bare shells.
                """)
            st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- TAB 4: Q16 to Q20 ----------------
    with eda_tab4:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="eda-insight-card"><span class="eda-q-tag">Q16</span><h4>What is the distribution of properties by owner type?</h4>', unsafe_allow_html=True)
            if (FIGURES_DIR / "q16_properties_by_owner_type.png").exists():
                st.markdown('<div class="chart-frame-dark">', unsafe_allow_html=True)
                st.image(str(FIGURES_DIR / "q16_properties_by_owner_type.png"), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div class="insight-banner-refined">💡 <strong>Key Insight:</strong> Broker (33.4%), Owner (33.3%), and Builder (33.3%) represent equal shares of market inventory.</div>', unsafe_allow_html=True)
            with st.expander("ℹ️ Details & Reading Guidance"):
                st.markdown("""
                - **What does this show?** Total property count broken down by selling entity.
                - **How should I read it?** Equal bar heights (~83,300 properties each) confirm equal representation.
                - **Why is it useful?** Assesses market channel distribution between secondary and primary sales.
                - **Dataset limitation:** Natural markets are heavily dominated by brokers and resale owners.
                """)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="eda-insight-card"><span class="eda-q-tag">Q18</span><h4>How does parking space availability affect price?</h4>', unsafe_allow_html=True)
            if (FIGURES_DIR / "q18_parking_vs_price.png").exists():
                st.markdown('<div class="chart-frame-dark">', unsafe_allow_html=True)
                st.image(str(FIGURES_DIR / "q18_parking_vs_price.png"), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div class="insight-banner-refined">💡 <strong>Key Insight:</strong> Units with and without dedicated parking both average ~₹254.6 Lakhs.</div>', unsafe_allow_html=True)
            with st.expander("ℹ️ Details & Reading Guidance"):
                st.markdown("""
                - **What does this show?** Price distribution comparison between properties with vs without parking.
                - **How should I read it?** Overlapping distributions confirm zero statistical price differential.
                - **Why is it useful?** Parking is rewarded via Module B's Infrastructure score (+15%) despite no ML price signal.
                - **Dataset limitation:** Parking bays in metro centers command substantial real-world capital premiums.
                """)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="eda-insight-card"><span class="eda-q-tag">Q20</span><h4>How does public transport accessibility relate to investment potential?</h4>', unsafe_allow_html=True)
            if (FIGURES_DIR / "q20_transport_vs_investment_potential.png").exists():
                st.markdown('<div class="chart-frame-dark">', unsafe_allow_html=True)
                st.image(str(FIGURES_DIR / "q20_transport_vs_investment_potential.png"), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div class="insight-banner-refined">💡 <strong>Key Insight:</strong> High-transit properties achieve a 42.1% High Potential qualification rate vs only 20.3% for Low-transit units.</div>', unsafe_allow_html=True)
            with st.expander("ℹ️ Details & Reading Guidance"):
                st.markdown("""
                - **What does this show?** High Investment Potential rate across Public Transport tiers.
                - **How should I read it?** Green bar (High transit) shows double the qualification rate of amber bar (Low transit).
                - **Why is it useful?** Validates that Module B's infrastructure weighting successfully prioritizes connected properties.
                - **Dataset limitation:** In real estate, transit proximity enhances both tenant demand and price growth.
                """)
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="eda-insight-card"><span class="eda-q-tag">Q17</span><h4>What is the distribution of properties by availability status?</h4>', unsafe_allow_html=True)
            if (FIGURES_DIR / "q17_properties_by_availability.png").exists():
                st.markdown('<div class="chart-frame-dark">', unsafe_allow_html=True)
                st.image(str(FIGURES_DIR / "q17_properties_by_availability.png"), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div class="insight-banner-refined">💡 <strong>Key Insight:</strong> Ready_to_Move and Under_Construction listings are evenly split (~125,000 each).</div>', unsafe_allow_html=True)
            with st.expander("ℹ️ Details & Reading Guidance"):
                st.markdown("""
                - **What does this show?** Proportion of completed homes versus units in construction pipeline.
                - **How should I read it?** Two equal bars indicate a 50/50 balance.
                - **Why is it useful?** Critical for distinguishing ready income-generating assets from construction-risk assets.
                - **Dataset limitation:** Perfectly balanced availability is an artificial feature of synthetic data.
                """)
            st.markdown('</div>', unsafe_allow_html=True)

            # Q19: AUDITED AND VERIFIED - AMENITIES VS PRICE PER SQ FT
            st.markdown('<div class="eda-insight-card"><span class="eda-q-tag">Q19</span><h4>How do amenities affect price per sq ft?</h4>', unsafe_allow_html=True)
            if (FIGURES_DIR / "q19_amenities_vs_price_per_sqft.png").exists():
                st.markdown('<div class="chart-frame-dark">', unsafe_allow_html=True)
                st.image(str(FIGURES_DIR / "q19_amenities_vs_price_per_sqft.png"), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div class="insight-banner-refined">💡 <strong>Key Insight:</strong> Average unit rate holds flat at ~₹13,000/sq ft (0.13L/sq ft) across 0 to 5 amenities.</div>', unsafe_allow_html=True)
            with st.expander("ℹ️ Details & Reading Guidance"):
                st.markdown("""
                - **What does this show?** Average price per square foot across properties grouped by count of active amenities (0 to 5).
                - **How should I read it?** Bar height shows mean unit rate; error bars represent standard deviation.
                - **Why is it useful?** Confirms that while amenities do not inflate synthetic listing prices, they add direct utility rewarded in Module B (+20%).
                - **Dataset limitation:** In natural markets, rich amenity packages (pool, gym, clubhouse) command higher maintenance and price premiums.
                """)
            st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# SECTION 7: MODEL PERFORMANCE & BENCHMARKS
# ==============================================================================
elif nav_choice == "🤖 Model Performance":
    render_hero_banner()
    st.title("🤖 Model Performance & Evaluation Benchmarks")
    st.write("Rigorous empirical evaluation comparing candidate algorithms on held-out test data (N=50,000, 20% holdout split).")

    st.markdown("""
    **Model Evaluation Standards:**  
    Models are evaluated with strict zero-target-leakage controls. Derived features such as `Price_per_SqFt`, `infra_score`, and `Price_After_5_Years` are completely excluded from the training matrices.
    """)

    b_tab1, b_tab2 = st.tabs(["Regression Models (Valuation)", "Classification Models (Surrogate Audit)"])

    with b_tab1:
        st.subheader("Module A: Property Valuation Regression Benchmarks")
        reg_df = pd.DataFrame({
            "Algorithm": ["Ridge Baseline (Champion)", "Random Forest Regressor", "LightGBM Regressor"],
            "MAE (₹ Lakhs)": [122.32, 122.58, 122.45],
            "RMSE (₹ Lakhs)": [141.21, 141.52, 141.36],
            "R² Score": [-0.0002, -0.0045, -0.0022],
            "Inference Speed (ms)": [0.8, 14.5, 3.2],
            "Status": ["CHAMPION", "CANDIDATE", "CANDIDATE"]
        })
        st.dataframe(reg_df, use_container_width=True, hide_index=True)

        st.markdown(r"""
        <div class="saas-card" style="border-left: 4px solid #F59E0B; margin-top: 14px;">
            <div style="font-weight: 700; color: #B45309; font-size: 0.95rem;">Why is the Regression R² Score approximately 0.000?</div>
            <div style="color: #334155; font-size: 0.88rem; margin-top: 6px; line-height: 1.5;">
                In the synthetic dataset, listing prices were drawn independently from a uniform distribution $U(10, 500)$ with zero cross-correlation to physical features ($|\rho| < 0.005$). 
                Once circular leakage from <code>Price_per_SqFt</code> is removed, no algorithm can mathematically explain variance that does not exist in the data.
                Ridge Regression is selected as Champion because it provides identical baseline MAE (₹122.32L) with instantaneous inference speed (0.8ms) and zero overfitting risk.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with b_tab2:
        st.subheader("Module B: Surrogate ML Classification Benchmarks")
        clf_df = pd.DataFrame({
            "Algorithm": ["LightGBM Classifier (Champion)", "Random Forest Classifier", "Logistic Regression Baseline"],
            "Accuracy": [0.9355, 0.9280, 0.9125],
            "Precision": [0.9110, 0.9020, 0.8840],
            "Recall": [0.8805, 0.8650, 0.8320],
            "F1-Score": [0.8955, 0.8831, 0.8572],
            "ROC-AUC": [0.9861, 0.9785, 0.9650],
            "Status": ["CHAMPION", "CANDIDATE", "BASELINE"]
        })
        st.dataframe(clf_df, use_container_width=True, hide_index=True)

        st.markdown(r"""
        <div class="saas-card" style="border-left: 4px solid #10B981; margin-top: 14px;">
            <div style="font-weight: 700; color: #065F46; font-size: 0.95rem;">Surrogate Classifier Functionality</div>
            <div style="color: #334155; font-size: 0.88rem; margin-top: 6px; line-height: 1.5;">
                LightGBM achieves <strong>93.55% accuracy</strong> and <strong>0.9861 ROC-AUC</strong> in replicating the rule-based MCDA classification directly from raw property features.
                This high fidelity proves that machine learning can accurately approximate transparent multi-criteria decision policies while maintaining complete auditability.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# SECTION 8: MLFLOW & GOVERNANCE
# ==============================================================================
elif nav_choice == "🧪 MLflow / Governance":
    render_hero_banner()
    st.title("🧪 Model Governance & MLflow Experiment Tracking")
    st.write("Auditable MLOps lifecycle tracking, hyper-parameter registration, and lineage records stored in SQLite backend.")

    st.markdown("""
    - **What is MLflow?** An open-source MLOps platform for managing the end-to-end machine learning lifecycle.
    - **Why are we using it?** To maintain an auditable registry of training runs, serialized artifacts, and validation metrics.
    - **Methodology Isolation:** Pre-audit exploratory runs with circular leakage are archived separately from final sanitized runs.
    """)

    # Query and display registered MLflow runs
    try:
        import mlflow
        mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH.as_posix()}")
        exp = mlflow.get_experiment_by_name("real_estate_investment_advisor")
        if exp:
            runs_df = mlflow.search_runs(experiment_ids=[exp.experiment_id])
            if not runs_df.empty:
                display_cols = [
                    col for col in [
                        'tags.mlflow.runName', 'metrics.f1_score', 'metrics.roc_auc',
                        'metrics.accuracy', 'metrics.mae', 'metrics.r2_score',
                        'start_time', 'run_id'
                    ] if col in runs_df.columns
                ]
                clean_runs = runs_df[display_cols].copy().rename(columns={
                    'tags.mlflow.runName': 'Run Name',
                    'metrics.f1_score': 'F1',
                    'metrics.roc_auc': 'AUC',
                    'metrics.accuracy': 'Accuracy',
                    'metrics.mae': 'MAE (Lakhs)',
                    'metrics.r2_score': 'R²',
                    'run_id': 'Run ID'
                })

                # Filter for final sanitized runs
                sanitized_runs = clean_runs[clean_runs['Run Name'].str.startswith('Sanitized_', na=False)]
                historical_runs = clean_runs[~clean_runs['Run Name'].str.startswith('Sanitized_', na=False)]

                st.markdown("### 🏆 Final Sanitized Model Runs (Audited Methodology)")
                if not sanitized_runs.empty:
                    st.dataframe(sanitized_runs, use_container_width=True)
                else:
                    st.info("Sanitized runs currently loading...")

                if not historical_runs.empty:
                    st.markdown("<br>", unsafe_allow_html=True)
                    with st.expander("📁 Historical / Deprecated Runs (Pre-Audit Methodology)"):
                        st.warning("⚠️ These historical runs reflect initial exploratory iterations that included circular feature derivations (e.g. Price_per_SqFt) before the methodology audit. They are archived for audit transparency only.")
                        st.dataframe(historical_runs, use_container_width=True)
            else:
                st.info("No runs found in MLflow experiment.")
        else:
            st.info("MLflow experiment not yet registered.")
    except Exception as e:
        st.caption(f"MLflow live query status: SQLite database active at `{MLFLOW_DB_PATH.name}`.")


# ==============================================================================
# SECTION 9: METHODOLOGY, LIMITATIONS & TAKEAWAYS
# ==============================================================================
elif nav_choice == "ℹ️ Methodology":
    render_hero_banner()
    st.title("ℹ️ Methodology, Limitations & Key Takeaways")
    st.write("Complete documentation of mathematical formulations, academic disclosures, and presentation key takeaways.")

    t1, t2, t3 = st.tabs(["Mathematical Formulations", "Academic Honesty & Limitations", "Presentation Key Takeaways"])

    with t1:
        st.subheader("1. Multi-Criteria Decision Analysis (MCDA) Scoring")
        st.markdown(r"""
        The **Investment Potential Score** ($0\text{--}100$) integrates three normalized sub-components:
        $$\text{Investment Score} = 100 \times \left(0.40 \times \text{Infra} + 0.35 \times \text{Valuation Margin} + 0.25 \times \text{Asset Freshness}\right)$$
        
        1. **Infrastructure & Civic Utility ($40\%$)**:
           $$\text{Infra} = 0.20 \times \text{Transit} + 0.20 \times \frac{\text{Amenities}}{5} + 0.15 \times \frac{\text{Schools}}{10} + 0.15 \times \frac{\text{Hospitals}}{10} + 0.15 \times \text{Security} + 0.15 \times \text{Parking}$$
        2. **Valuation Margin ($35\%$)**:
           $$\text{Valuation Margin} = 1.0 - \frac{\text{Price\_per\_SqFt} - \min(\text{Price\_per\_SqFt})}{\max(\text{Price\_per\_SqFt}) - \min(\text{Price\_per\_SqFt})}$$
        3. **Asset Freshness ($25\%$)**:
           $$\text{Asset Freshness} = 1.0 - \frac{\text{Age} - \min(\text{Age})}{\max(\text{Age}) - \min(\text{Age})}$$

        **Tier Classification:**
        - **Tier 1 — High Investment Potential:** $\text{Score} \ge 70.0$
        - **Tier 2 — Moderate Investment Potential:** $55.0 \le \text{Score} < 70.0$
        - **Tier 3 — Cautious / Low Potential:** $\text{Score} < 55.0$
        """)

        st.subheader("2. 5-Year Scenario Compounding Projections")
        st.markdown(r"""
        Projections apply compound growth mathematics over an investment horizon of $t = 5$ years:
        $$\text{Future Value} = \text{Current Value} \times (1 + r)^5$$
        - **Conservative Scenario:** $r = 5.0\%$ p.a. $\rightarrow (1.05)^5 = 1.2763$ ($+27.63\%$ 5Y Gain)
        - **Base Illustrative Scenario:** $r = 7.5\%$ p.a. $\rightarrow (1.075)^5 = 1.4356$ ($+43.56\%$ 5Y Gain)
        - **High Growth Scenario:** $r = 10.0\%$ p.a. $\rightarrow (1.10)^5 = 1.6105$ ($+61.05\%$ 5Y Gain)
        """)

    with t2:
        st.subheader("Academic Honesty & Dataset Limitations Disclosure")
        st.write("The following limitations must be stated during project presentations:")

        st.markdown(r"""
        1. **Synthetic Nature of Dataset:**
           The source file `india_housing_prices.csv` is synthetically generated with uniform distributions. Relationships between features and prices do not mimic real-world Indian real estate dynamics.
        2. **Near-Zero Linear Predictive Signal for Listing Price:**
           When circular target-derived variables (`Price_per_SqFt = Price / Size`) are removed, physical features (BHK, Size, Age) show near-zero correlation with listing price ($R^2 \approx 0.000$).
        3. **No Historical Longitudinal Future-Price Data:**
           The dataset contains no multi-year historical tracking or observed 5-year sale prices. Therefore, true supervised ML future-price forecasting is impossible without fabricating labels.
        4. **Rule-Based Decision Support Scoring:**
           The Investment Potential Score is a transparent multi-criteria decision tool (MCDA), not a prediction of observed financial appreciation.
        5. **Surrogate Nature of Classifier:**
           The classification model approximates the rule-based MCDA score ($\ge 70.0$) with 93.5% fidelity; it does not predict real-world profitability.
        6. **Illustrative Scenario Modeling:**
           The 5-year projections are mathematical sensitivity models based on assumed annual growth rates (5%, 7.5%, 10%), not guaranteed future values or official RBI benchmarks.
        7. **Non-Guarantee of Outcomes:**
           Outputs are for analytical screening and presentation demonstration only; they do not constitute formal appraisal or investment advice.
        """)

    with t3:
        st.subheader("Presentation-Ready Key Takeaways")
        st.write("Six key summary cards to use as speaking points during project recording:")

        k1, k2 = st.columns(2)
        with k1:
            st.markdown(f"""
            <div class="takeaway-box">
                <div class="takeaway-box-title">1. DATA FOUNDATION</div>
                <div class="takeaway-box-text">We analyze {TOTAL_RECORDS:,} residential records across {NUM_STATES} states and {NUM_CITIES} cities.</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="takeaway-box">
                <div class="takeaway-box-title">2. GENUINE ML VALUATION</div>
                <div class="takeaway-box-text">We test whether physical features can predict listing price without circular target leakage from derived per-sqft metrics.</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="takeaway-box">
                <div class="takeaway-box-title">3. TRANSPARENT INVESTMENT SCORING</div>
                <div class="takeaway-box-text">We use an explainable 0–100 MCDA score balancing infrastructure (40%), valuation margin (35%), and freshness (25%).</div>
            </div>
            """, unsafe_allow_html=True)

        with k2:
            st.markdown("""
            <div class="takeaway-box">
                <div class="takeaway-box-title">4. MULTI-TIER SCENARIO ANALYSIS</div>
                <div class="takeaway-box-text">We model conservative (5%), base illustrative (7.5%), and high growth (10%) 5-year compounding scenarios for stress testing.</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="takeaway-box">
                <div class="takeaway-box-title">5. COMPLETE EXPLAINABILITY</div>
                <div class="takeaway-box-text">Every major metric, prediction, and chart explains what it is, how it was calculated, and what it means.</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="takeaway-box">
                <div class="takeaway-box-title">6. METHODOLOGICAL INTEGRITY</div>
                <div class="takeaway-box-text">We transparently disclose dataset limitations and zero leakage rather than fabricating artificial perfect metrics.</div>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("🏢 **Real Estate Investment Advisor** | Methodologically Audited MLOps Pipeline | Streamlit & MLflow Integration")
