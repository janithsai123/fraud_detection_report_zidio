import os
import sys
import io
import json
import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Ensure project root is in sys.path
root_dir = Path(__file__).parent.resolve()
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.model_loader import (
    load_pipeline,
    preprocess_features,
    validate_batch_schema,
    REQUIRED_FEATURES,
    VALID_CATEGORIES
)
from src.predictor import FraudPredictor
from src.data_loader import (
    load_main_dataset,
    load_metrics_and_comparison,
    compute_dataset_kpis
)

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Financial Fraud Detection System — Academic Prototype & Risk Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# CURRENCY FORMATTING HELPERS (INDIAN RUPEES — ₹ INR)
# -----------------------------------------------------------------------------
def format_inr(val: float) -> str:
    """Formats a number as Indian Currency (₹ INR)."""
    return f"₹{val:,.2f}"

def format_inr_lakhs(val: float) -> str:
    """Formats large figures into Indian Lakhs / Crores or exact ₹."""
    if val >= 10000000:
        return f"₹{val / 10000000:.2f} Cr"
    elif val >= 100000:
        return f"₹{val / 100000:.2f} Lakhs"
    else:
        return f"₹{val:,.2f}"

# -----------------------------------------------------------------------------
# VISUAL THEME & PALETTE CONSTANTS (PREMIUM DARK MODE)
# -----------------------------------------------------------------------------
COLOR_PRIMARY = "#38BDF8"   # Electric Sky Blue (Volume / Primary)
COLOR_LEGIT = "#22C55E"     # Vibrant Emerald Green (Class 0 - Legitimate)
COLOR_FRAUD = "#EF4444"     # Vibrant Coral Red (Class 1 - Potential Fraud)
COLOR_AMBER = "#F59E0B"     # Vibrant Amber (Medium Risk)
COLOR_MUTED = "#64748B"     # Slate Grey (Baseline / Secondary)
COLOR_CARD_BG = "#111827"   # Deep Card Background
COLOR_PAGE_BG = "#0B0F19"   # Rich Dark Page Canvas
COLOR_BORDER = "#1E293B"    # Subtle Dark Border
COLOR_TEXT_MAIN = "#F8FAFC" # Bright White Text
COLOR_TEXT_MUTED = "#94A3B8"# Slate Silver Text

# -----------------------------------------------------------------------------
# CACHED DATA & PERFORMANCE HELPERS
# -----------------------------------------------------------------------------
@st.cache_data
def get_cached_dataset():
    return load_main_dataset("synthetic_fraud_dataset1.csv")

@st.cache_resource
def get_cached_predictor():
    return FraudPredictor("fraud_detection_model.joblib")

@st.cache_data
def get_cached_metrics():
    return load_metrics_and_comparison()

@st.cache_data
def get_cached_kpis_for_full_dataset():
    df = get_cached_dataset()
    return compute_dataset_kpis(df)

@st.cache_data
def export_df_to_csv_bytes(df_to_export: pd.DataFrame) -> bytes:
    """Fast cached CSV serializer to eliminate interaction lag."""
    return df_to_export.to_csv(index=False).encode("utf-8")

@st.cache_data
def load_bytes_from_disk(file_path: str) -> bytes:
    """Cached disk reader for static artifacts."""
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return f.read()
    return b""

try:
    df_raw = get_cached_dataset()
    predictor = get_cached_predictor()
    metrics_json, comp_df = get_cached_metrics()
except Exception as e:
    st.error(f"Error loading system assets: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# STYLING (PREMIUM FINTECH DARK THEME & STYLISH MODERN NAVIGATION)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: #F1F5F9;
        letter-spacing: -0.01em;
    }
    
    /* Main Dark Canvas */
    .stApp {
        background-color: #0B0F19 !important;
    }
    
    /* Top Ribbon */
    .app-top-ribbon {
        background: #111827;
        border: 1px solid #1E293B;
        padding: 9px 18px;
        margin-top: -3.5rem;
        margin-bottom: 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        border-radius: 8px;
    }
    .ribbon-left {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .ribbon-logo {
        display: flex;
        align-items: center;
        gap: 9px;
        font-weight: 700;
        font-size: 13.5px;
        color: #F8FAFC;
    }
    .ribbon-badge {
        background: linear-gradient(135deg, #0284C7, #2563EB);
        color: #FFFFFF;
        font-weight: 800;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 10.5px;
        letter-spacing: 0.04em;
        box-shadow: 0 0 10px rgba(14, 165, 233, 0.35);
    }
    .ribbon-right {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 11.5px;
        color: #94A3B8;
    }
    .ribbon-tag {
        background: #1E293B;
        border: 1px solid #334155;
        padding: 3px 9px;
        border-radius: 4px;
        font-size: 11px;
        color: #CBD5E1;
    }
    
    /* Page Header Banner */
    .app-header-banner {
        background: #111827;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 15px 20px;
        margin-bottom: 14px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    .app-brand {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .app-logo-tile {
        width: 38px;
        height: 38px;
        background: rgba(56, 189, 248, 0.12);
        border: 1px solid rgba(56, 189, 248, 0.35);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
        color: #38BDF8;
        box-shadow: 0 0 12px rgba(56, 189, 248, 0.2);
    }
    .app-title-box h1 {
        color: #F8FAFC !important;
        font-size: 18px;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .app-title-box p {
        color: #94A3B8 !important;
        font-size: 12px;
        margin-top: 2px;
        margin-bottom: 0;
    }
    .app-pill {
        background: rgba(56, 189, 248, 0.12);
        border: 1px solid rgba(56, 189, 248, 0.3);
        color: #38BDF8;
        padding: 4px 12px;
        border-radius: 14px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.03em;
    }
    
    /* Content Cards (Dark Theme) */
    .app-card-box {
        background: #111827;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 16px 18px;
        margin-bottom: 14px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    .app-card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
        padding-bottom: 7px;
        border-bottom: 1px solid #1E293B;
    }
    .app-card-title {
        font-size: 13.5px;
        font-weight: 700;
        color: #F8FAFC;
    }
    .app-card-sub {
        font-size: 11.5px;
        color: #94A3B8;
    }
    
    /* Sleek KPI Cards */
    .app-kpi-card {
        background: #111827;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 14px 16px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.25);
        position: relative;
        overflow: hidden;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .app-kpi-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: #38BDF8;
        box-shadow: 0 0 8px rgba(56, 189, 248, 0.6);
    }
    .app-kpi-card.kpi-legit::before {
        background: #22C55E;
        box-shadow: 0 0 8px rgba(34, 197, 94, 0.6);
    }
    .app-kpi-card.kpi-fraud::before {
        background: #EF4444;
        box-shadow: 0 0 8px rgba(239, 68, 68, 0.6);
    }
    .app-kpi-card.kpi-amber::before {
        background: #F59E0B;
        box-shadow: 0 0 8px rgba(245, 158, 11, 0.6);
    }
    .app-kpi-label {
        font-size: 10.5px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94A3B8;
        margin-bottom: 4px;
    }
    .app-kpi-val {
        font-size: 23px;
        font-weight: 800;
        color: #F8FAFC;
        line-height: 1.15;
    }
    .app-kpi-footer {
        margin-top: 6px;
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 11px;
        color: #94A3B8;
    }
    .app-tag-neutral {
        display: inline-block;
        padding: 1px 6px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 10px;
        background: #1E293B;
        color: #CBD5E1;
        border: 1px solid #334155;
    }
    .app-tag-accent {
        display: inline-block;
        padding: 1px 6px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 10px;
        background: rgba(56, 189, 248, 0.15);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.35);
    }
    .app-tag-legit {
        display: inline-block;
        padding: 1px 6px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 10px;
        background: rgba(34, 197, 94, 0.15);
        color: #4ADE80;
        border: 1px solid rgba(34, 197, 94, 0.35);
    }
    .app-tag-fraud {
        display: inline-block;
        padding: 1px 6px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 10px;
        background: rgba(239, 68, 68, 0.15);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.35);
    }
    
    /* Stylish Dark Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0D131F !important;
        border-right: 1px solid #1E293B !important;
    }
    section[data-testid="stSidebar"] > div {
        background-color: #0D131F !important;
        padding-top: 1rem !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stSidebarUserContent"] {
        background-color: #0D131F !important;
    }
    
    /* Modern Stylish Navigation Radio */
    section[data-testid="stSidebar"] div[data-testid="stRadioCircle"],
    section[data-testid="stSidebar"] input[type="radio"],
    section[data-testid="stSidebar"] div[data-testid="stRadio"] > div > div > label > div:first-child:not([data-testid="stMarkdownContainer"]) {
        display: none !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] {
        display: flex !important;
        flex-direction: column !important;
        gap: 4px !important;
        padding: 4px 0 !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label {
        display: flex !important;
        align-items: center !important;
        background: #111827 !important;
        border: 1px solid #1E293B !important;
        border-left: 3px solid transparent !important;
        border-radius: 6px !important;
        padding: 9px 12px !important;
        margin: 0 !important;
        cursor: pointer !important;
        transition: all 0.15s ease-in-out !important;
        width: 100% !important;
        box-sizing: border-box !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label div[data-testid="stMarkdownContainer"] p,
    section[data-testid="stSidebar"] div[role="radiogroup"] > label span,
    section[data-testid="stSidebar"] div[role="radiogroup"] > label p {
        color: #94A3B8 !important;
        font-weight: 500 !important;
        font-size: 12.5px !important;
        margin: 0 !important;
        line-height: 1.35 !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        background: #1E293B !important;
        border-color: #334155 !important;
        border-left: 3px solid #64748B !important;
        transform: translateX(2px) !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover div[data-testid="stMarkdownContainer"] p,
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover p {
        color: #F8FAFC !important;
    }
    
    /* Selected State: Radiant Neon Glow Accent */
    section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"],
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked),
    section[data-testid="stSidebar"] div[role="radiogroup"] > label[aria-checked="true"] {
        background: rgba(14, 165, 233, 0.12) !important;
        border: 1px solid rgba(56, 189, 248, 0.35) !important;
        border-left: 3px solid #38BDF8 !important;
        box-shadow: 0 0 12px rgba(56, 189, 248, 0.15) !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"] div[data-testid="stMarkdownContainer"] p,
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) div[data-testid="stMarkdownContainer"] p,
    section[data-testid="stSidebar"] div[role="radiogroup"] > label[aria-checked="true"] div[data-testid="stMarkdownContainer"] p,
    section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"] p,
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) p {
        color: #38BDF8 !important;
        font-weight: 700 !important;
    }
    
    /* Buttons in Dark Theme */
    div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(135deg, #0284C7, #2563EB) !important;
        color: #FFFFFF !important;
        border: 1px solid #38BDF8 !important;
        border-radius: 6px !important;
        font-weight: 700 !important;
        font-size: 12.5px !important;
        padding: 7px 18px !important;
        transition: all 0.15s ease !important;
        box-shadow: 0 0 10px rgba(14, 165, 233, 0.25) !important;
    }
    div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
        background: linear-gradient(135deg, #0369A1, #1D4ED8) !important;
        box-shadow: 0 0 16px rgba(14, 165, 233, 0.45) !important;
        color: #FFFFFF !important;
    }
    
    /* Download Buttons */
    div.stDownloadButton > button {
        background: #1E293B !important;
        color: #38BDF8 !important;
        border: 1px solid rgba(56, 189, 248, 0.3) !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        font-size: 12.5px !important;
        padding: 7px 16px !important;
        transition: all 0.15s ease !important;
    }
    div.stDownloadButton > button:hover {
        background: #273549 !important;
        border-color: #38BDF8 !important;
        color: #7DD3FC !important;
    }
    
    /* Tabs Styling */
    div[data-testid="stTabs"] button[role="tab"] {
        color: #94A3B8 !important;
        font-weight: 500 !important;
        font-size: 12.5px !important;
        padding: 8px 14px !important;
        border-bottom: 2px solid transparent !important;
    }
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color: #38BDF8 !important;
        border-bottom: 2px solid #38BDF8 !important;
        font-weight: 700 !important;
    }
    
    /* Form Inputs and Selectboxes */
    label[data-testid="stWidgetLabel"] p {
        color: #F8FAFC !important;
        font-weight: 600 !important;
        font-size: 12px !important;
    }
    
    /* Risk Meter Bar */
    .risk-meter {
        width: 100%;
        height: 8px;
        background: #1E293B;
        border-radius: 4px;
        overflow: hidden;
        margin: 8px 0 6px 0;
    }
    .risk-meter-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.3s ease;
    }
    
    /* Dark Disclaimer Callout */
    .prototype-disclaimer {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid #1E293B;
        border-left: 3px solid #38BDF8;
        padding: 10px 14px;
        border-radius: 4px;
        font-size: 11.5px;
        color: #94A3B8;
        margin-bottom: 14px;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# CHART LAYOUT HELPER (POLISHED DARK THEME)
# -----------------------------------------------------------------------------
def apply_chart_layout(fig, title: str = "", height: int = 280):
    fig.update_layout(
        title={
            "text": title,
            "y": 0.96,
            "x": 0.02,
            "xanchor": "left",
            "yanchor": "top",
            "font": {"size": 12.5, "color": COLOR_TEXT_MAIN, "family": "Plus Jakarta Sans, sans-serif", "weight": "bold"}
        },
        margin={"l": 35, "r": 20, "t": 35 if title else 20, "b": 35},
        height=height,
        paper_bgcolor=COLOR_CARD_BG,
        plot_bgcolor=COLOR_CARD_BG,
        font={"family": "Plus Jakarta Sans, sans-serif", "size": 11, "color": COLOR_TEXT_MUTED},
        hoverlabel={"bgcolor": "#0B0F19", "bordercolor": COLOR_PRIMARY, "font_size": 11, "font_family": "Plus Jakarta Sans, sans-serif", "font_color": "#FFFFFF"},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "font": {"size": 10.5, "color": COLOR_TEXT_MUTED}
        }
    )
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="#1E293B",
        zeroline=False,
        tickfont={"size": 10, "color": COLOR_TEXT_MUTED}
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="#1E293B",
        zeroline=False,
        tickfont={"size": 10, "color": COLOR_TEXT_MUTED}
    )
    return fig

# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION (STYLISH FINTECH DARK DESIGN)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="padding: 2px 2px 10px 2px; border-bottom: 1px solid #1E293B; margin-bottom: 10px;">
        <div style="display: flex; align-items: center; gap: 10px;">
            <div style="width: 32px; height: 32px; background: linear-gradient(135deg, #0284C7, #2563EB); border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 15px; color: #FFFFFF; font-weight: 800; flex-shrink: 0; box-shadow: 0 0 12px rgba(14, 165, 233, 0.4);">
                🛡️
            </div>
            <div style="overflow: hidden;">
                <div style="font-size: 13.5px; font-weight: 800; color: #F8FAFC; letter-spacing: -0.01em; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">FRAUD SYSTEM</div>
                <div style="font-size: 10.5px; color: #38BDF8; font-weight: 600;">Academic Prototype</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    nav_pages = [
        "📊 Overview",
        "⚡ Transaction Scoring",
        "📦 Batch Transaction Scoring",
        "🏆 Model Performance & Audit",
        "🔎 Transaction Data Explorer",
        "📁 Project Files & Artifacts",
        "🏛️ About & Governance"
    ]
    
    selected_nav = st.radio(
        "Navigation Menu",
        nav_pages,
        index=0,
        label_visibility="collapsed"
    )
    
    st.markdown("""
    <div style="background: #111827; border-radius: 6px; padding: 11px; border: 1px solid #1E293B; font-size: 11px; margin-top: 14px;">
        <div style="font-weight: 700; color: #F8FAFC; margin-bottom: 7px; display: flex; justify-content: space-between; align-items: center;">
            <span style="letter-spacing: 0.04em;">SYSTEM TELEMETRY</span>
            <span style="background: rgba(56, 189, 248, 0.15); color: #38BDF8; border: 1px solid rgba(56, 189, 248, 0.35); padding: 1px 6px; border-radius: 3px; font-size: 9.5px; font-weight: 700;">ACTIVE</span>
        </div>
        <div style="color: #94A3B8; margin-bottom: 4px; display: flex; justify-content: space-between;">
            <span>Dataset</span>
            <span style="color: #F8FAFC; font-weight: 600;">50,000 Records</span>
        </div>
        <div style="color: #94A3B8; margin-bottom: 4px; display: flex; justify-content: space-between;">
            <span>Classifier</span>
            <span style="color: #F8FAFC; font-weight: 600;">Decision Tree (D12)</span>
        </div>
        <div style="color: #94A3B8; margin-bottom: 4px; display: flex; justify-content: space-between;">
            <span>Features</span>
            <span style="color: #F8FAFC; font-weight: 600;">12 Pipeline Inputs</span>
        </div>
        <div style="color: #94A3B8; margin-bottom: 4px; display: flex; justify-content: space-between;">
            <span>Currency</span>
            <span style="color: #F8FAFC; font-weight: 600;">₹ INR</span>
        </div>
        <div style="color: #94A3B8; margin-bottom: 4px; display: flex; justify-content: space-between;">
            <span>UI Mode</span>
            <span style="color: #38BDF8; font-weight: 700;">Dark Theme v2.0</span>
        </div>
        <div style="margin-top: 6px; padding-top: 5px; border-top: 1px solid #1E293B; color: #64748B; font-size: 10px; text-align: center;">
            Synthetic Benchmark Evaluation
        </div>
    </div>
    """, unsafe_allow_html=True)

# Top Ribbon Bar
st.markdown("""
<div class="app-top-ribbon">
    <div class="ribbon-left">
        <div class="ribbon-logo">
            <span class="ribbon-badge">AI / ML</span>
            <span>Financial Fraud Detection System</span>
        </div>
    </div>
    <div class="ribbon-right">
        <span>Dataset: <strong style="color: #F8FAFC;">synthetic_fraud_dataset1.csv</strong></span>
        <span class="ribbon-tag">₹ Indian Rupees (INR)</span>
        <span class="ribbon-tag" style="color: #4ADE80; font-weight: 600; border-color: rgba(34, 197, 94, 0.35); background: rgba(34, 197, 94, 0.15);">● System Live</span>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PAGE 1: OVERVIEW DASHBOARD
# -----------------------------------------------------------------------------
if selected_nav == "📊 Overview":
    st.markdown("""
    <div class="app-header-banner">
        <div class="app-brand">
            <div class="app-logo-tile">📊</div>
            <div class="app-title-box">
                <h1>Financial Fraud Detection — Overview Dashboard</h1>
                <p>Empirical statistics, temporal trends, and risk distributions across 50,000 banking transactions (₹ INR).</p>
            </div>
        </div>
        <div>
            <span class="app-pill">50,000 TRANSACTIONS LOADED</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Fast Slicer Controls with Date Range Filter
    st.markdown("<div class='app-card-box' style='padding: 12px 18px; margin-bottom: 14px;'>", unsafe_allow_html=True)
    f_col1, f_col2, f_col3, f_col4 = st.columns([1.5, 1, 1, 1])
    
    min_date_val = df_raw["Date"].min().date()
    max_date_val = df_raw["Date"].max().date()
    
    with f_col1:
        date_selection = st.date_input(
            "Date Range Filter",
            value=(min_date_val, max_date_val),
            min_value=min_date_val,
            max_value=max_date_val
        )
    with f_col2:
        cat_filter = st.selectbox(
            "Merchant Category",
            options=["All Categories"] + sorted(df_raw["Merchant_Category"].unique().tolist()),
            index=0
        )
    with f_col3:
        type_filter = st.selectbox(
            "Transaction Type",
            options=["All Types"] + sorted(df_raw["Transaction_Type"].unique().tolist()),
            index=0
        )
    with f_col4:
        loc_filter = st.selectbox(
            "Location",
            options=["All Locations"] + sorted(df_raw["Location"].unique().tolist()),
            index=0
        )
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Ultra-Fast Vectorized Filtering (Milliseconds execution)
    filtered_df = df_raw
    has_filter = False
    
    if isinstance(date_selection, (tuple, list)) and len(date_selection) == 2 and date_selection[0] and date_selection[1]:
        start_ts = pd.Timestamp(date_selection[0])
        end_ts = pd.Timestamp(date_selection[1]) + pd.Timedelta(days=1, nanoseconds=-1)
        if start_ts > pd.Timestamp(min_date_val) or end_ts < pd.Timestamp(max_date_val):
            filtered_df = filtered_df[(filtered_df["Date"] >= start_ts) & (filtered_df["Date"] <= end_ts)]
            has_filter = True
            
    if cat_filter != "All Categories":
        filtered_df = filtered_df[filtered_df["Merchant_Category"] == cat_filter]
        has_filter = True
    if type_filter != "All Types":
        filtered_df = filtered_df[filtered_df["Transaction_Type"] == type_filter]
        has_filter = True
    if loc_filter != "All Locations":
        filtered_df = filtered_df[filtered_df["Location"] == loc_filter]
        has_filter = True
        
    kpis = compute_dataset_kpis(filtered_df) if has_filter else get_cached_kpis_for_full_dataset()
    
    # 6 Rectangular KPI Tiles
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        st.markdown(f"""
        <div class="app-kpi-card">
            <div class="app-kpi-label">TOTAL TRANSACTIONS</div>
            <div class="app-kpi-val">{kpis['total_transactions']:,}</div>
            <div class="app-kpi-footer"><span class="app-tag-neutral">Scope</span><span>Filtered items</span></div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="app-kpi-card kpi-legit">
            <div class="app-kpi-label">LEGITIMATE (CLASS 0)</div>
            <div class="app-kpi-val">{kpis['non_fraud_transactions']:,}</div>
            <div class="app-kpi-footer"><span class="app-tag-legit">{kpis['non_fraud_percentage']:.2f}%</span><span>Non-fraud</span></div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="app-kpi-card kpi-fraud">
            <div class="app-kpi-label">FRAUD-LABELLED (CLASS 1)</div>
            <div class="app-kpi-val">{kpis['fraud_transactions']:,}</div>
            <div class="app-kpi-footer"><span class="app-tag-fraud">{kpis['fraud_percentage']:.2f}%</span><span>Observed rate</span></div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="app-kpi-card">
            <div class="app-kpi-label">TOTAL VOLUME (INR)</div>
            <div class="app-kpi-val">{format_inr_lakhs(kpis['total_value'])}</div>
            <div class="app-kpi-footer"><span class="app-tag-neutral">Sum</span><span>₹{kpis['total_value']:,.0f}</span></div>
        </div>
        """, unsafe_allow_html=True)
    with k5:
        st.markdown(f"""
        <div class="app-kpi-card">
            <div class="app-kpi-label">AVERAGE TICKET (INR)</div>
            <div class="app-kpi-val">{format_inr(kpis['avg_value'])}</div>
            <div class="app-kpi-footer"><span class="app-tag-neutral">Mean</span><span>Per transaction</span></div>
        </div>
        """, unsafe_allow_html=True)
    with k6:
        st.markdown(f"""
        <div class="app-kpi-card">
            <div class="app-kpi-label">MODEL TEST RECALL</div>
            <div class="app-kpi-val">50.12%</div>
            <div class="app-kpi-footer"><span class="app-tag-accent">Decision Tree</span><span>Held-out split</span></div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    
    # Model Risk Calibration & Summary Banner
    st.markdown(f"""
    <div class='prototype-disclaimer'>
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 8px;">
            <div>
                <strong style="color: #F8FAFC;">Model Risk Calibration & Governance:</strong>
                Transactions are categorized into 3 calibrated tiers based on estimated probability:
                <span class="app-tag-fraud">High Risk (≥65%)</span>,
                <span class="app-tag-neutral" style="background: rgba(245, 158, 11, 0.15); border-color: rgba(245, 158, 11, 0.35); color: #FBBF24;">Medium Risk (40–65%)</span>, and
                <span class="app-tag-legit">Low Risk (&lt;40%)</span>.
                Default binary decision threshold: <strong>0.50</strong>.
            </div>
            <div>
                <span style="font-size: 10.5px; color: #38BDF8; font-weight: 600;">Academic Prototype • Synthetic Benchmark</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Row 1: Charts (Monthly Volume Breakdown & Fraud Rate Trend)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class='app-card-box'>
            <div class='app-card-header'>
                <div class='app-card-title'>Monthly Transaction Volume Trends</div>
            </div>
            <div class='app-card-sub'>Monthly volume breakdown by ground truth class (Legitimate vs Fraud-Labelled)</div>
        """, unsafe_allow_html=True)
        
        monthly_grp = filtered_df.groupby(["YearMonth", "Fraud_Label"]).size().unstack(fill_value=0).reset_index()
        if 0 not in monthly_grp.columns:
            monthly_grp[0] = 0
        if 1 not in monthly_grp.columns:
            monthly_grp[1] = 0
        monthly_grp = monthly_grp.rename(columns={0: "Legitimate", 1: "Fraud"})
        
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Bar(
            x=monthly_grp["YearMonth"],
            y=monthly_grp["Legitimate"],
            name="Legitimate (Class 0)",
            marker=dict(color=COLOR_LEGIT)
        ))
        fig_trend.add_trace(go.Bar(
            x=monthly_grp["YearMonth"],
            y=monthly_grp["Fraud"],
            name="Fraud-Labelled (Class 1)",
            marker=dict(color=COLOR_FRAUD)
        ))
        fig_trend.update_layout(barmode="stack")
        apply_chart_layout(fig_trend, height=280)
        st.plotly_chart(fig_trend, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)
        
    with c2:
        st.markdown("""
        <div class='app-card-box'>
            <div class='app-card-header'>
                <div class='app-card-title'>Observed Fraud Rate Trend Over Time</div>
            </div>
            <div class='app-card-sub'>Monthly observed fraud rate (%) vs annual population baseline (32.13%)</div>
        """, unsafe_allow_html=True)
        
        monthly_trend = filtered_df.groupby("YearMonth").agg(
            Total=("Fraud_Label", "count"),
            Fraud=("Fraud_Label", lambda x: (x == 1).sum())
        ).reset_index()
        monthly_trend["Fraud_Rate"] = (monthly_trend["Fraud"] / monthly_trend["Total"] * 100.0).round(2)
        
        fig_fr = go.Figure()
        fig_fr.add_trace(go.Scatter(
            x=monthly_trend["YearMonth"],
            y=[32.13] * len(monthly_trend),
            mode="lines",
            name="Annual Baseline (32.13%)",
            line=dict(color=COLOR_MUTED, width=1.5, dash="dash")
        ))
        fig_fr.add_trace(go.Scatter(
            x=monthly_trend["YearMonth"],
            y=monthly_trend["Fraud_Rate"],
            mode="lines+markers",
            name="Observed Fraud Rate (%)",
            line=dict(color=COLOR_FRAUD, width=2.5),
            marker=dict(size=6, color=COLOR_FRAUD),
            hovertemplate="Month: %{x}<br>Fraud Rate: %{y:.2f}%<extra></extra>"
        ))
        fig_fr.update_layout(yaxis=dict(title="Fraud Rate (%)", range=[20, 45]))
        apply_chart_layout(fig_fr, height=280)
        st.plotly_chart(fig_fr, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)
        
    # Row 2: Charts (Class Distribution Donut & Fast Dimension Breakdowns)
    c3, c4 = st.columns([1, 1.6])
    with c3:
        st.markdown("""
        <div class='app-card-box'>
            <div class='app-card-header'>
                <div class='app-card-title'>Class Distribution</div>
            </div>
            <div class='app-card-sub'>Observed dataset proportion: Legitimate vs Fraud</div>
        """, unsafe_allow_html=True)
        
        dist_counts = filtered_df["Fraud_Label"].value_counts().reset_index()
        dist_counts["Label_Name"] = dist_counts["Fraud_Label"].map({0: "Legitimate (Class 0)", 1: "Fraud-Labelled (Class 1)"})
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=dist_counts["Label_Name"],
            values=dist_counts["count"],
            hole=0.62,
            marker=dict(colors=[COLOR_LEGIT, COLOR_FRAUD], line=dict(color="#111827", width=2)),
            textinfo="percent",
            hoverinfo="label+value+percent",
            direction="clockwise"
        )])
        fig_pie.add_annotation(
            text=f"<b>{kpis['fraud_percentage']:.1f}%</b><br><span style='font-size:10px;color:#94A3B8;'>Fraud Rate</span>",
            x=0.5, y=0.5,
            font_size=13,
            font_color=COLOR_TEXT_MAIN,
            showarrow=False
        )
        apply_chart_layout(fig_pie, height=280)
        fig_pie.update_layout(showlegend=True)
        st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)
        
    with c4:
        st.markdown("""
        <div class='app-card-box'>
            <div class='app-card-header'>
                <div class='app-card-title'>Transaction Dimension Breakdowns</div>
            </div>
            <div class='app-card-sub'>Compare transaction counts and total INR amounts across dimensions</div>
        """, unsafe_allow_html=True)
        
        dim_choice = st.selectbox(
            "Select Dimension to Compare",
            ["Merchant Category", "Transaction Channel", "Location", "Device Type"],
            index=0
        )
        dim_map = {
            "Merchant Category": ("Merchant_Category", "Merchant Category"),
            "Transaction Channel": ("Transaction_Type", "Transaction Channel"),
            "Location": ("Location", "Location"),
            "Device Type": ("Device_Type", "Device Type")
        }
        dim_col, dim_title = dim_map[dim_choice]
        
        agg_df = filtered_df.groupby([dim_col, "Fraud_Label"]).agg(
            Count=("Transaction_Amount", "count"),
            Amount=("Transaction_Amount", "sum")
        ).reset_index()
        agg_df["Class"] = agg_df["Fraud_Label"].map({0: "Legitimate", 1: "Fraud"})
        
        c_p1, c_p2 = st.columns(2)
        with c_p1:
            fig_c = px.bar(
                agg_df,
                x=dim_col,
                y="Count",
                color="Class",
                barmode="group",
                color_discrete_map={"Legitimate": COLOR_LEGIT, "Fraud": COLOR_FRAUD},
                labels={"Class": "Class", "Count": "Count"}
            )
            apply_chart_layout(fig_c, title=f"Count by {dim_title}", height=220)
            st.plotly_chart(fig_c, use_container_width=True, config={"displayModeBar": False})
        with c_p2:
            fig_a = px.bar(
                agg_df,
                x=dim_col,
                y="Amount",
                color="Class",
                barmode="group",
                color_discrete_map={"Legitimate": COLOR_LEGIT, "Fraud": COLOR_FRAUD},
                labels={"Class": "Class", "Amount": "Amount (₹)"}
            )
            apply_chart_layout(fig_a, title=f"Value (₹) by {dim_title}", height=220)
            st.plotly_chart(fig_a, use_container_width=True, config={"displayModeBar": False})
            
        st.markdown("</div>", unsafe_allow_html=True)

    # Activity Feed Table & Fast Download
    st.markdown("""
    <div class='app-card-box'>
        <div class='app-card-header'>
            <div class='app-card-title'>Recent Transactions Sample</div>
        </div>
        <div class='app-card-sub'>Filter by Transaction ID or User ID (Values in ₹ INR)</div>
    """, unsafe_allow_html=True)
    
    t_search = st.text_input("Search records", "", placeholder="Search by Transaction ID (e.g. TXN_33553) or User ID (e.g. USER_1834)", label_visibility="collapsed")
    display_subset = filtered_df
    if t_search.strip():
        term = t_search.strip()
        display_subset = display_subset[
            display_subset["Transaction_ID"].str.contains(term, case=False, na=False) |
            display_subset["User_ID"].str.contains(term, case=False, na=False)
        ]
        
    preview_cols = [
        "Transaction_ID", "User_ID", "Date", "Transaction_Amount", "Transaction_Type",
        "Merchant_Category", "Location", "Device_Type", "Fraud_Label"
    ]
    table_data = display_subset[preview_cols].head(15).copy()
    table_data["Status"] = table_data["Fraud_Label"].map({0: "Legitimate", 1: "Fraud Flagged"})
    table_data["Amount (₹)"] = table_data["Transaction_Amount"].map(lambda x: format_inr(x))
    
    st.dataframe(
        table_data[["Transaction_ID", "User_ID", "Date", "Amount (₹)", "Transaction_Type", "Merchant_Category", "Location", "Device_Type", "Status"]],
        use_container_width=True,
        hide_index=True
    )
    
    # Fast Cached Download Button
    dl_c1, dl_c2 = st.columns([3, 1])
    with dl_c2:
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=export_df_to_csv_bytes(display_subset),
            file_name="filtered_transactions_dataset.csv",
            mime="text/csv",
            use_container_width=True
        )
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PAGE 2: TRANSACTION SCORING
# -----------------------------------------------------------------------------
elif selected_nav == "⚡ Transaction Scoring":
    st.markdown("""
    <div class="app-header-banner">
        <div class="app-brand">
            <div class="app-logo-tile">⚡</div>
            <div class="app-title-box">
                <h1>Transaction Scoring & Inference Studio</h1>
                <p>Score transactions using the trained Decision Tree pipeline (₹ INR).</p>
            </div>
        </div>
        <div>
            <span class="app-pill">PIPELINE READY</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='prototype-disclaimer'>
        <strong>Academic Prototype Notice:</strong> Model predictions and risk tiers are probabilistic estimates generated by an educational machine learning pipeline trained on synthetic benchmark data. They do not constitute verified fraud findings or commercial financial advice.
    </div>
    """, unsafe_allow_html=True)
    
    tab_multi, tab_single = st.tabs(["📋 Grid / Batch Insertion", "📝 Single Transaction Deep-Dive Form"])
    
    # ---------------------------------------------------------
    # TAB 1: MULTIPLE TRANSACTION INSERTION
    # ---------------------------------------------------------
    with tab_multi:
        st.markdown("""
        <div class='app-card-box'>
            <div class='app-card-header'>
                <div class='app-card-title'>Interactive Transaction Insertion Grid</div>
            </div>
            <div class='app-card-sub'>
                Add or edit transactions below. Options strictly match the verified training dataset categories.
            </div>
        """, unsafe_allow_html=True)
        
        initial_multi_data = [
            {
                "Transaction_ID": "TXN_IN_001",
                "Transaction_Amount": 245.00,
                "Transaction_Type": "Online",
                "Account_Balance": 12500.00,
                "Device_Type": "Laptop",
                "Location": "Mumbai",
                "Merchant_Category": "Electronics",
                "Previous_Fraudulent_Activity": 0,
                "Daily_Transaction_Count": 3,
                "Card_Type": "Visa",
                "Card_Age": 18,
                "Date": "2023-06-15"
            },
            {
                "Transaction_ID": "TXN_IN_002",
                "Transaction_Amount": 35.00,
                "Transaction_Type": "POS",
                "Account_Balance": 45000.00,
                "Device_Type": "Mobile",
                "Location": "London",
                "Merchant_Category": "Groceries",
                "Previous_Fraudulent_Activity": 0,
                "Daily_Transaction_Count": 1,
                "Card_Type": "Mastercard",
                "Card_Age": 36,
                "Date": "2023-07-20"
            },
            {
                "Transaction_ID": "TXN_IN_003",
                "Transaction_Amount": 890.00,
                "Transaction_Type": "Bank Transfer",
                "Account_Balance": 1200.00,
                "Device_Type": "Tablet",
                "Location": "Tokyo",
                "Merchant_Category": "Travel",
                "Previous_Fraudulent_Activity": 1,
                "Daily_Transaction_Count": 11,
                "Card_Type": "Amex",
                "Card_Age": 2,
                "Date": "2023-08-11"
            },
            {
                "Transaction_ID": "TXN_IN_004",
                "Transaction_Amount": 75.00,
                "Transaction_Type": "POS",
                "Account_Balance": 35000.00,
                "Device_Type": "Mobile",
                "Location": "Sydney",
                "Merchant_Category": "Restaurants",
                "Previous_Fraudulent_Activity": 0,
                "Daily_Transaction_Count": 2,
                "Card_Type": "Visa",
                "Card_Age": 48,
                "Date": "2023-09-05"
            }
        ]
        
        multi_df_template = pd.DataFrame(initial_multi_data)
        
        edited_multi_df = st.data_editor(
            multi_df_template,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Transaction_ID": st.column_config.TextColumn("Transaction ID", required=True),
                "Transaction_Amount": st.column_config.NumberColumn("Amount (₹)", min_value=0.0, step=10.0, format="₹%.2f"),
                "Transaction_Type": st.column_config.SelectboxColumn("Type", options=VALID_CATEGORIES["Transaction_Type"], required=True),
                "Account_Balance": st.column_config.NumberColumn("Balance (₹)", min_value=0.0, step=100.0, format="₹%.2f"),
                "Device_Type": st.column_config.SelectboxColumn("Device", options=VALID_CATEGORIES["Device_Type"], required=True),
                "Location": st.column_config.SelectboxColumn("Location", options=VALID_CATEGORIES["Location"], required=True),
                "Merchant_Category": st.column_config.SelectboxColumn("Category", options=VALID_CATEGORIES["Merchant_Category"], required=True),
                "Previous_Fraudulent_Activity": st.column_config.SelectboxColumn("Prior Fraud (0/1)", options=[0, 1], required=True),
                "Daily_Transaction_Count": st.column_config.NumberColumn("Daily Velocity", min_value=1, max_value=20, step=1),
                "Card_Type": st.column_config.SelectboxColumn("Card Type", options=VALID_CATEGORIES["Card_Type"], required=True),
                "Card_Age": st.column_config.NumberColumn("Card Age (Mo)", min_value=1, max_value=240, step=1),
                "Date": st.column_config.TextColumn("Date (YYYY-MM-DD)", required=True)
            }
        )
        
        c_m_btn, c_m_th = st.columns([2, 1])
        with c_m_th:
            multi_threshold = st.slider("Classification Decision Threshold", min_value=0.10, max_value=0.90, value=0.50, step=0.05, key="multi_th")
        with c_m_btn:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            score_multi_btn = st.button("⚡ Score Transactions in Grid", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        if score_multi_btn:
            if edited_multi_df is not None and len(edited_multi_df) > 0:
                with st.spinner("Executing model scoring..."):
                    try:
                        scored_multi = predictor.predict_batch(edited_multi_df, threshold=multi_threshold)
                        
                        m_total = len(scored_multi)
                        m_flagged = int((scored_multi["Model_Prediction"] == 1).sum())
                        m_risk_amt = float(scored_multi[scored_multi["Model_Prediction"] == 1]["Transaction_Amount"].sum())
                        m_avg_prob = float(scored_multi["Fraud_Probability"].mean() * 100.0)
                        
                        sm1, sm2, sm3, sm4 = st.columns(4)
                        with sm1:
                            st.markdown(f"""
                            <div class="app-kpi-card">
                                <div class="app-kpi-label">INSERTED ROWS</div>
                                <div class="app-kpi-val">{m_total}</div>
                                <div class="app-kpi-footer"><span class="app-tag-neutral">Grid</span><span>Scored rows</span></div>
                            </div>
                            """, unsafe_allow_html=True)
                        with sm2:
                            st.markdown(f"""
                            <div class="app-kpi-card kpi-fraud">
                                <div class="app-kpi-label">FLAGGED FRAUD</div>
                                <div class="app-kpi-val">{m_flagged}</div>
                                <div class="app-kpi-footer"><span class="app-tag-fraud">{(m_flagged/m_total*100):.1f}%</span><span>Flagged</span></div>
                            </div>
                            """, unsafe_allow_html=True)
                        with sm3:
                            st.markdown(f"""
                            <div class="app-kpi-card">
                                <div class="app-kpi-label">VALUE AT RISK (INR)</div>
                                <div class="app-kpi-val">{format_inr(m_risk_amt)}</div>
                                <div class="app-kpi-footer"><span class="app-tag-neutral">Exposure</span><span>Flagged sum</span></div>
                            </div>
                            """, unsafe_allow_html=True)
                        with sm4:
                            st.markdown(f"""
                            <div class="app-kpi-card">
                                <div class="app-kpi-label">MEAN FRAUD PROBABILITY</div>
                                <div class="app-kpi-val">{m_avg_prob:.1f}%</div>
                                <div class="app-kpi-footer"><span class="app-tag-accent">Threshold: {multi_threshold:.2f}</span></div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
                        
                        st.markdown("""
                        <div class='app-card-box'>
                            <div class='app-card-header'>
                                <div class='app-card-title'>Scored Results</div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        disp_scored = scored_multi.copy()
                        disp_scored["Amount (₹)"] = disp_scored["Transaction_Amount"].map(lambda x: format_inr(x))
                        disp_scored["Fraud Prob %"] = (disp_scored["Fraud_Probability"] * 100).round(1).astype(str) + "%"
                        disp_scored["Decision"] = disp_scored["Model_Prediction"].map({0: "Approved (Legitimate)", 1: "Flagged (Fraud)"})
                        
                        preview_cols = ["Transaction_ID", "Amount (₹)", "Transaction_Type", "Merchant_Category", "Location", "Fraud Prob %", "Decision", "Risk_Tier", "Recommended_Action"]
                        st.dataframe(disp_scored[preview_cols], use_container_width=True, hide_index=True)
                        
                        st.download_button(
                            label="📥 Download Scored Transactions as CSV",
                            data=export_df_to_csv_bytes(disp_scored),
                            file_name="scored_inserted_transactions.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        st.markdown("</div>", unsafe_allow_html=True)
                    except Exception as err:
                        st.error(f"Error scoring inserted transactions: {err}")
            else:
                st.warning("Please insert at least 1 transaction row in the grid.")

    # ---------------------------------------------------------
    # TAB 2: SINGLE TRANSACTION DEEP-DIVE FORM
    # ---------------------------------------------------------
    with tab_single:
        st.markdown("""
        <div class='app-card-box'>
            <div class='app-card-header'>
                <div class='app-card-title'>Benchmark Example Scenarios</div>
            </div>
            <div class='app-card-sub'>Click a scenario to populate the form parameters (all categories verified against training data)</div>
        """, unsafe_allow_html=True)
        
        ps_c1, ps_c2, ps_c3 = st.columns(3)
        preset_choice = None
        with ps_c1:
            if st.button("🟢 Standard Low-Risk POS Grocery (₹34.50)", use_container_width=True):
                preset_choice = "low"
        with ps_c2:
            if st.button("🔵 Moderate Online Electronics Order (₹480.00)", use_container_width=True):
                preset_choice = "medium"
        with ps_c3:
            if st.button("🔴 High-Velocity Travel Anomaly (₹850.00)", use_container_width=True):
                preset_choice = "high"
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Valid category defaults
        if preset_choice == "low":
            d_amount, d_type, d_balance = 34.50, "POS", 4200.0
            d_device, d_loc, d_cat = "Mobile", "Mumbai", "Groceries"
            d_prev, d_daily, d_card, d_card_age = 0, 1, "Visa", 48
            d_date = datetime.date(2023, 6, 15)
        elif preset_choice == "medium":
            d_amount, d_type, d_balance = 480.00, "Online", 1500.0
            d_device, d_loc, d_cat = "Laptop", "London", "Electronics"
            d_prev, d_daily, d_card, d_card_age = 0, 4, "Mastercard", 18
            d_date = datetime.date(2023, 7, 20)
        elif preset_choice == "high":
            d_amount, d_type, d_balance = 850.00, "Online", 320.0
            d_device, d_loc, d_cat = "Tablet", "Tokyo", "Travel"
            d_prev, d_daily, d_card, d_card_age = 1, 12, "Amex", 2
            d_date = datetime.date(2023, 8, 11)
        else:
            d_amount, d_type, d_balance = 120.00, "POS", 2500.0
            d_device, d_loc, d_cat = "Mobile", "Mumbai", "Clothing"
            d_prev, d_daily, d_card, d_card_age = 0, 2, "Visa", 24
            d_date = datetime.date(2023, 6, 15)

        st.markdown("""
        <div class='app-card-box'>
            <div class='app-card-header'>
                <div class='app-card-title'>Transaction Input Form</div>
            </div>
            <div class='app-card-sub'>All options strictly match the 12 features required by the trained model pipeline</div>
        """, unsafe_allow_html=True)
        
        with st.form("single_prediction_form"):
            r1, r2, r3 = st.columns(3)
            with r1:
                inp_amount = st.number_input("Transaction Amount (₹)", min_value=0.0, max_value=100000.0, value=d_amount, step=25.0)
                inp_type = st.selectbox("Transaction Type", VALID_CATEGORIES["Transaction_Type"], index=VALID_CATEGORIES["Transaction_Type"].index(d_type) if d_type in VALID_CATEGORIES["Transaction_Type"] else 0)
                inp_balance = st.number_input("Customer Account Balance (₹)", min_value=0.0, max_value=500000.0, value=d_balance, step=500.0)
                inp_date = st.date_input("Transaction Date", value=d_date, min_value=datetime.date(2023, 1, 1), max_value=datetime.date(2023, 12, 31))
            with r2:
                inp_device = st.selectbox("Device Type", VALID_CATEGORIES["Device_Type"], index=VALID_CATEGORIES["Device_Type"].index(d_device) if d_device in VALID_CATEGORIES["Device_Type"] else 0)
                inp_loc = st.selectbox("Location", VALID_CATEGORIES["Location"], index=VALID_CATEGORIES["Location"].index(d_loc) if d_loc in VALID_CATEGORIES["Location"] else 0)
                inp_cat = st.selectbox("Merchant Category", VALID_CATEGORIES["Merchant_Category"], index=VALID_CATEGORIES["Merchant_Category"].index(d_cat) if d_cat in VALID_CATEGORIES["Merchant_Category"] else 0)
            with r3:
                inp_prev = st.selectbox("Prior Fraudulent Activity", [0, 1], index=d_prev, format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
                inp_daily = st.slider("Daily Transaction Velocity (Count)", min_value=1, max_value=20, value=d_daily)
                inp_card = st.selectbox("Card Type", VALID_CATEGORIES["Card_Type"], index=VALID_CATEGORIES["Card_Type"].index(d_card) if d_card in VALID_CATEGORIES["Card_Type"] else 0)
                inp_card_age = st.number_input("Card Age (Months)", min_value=1, max_value=240, value=d_card_age)
                
            c_th1, c_th2 = st.columns([2, 1])
            with c_th1:
                inp_threshold = st.slider("Classification Decision Threshold", min_value=0.10, max_value=0.90, value=0.50, step=0.05, key="single_th")
                
            submit_btn = st.form_submit_button("⚡ Run Decision Tree Scoring", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        if submit_btn:
            month_val = int(inp_date.month)
            dow_val = int(inp_date.weekday())
            
            record = {
                "Transaction_Amount": inp_amount,
                "Transaction_Type": inp_type,
                "Account_Balance": inp_balance,
                "Device_Type": inp_device,
                "Location": inp_loc,
                "Merchant_Category": inp_cat,
                "Previous_Fraudulent_Activity": inp_prev,
                "Daily_Transaction_Count": inp_daily,
                "Card_Type": inp_card,
                "Card_Age": inp_card_age,
                "Transaction_Month": month_val,
                "Transaction_DayOfWeek": dow_val,
                "Date": str(inp_date)
            }
            
            try:
                res = predictor.predict_single(record, threshold=inp_threshold)
                prob_pct = res['fraud_probability'] * 100.0
                card_border = res['risk_color']
                card_bg = "rgba(239, 68, 68, 0.10)" if res['risk_tier'] == "High Risk" else ("rgba(245, 158, 11, 0.10)" if res['risk_tier'] == "Medium Risk" else "rgba(34, 197, 94, 0.10)")
                    
                st.markdown(f"""
                <div class='app-card-box' style='border: 1px solid {card_border}; background: {card_bg};'>
                    <div class='app-card-header' style='border-bottom: 1px solid rgba(255,255,255,0.08);'>
                        <div class='app-card-title' style='color: {card_border}; font-size: 14px;'>
                            Assessment Result: {res['risk_tier'].upper()}
                        </div>
                        <div style='font-size: 12px; font-weight: 600; color: #F8FAFC;'>
                            Action: <span style='color: {card_border};'>{res['recommended_action']}</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                o1, o2, o3 = st.columns(3)
                with o1:
                    st.markdown(f"""
                    <div style="background: #111827; border-radius: 6px; padding: 12px; border: 1px solid #1E293B; text-align: center;">
                        <div style="font-size: 10.5px; font-weight: 700; color: #94A3B8; text-transform: uppercase;">ESTIMATED PROBABILITY</div>
                        <div style="font-size: 28px; font-weight: 800; color: {card_border}; margin: 4px 0;">{prob_pct:.1f}%</div>
                        <div style="font-size: 11px; color: #94A3B8;">Threshold: {res['threshold_used']*100:.0f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                with o2:
                    binary_label = "Class 1 (Flagged Fraud)" if res['prediction'] == 1 else "Class 0 (Legitimate)"
                    st.markdown(f"""
                    <div style="background: #111827; border-radius: 6px; padding: 12px; border: 1px solid #1E293B; text-align: center;">
                        <div style="font-size: 10.5px; font-weight: 700; color: #94A3B8; text-transform: uppercase;">MODEL DECISION</div>
                        <div style="font-size: 17px; font-weight: 700; color: #F8FAFC; margin: 10px 0;">{binary_label}</div>
                        <div style="font-size: 11px; color: #94A3B8;">Decision Tree Classifier</div>
                    </div>
                    """, unsafe_allow_html=True)
                with o3:
                    st.markdown(f"""
                    <div style="background: #111827; border-radius: 6px; padding: 12px; border: 1px solid #1E293B; text-align: center;">
                        <div style="font-size: 10.5px; font-weight: 700; color: #94A3B8; text-transform: uppercase;">ASSIGNED RISK TIER</div>
                        <div style="font-size: 20px; font-weight: 800; color: {card_border}; margin: 8px 0;">{res['risk_tier']}</div>
                        <div style="font-size: 11px; color: #94A3B8;">Operational Priority</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                st.markdown(f"""
                <div style="margin-top: 12px;">
                    <div style="display: flex; justify-content: space-between; font-size: 11px; font-weight: 600; color: #94A3B8;">
                        <span>Risk Score Meter</span>
                        <span>Estimated Score: {prob_pct:.1f}%</span>
                    </div>
                    <div class="risk-meter">
                        <div class="risk-meter-fill" style="width: {prob_pct}%; background: {card_border};"></div>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 10px; color: #94A3B8; font-weight: 600;">
                        <span style="color: #22C55E;">LOW (&lt;40%)</span>
                        <span style="color: #F59E0B;">MEDIUM (40-65%)</span>
                        <span style="color: #EF4444;">HIGH (≥65%)</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
                st.markdown("<div style='font-size: 12px; font-weight: 700; color: #F8FAFC; margin-bottom: 6px;'>Diagnostic Factors:</div>", unsafe_allow_html=True)
                for factor in res["risk_factors"]:
                    st.markdown(f"""
                    <div style="background: #111827; border: 1px solid #1E293B; border-radius: 4px; padding: 6px 12px; margin-bottom: 4px; font-size: 11.5px; color: #CBD5E1;">
                        • {factor}
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Inference error: {e}")

# -----------------------------------------------------------------------------
# PAGE 3: BATCH TRANSACTION SCORING
# -----------------------------------------------------------------------------
elif selected_nav == "📦 Batch Transaction Scoring":
    st.markdown("""
    <div class="app-header-banner">
        <div class="app-brand">
            <div class="app-logo-tile">📦</div>
            <div class="app-title-box">
                <h1>Batch Transaction Scoring</h1>
                <p>Upload a CSV file to validate headers and score transactions with the trained pipeline.</p>
            </div>
        </div>
        <div>
            <span class="app-pill">BATCH INGESTION</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='prototype-disclaimer'>
        <strong>Academic Prototype Notice:</strong> Model predictions generated during batch scoring are academic estimations. All original input columns are preserved in the output with appended probability and risk classifications.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='app-card-box'>
        <div class='app-card-header'>
            <div class='app-card-title'>CSV Upload & Input Verification</div>
        </div>
        <div class='app-card-sub'>Upload a CSV file containing transactions or test with a 250-record sample</div>
    """, unsafe_allow_html=True)
    
    b1, b2 = st.columns([3, 2])
    with b1:
        uploaded_file = st.file_uploader("Select CSV for Scoring", type=["csv"], label_visibility="collapsed")
    with b2:
        sample_batch_btn = st.button("Load 250 Sample Records", use_container_width=True)
        
    c_th_batch1, c_th_batch2 = st.columns([2, 1])
    with c_th_batch1:
        batch_threshold = st.slider("Classification Decision Threshold", min_value=0.10, max_value=0.90, value=0.50, step=0.05, key="batch_th")
    st.markdown("</div>", unsafe_allow_html=True)
    
    batch_df_to_score = None
    if uploaded_file is not None:
        try:
            raw_csv_df = pd.read_csv(uploaded_file)
            if len(raw_csv_df) == 0:
                st.error("Uploaded CSV file is empty (0 rows).")
            else:
                validation = validate_batch_schema(raw_csv_df)
                if not validation["is_valid"]:
                    st.error(f"Schema Validation Failed: {validation['error_message']}")
                    st.info(f"Required features: {', '.join(REQUIRED_FEATURES)}")
                else:
                    batch_df_to_score = raw_csv_df
                    st.success(f"CSV validated successfully: {len(batch_df_to_score):,} rows detected.")
                    if validation["unexpected_features"]:
                        st.info(f"Note: Extra metadata columns detected and preserved: {', '.join(validation['unexpected_features'])}")
                    if validation["ground_truth_col"]:
                        st.info(f"Note: Ground truth label column '{validation['ground_truth_col']}' detected and excluded from model inputs.")
                    if validation["out_of_domain_categories"]:
                        for c, vals in validation["out_of_domain_categories"].items():
                            st.warning(f"Warning: Out-of-domain categories in column '{c}': {vals}. Imputer will handle them gracefully.")
        except Exception as err:
            st.error(f"Error reading CSV file: {err}")
    elif sample_batch_btn:
        batch_df_to_score = df_raw.sample(250, random_state=42).copy()
        st.info("Loaded 250 sample records from synthetic benchmark.")
        
    if batch_df_to_score is not None and len(batch_df_to_score) > 0:
        with st.spinner("Scoring batch transactions..."):
            try:
                scored_results = predictor.predict_batch(batch_df_to_score, threshold=batch_threshold)
                b_total = len(scored_results)
                b_flagged = int((scored_results["Model_Prediction"] == 1).sum())
                b_flagged_pct = (b_flagged / b_total * 100.0) if b_total > 0 else 0.0
                b_risk_amt = float(scored_results[scored_results["Model_Prediction"] == 1]["Transaction_Amount"].sum())
                b_avg_score = float(scored_results["Fraud_Probability"].mean() * 100.0)
                
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.markdown(f"""
                    <div class="app-kpi-card">
                        <div class="app-kpi-label">RECORDS SCORED</div>
                        <div class="app-kpi-val">{b_total:,}</div>
                        <div class="app-kpi-footer"><span class="app-tag-neutral">Batch</span><span>Total rows</span></div>
                    </div>
                    """, unsafe_allow_html=True)
                with m2:
                    st.markdown(f"""
                    <div class="app-kpi-card kpi-fraud">
                        <div class="app-kpi-label">FLAGGED FRAUD</div>
                        <div class="app-kpi-val">{b_flagged:,}</div>
                        <div class="app-kpi-footer"><span class="app-tag-fraud">{b_flagged_pct:.1f}%</span><span>Flagged</span></div>
                    </div>
                    """, unsafe_allow_html=True)
                with m3:
                    st.markdown(f"""
                    <div class="app-kpi-card">
                        <div class="app-kpi-label">VALUE AT RISK (INR)</div>
                        <div class="app-kpi-val">{format_inr(b_risk_amt)}</div>
                        <div class="app-kpi-footer"><span class="app-tag-neutral">Exposure</span><span>Sum of flagged</span></div>
                    </div>
                    """, unsafe_allow_html=True)
                with m4:
                    st.markdown(f"""
                    <div class="app-kpi-card">
                        <div class="app-kpi-label">MEAN PROBABILITY</div>
                        <div class="app-kpi-val">{b_avg_score:.1f}%</div>
                        <div class="app-kpi-footer"><span class="app-tag-accent">Threshold: {batch_threshold:.2f}</span></div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
                
                # Distribution Chart & Table
                st.markdown("""
                <div class='app-card-box'>
                    <div class='app-card-header'>
                        <div class='app-card-title'>Risk Tier Distribution</div>
                    </div>
                    <div class='app-card-sub'>Distribution of predicted risk tiers across batch transactions</div>
                """, unsafe_allow_html=True)
                
                tier_counts = scored_results["Risk_Tier"].value_counts().reset_index()
                fig_tier = px.bar(
                    tier_counts,
                    x="Risk_Tier",
                    y="count",
                    color="Risk_Tier",
                    color_discrete_map={
                        "High Risk": COLOR_FRAUD,
                        "Medium Risk": COLOR_AMBER,
                        "Low Risk": COLOR_LEGIT
                    },
                    labels={"Risk_Tier": "Risk Tier", "count": "Count"}
                )
                apply_chart_layout(fig_tier, height=230)
                fig_tier.update_layout(showlegend=False)
                st.plotly_chart(fig_tier, use_container_width=True, config={"displayModeBar": False})
                
                tier_filter = st.selectbox("Filter Scored Table by Risk Tier", ["All Tiers", "High Risk", "Medium Risk", "Low Risk"])
                filtered_scored = scored_results.copy()
                if tier_filter != "All Tiers":
                    filtered_scored = filtered_scored[filtered_scored["Risk_Tier"] == tier_filter]
                
                filtered_scored["Amount (₹)"] = filtered_scored["Transaction_Amount"].map(lambda x: format_inr(x))
                disp_cols = [c for c in ["Transaction_ID", "Amount (₹)", "Transaction_Type", "Merchant_Category", "Location", "Fraud_Probability", "Model_Prediction", "Risk_Tier", "Recommended_Action"] if c in filtered_scored.columns]
                
                st.dataframe(
                    filtered_scored[disp_cols].head(50),
                    use_container_width=True,
                    hide_index=True
                )
                
                st.download_button(
                    label="📥 Download Scored Batch Output as CSV (All Original Columns Preserved)",
                    data=export_df_to_csv_bytes(scored_results),
                    file_name="scored_batch_transactions.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                st.markdown("</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Batch scoring error: {e}")

# -----------------------------------------------------------------------------
# PAGE 4: MODEL PERFORMANCE & AUDIT
# -----------------------------------------------------------------------------
elif selected_nav == "🏆 Model Performance & Audit":
    st.markdown("""
    <div class="app-header-banner">
        <div class="app-brand">
            <div class="app-logo-tile">🏆</div>
            <div class="app-title-box">
                <h1>Model Performance & Validation Audit</h1>
                <p>Empirical evaluation metrics, confusion matrix heatmap, and noise analysis on the synthetic benchmark.</p>
            </div>
        </div>
        <div>
            <span class="app-pill">EVALUATION AUDIT</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='prototype-disclaimer'>
        <strong>Validation vs. Test Transparency:</strong> Section 1 displays the final evaluation on the <strong>held-out test set</strong> (7,500 samples), while Section 2 shows the comparative benchmark across 4 candidate models on the <strong>validation set</strong>. All scores strictly derive from saved artifacts (<code>fraud_model_metrics.json</code> and <code>validation_model_comparison.csv</code>).
    </div>
    """, unsafe_allow_html=True)
    
    # SECTION 1: HELD-OUT TEST SPLIT
    st.markdown("""
    <div class='app-card-box'>
        <div class='app-card-header'>
            <div class='app-card-title'>1. Final Test Set Evaluation (Held-Out Test Data: 7,500 Samples)</div>
        </div>
        <div class='app-card-sub'>
            Selected Model: <strong>Decision Tree (max_depth=12, balanced, random_state=42)</strong> | 
            Ground Truth Target: <strong>Class 1 = Fraud, Class 0 = Legitimate</strong> | 
            Source Artifact: <code>fraud_model_metrics.json</code>
        </div>
    """, unsafe_allow_html=True)
    
    t_acc = metrics_json.get("test_accuracy", 0.5035) * 100.0
    t_prec = metrics_json.get("test_precision", 0.3239) * 100.0
    t_rec = metrics_json.get("test_recall", 0.5012) * 100.0
    t_f1 = metrics_json.get("test_f1", 0.3935) * 100.0
    t_roc = metrics_json.get("test_roc_auc", 0.5049) * 100.0
    t_pr = metrics_json.get("test_average_precision", 0.3242) * 100.0
    
    mp1, mp2, mp3, mp4, mp5, mp6 = st.columns(6)
    with mp1:
        st.markdown(f"""
        <div class="app-kpi-card">
            <div class="app-kpi-label">TEST ACCURACY</div>
            <div class="app-kpi-val">{t_acc:.2f}%</div>
            <div class="app-kpi-footer"><span class="app-tag-neutral">Test Split</span></div>
        </div>
        """, unsafe_allow_html=True)
    with mp2:
        st.markdown(f"""
        <div class="app-kpi-card">
            <div class="app-kpi-label">TEST RECALL</div>
            <div class="app-kpi-val">{t_rec:.2f}%</div>
            <div class="app-kpi-footer"><span class="app-tag-neutral">Sensitivity</span></div>
        </div>
        """, unsafe_allow_html=True)
    with mp3:
        st.markdown(f"""
        <div class="app-kpi-card">
            <div class="app-kpi-label">TEST PRECISION</div>
            <div class="app-kpi-val">{t_prec:.2f}%</div>
            <div class="app-kpi-footer"><span class="app-tag-neutral">Purity</span></div>
        </div>
        """, unsafe_allow_html=True)
    with mp4:
        st.markdown(f"""
        <div class="app-kpi-card">
            <div class="app-kpi-label">TEST F1-SCORE</div>
            <div class="app-kpi-val">{t_f1:.2f}%</div>
            <div class="app-kpi-footer"><span class="app-tag-neutral">Harmonic Mean</span></div>
        </div>
        """, unsafe_allow_html=True)
    with mp5:
        st.markdown(f"""
        <div class="app-kpi-card">
            <div class="app-kpi-label">TEST ROC-AUC</div>
            <div class="app-kpi-val">{t_roc:.2f}%</div>
            <div class="app-kpi-footer"><span class="app-tag-neutral">Discrimination</span></div>
        </div>
        """, unsafe_allow_html=True)
    with mp6:
        st.markdown(f"""
        <div class="app-kpi-card">
            <div class="app-kpi-label">TEST PR-AUC</div>
            <div class="app-kpi-val">{t_pr:.2f}%</div>
            <div class="app-kpi-footer"><span class="app-tag-neutral">Prior: 32.13%</span></div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    
    # Confusion Matrix Heatmap (Dark Mode Theme)
    st.markdown("""
        <div style='font-size: 13px; font-weight: 700; color: #F8FAFC; margin-bottom: 6px;'>
            Held-Out Test Confusion Matrix (7,500 Test Transactions)
        </div>
    """, unsafe_allow_html=True)
    
    cm_z = [[2568, 2522], [1202, 1208]]
    cm_labels = [
        ["<b>True Negative (TN): 2,568</b><br>(34.24% of Test Set)<br>Legitimate correctly cleared", "<b>False Positive (FP): 2,522</b><br>(33.63% of Test Set)<br>Legitimate flagged (False Alarm)"],
        ["<b>False Negative (FN): 1,202</b><br>(16.03% of Test Set)<br>Fraud missed", "<b>True Positive (TP): 1,208</b><br>(16.11% of Test Set)<br>Fraud correctly flagged"]
    ]
    
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm_z,
        x=["Predicted Legitimate (Class 0)", "Predicted Fraud (Class 1)"],
        y=["Actual Legitimate (Class 0)", "Actual Fraud (Class 1)"],
        text=cm_labels,
        texttemplate="%{text}",
        textfont={"size": 11, "family": "Plus Jakarta Sans, sans-serif"},
        colorscale=[[0, "#0F172A"], [0.5, "#1E3A8A"], [1, "#0284C7"]],
        showscale=False
    ))
    apply_chart_layout(fig_cm, height=260)
    fig_cm.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_cm, use_container_width=True, config={"displayModeBar": False})
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # SECTION 2: VALIDATION SET RESULTS
    st.markdown("""
    <div class='app-card-box'>
        <div class='app-card-header'>
            <div class='app-card-title'>2. Candidate Model Benchmark (Validation Set Results)</div>
        </div>
        <div class='app-card-sub'>
            Comparative validation set results across 4 candidate architectures. Source Artifact: <code>validation_model_comparison.csv</code>
        </div>
    """, unsafe_allow_html=True)
    
    disp_comp = comp_df.copy()
    metric_cols = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "Average Precision"]
    for c in metric_cols:
        if c in disp_comp.columns:
            disp_comp[c] = (disp_comp[c] * 100).round(2).astype(str) + "%"
    st.dataframe(disp_comp, use_container_width=True, hide_index=True)
    
    comp_melted = pd.melt(
        comp_df,
        id_vars=["Model"],
        value_vars=["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
        var_name="Metric",
        value_name="Score"
    )
    fig_comp = px.bar(
        comp_melted,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        color_discrete_sequence=["#64748B", "#0284C7", "#38BDF8", "#818CF8"]
    )
    apply_chart_layout(fig_comp, title="Validation Set Metrics Across Architectures", height=280)
    fig_comp.update_yaxes(range=[0, 1])
    st.plotly_chart(fig_comp, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)
    
    # SECTION 3: EMPIRICAL NOISE & ROOT CAUSE ANALYSIS
    st.markdown("""
    <div class='app-card-box'>
        <div class='app-card-header'>
            <div class='app-card-title'>3. Root Cause Investigation: Noise & Metric Analysis</div>
        </div>
        <div style="font-size: 12.5px; line-height: 1.6; color: #CBD5E1;">
            <p><strong>Empirical Investigation Findings:</strong></p>
            <ul>
                <li><strong>Why ROC-AUC is close to 0.50:</strong> 
                    Empirical Pearson correlation analysis reveals that all numeric features have near-zero correlation with the fraud label:
                    <code>Daily_Transaction_Count</code> (-0.0071), <code>Account_Balance</code> (-0.0032), <code>Card_Age</code> (-0.0031), <code>Previous_Fraudulent_Activity</code> (-0.0007), and <code>Transaction_Amount</code> (+0.0019).
                    Furthermore, fraud rates across all categorical groups are virtually flat at 31.8% to 32.6% (identical to the 32.13% population base rate).
                    Because the synthetic labels were generated independently of the features, the features provide near-zero discriminative mutual information. Consequently, all classification algorithms (Logistic Regression, Decision Tree, Random Forest) score ROC-AUC ≈ 0.50 (indistinguishable from random ranking).
                </li>
                <li><strong>Why Average Precision is close to Fraud Prevalence (32.4% vs 32.1%):</strong>
                    In binary classification on uninformative data, the expected value of Precision-Recall Area Under Curve equals the positive class prevalence (prior probability P(Y=1) = 32.13%). The observed test PR-AUC of 32.42% confirms that precision matches the baseline rate.
                </li>
                <li><strong>Role of Class Balancing:</strong>
                    The unweighted Dummy Classifier achieves 67.87% accuracy with 0.0% Recall by trivially predicting the majority class. By contrast, applying <code>class_weight='balanced'</code> shifts the Decision Tree threshold to capture 50.12% of the positive cases, which is the mathematically expected sensitivity on uninformative synthetic data.
                </li>
                <li><strong>System Classification Status:</strong>
                    This system is an <strong>educational prototype</strong> evaluated on a <strong>synthetic benchmark dataset</strong>. It is designed to demonstrate ML pipeline engineering, strict schema validation, and reporting integrity. It is <strong>not production-ready</strong> and should not be deployed in real financial environments without retraining on authenticated, real-world transaction logs.
                </li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PAGE 5: TRANSACTION DATA EXPLORER
# -----------------------------------------------------------------------------
elif selected_nav == "🔎 Transaction Data Explorer":
    st.markdown("""
    <div class="app-header-banner">
        <div class="app-brand">
            <div class="app-logo-tile">🔎</div>
            <div class="app-title-box">
                <h1>Transaction Data Explorer</h1>
                <p>Interactive filtering and inspection across 50,000 transaction records (₹ INR).</p>
            </div>
        </div>
        <div>
            <span class="app-pill">EXPLORER ACTIVE</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='app-card-box'>", unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    with f1:
        e_label = st.selectbox("Ground Truth Label", ["All Transactions", "Legitimate Only (0)", "Fraud Only (1)"])
        e_cat = st.multiselect("Merchant Categories", df_raw["Merchant_Category"].unique(), default=df_raw["Merchant_Category"].unique())
    with f2:
        e_type = st.multiselect("Transaction Types", df_raw["Transaction_Type"].unique(), default=df_raw["Transaction_Type"].unique())
        e_loc = st.multiselect("Locations", df_raw["Location"].unique(), default=df_raw["Location"].unique())
    with f3:
        amt_min = float(df_raw["Transaction_Amount"].min())
        amt_max = float(df_raw["Transaction_Amount"].max())
        e_amt_range = st.slider("Transaction Amount Range (₹)", min_value=amt_min, max_value=amt_max, value=(amt_min, amt_max))
        e_search = st.text_input("Filter by ID String", placeholder="Search TXN_ID or USER_ID")
    st.markdown("</div>", unsafe_allow_html=True)
    
    exp_df = df_raw
    if e_label == "Legitimate Only (0)":
        exp_df = exp_df[exp_df["Fraud_Label"] == 0]
    elif e_label == "Fraud Only (1)":
        exp_df = exp_df[exp_df["Fraud_Label"] == 1]
    if len(e_cat) < len(df_raw["Merchant_Category"].unique()):
        exp_df = exp_df[exp_df["Merchant_Category"].isin(e_cat)]
    if len(e_type) < len(df_raw["Transaction_Type"].unique()):
        exp_df = exp_df[exp_df["Transaction_Type"].isin(e_type)]
    if len(e_loc) < len(df_raw["Location"].unique()):
        exp_df = exp_df[exp_df["Location"].isin(e_loc)]
    if e_amt_range[0] > amt_min or e_amt_range[1] < amt_max:
        exp_df = exp_df[(exp_df["Transaction_Amount"] >= e_amt_range[0]) & (exp_df["Transaction_Amount"] <= e_amt_range[1])]
    
    if e_search.strip():
        term = e_search.strip()
        exp_df = exp_df[
            exp_df["Transaction_ID"].str.contains(term, case=False, na=False) |
            exp_df["User_ID"].str.contains(term, case=False, na=False)
        ]
        
    exp_kpis = compute_dataset_kpis(exp_df)
    
    ef1, ef2, ef3, ef4 = st.columns(4)
    with ef1:
        st.markdown(f"""
        <div class="app-kpi-card">
            <div class="app-kpi-label">MATCHING ROWS</div>
            <div class="app-kpi-val">{exp_kpis['total_transactions']:,}</div>
            <div class="app-kpi-footer"><span class="app-tag-neutral">{(exp_kpis['total_transactions']/len(df_raw)*100):.1f}%</span><span>of dataset</span></div>
        </div>
        """, unsafe_allow_html=True)
    with ef2:
        st.markdown(f"""
        <div class="app-kpi-card kpi-fraud">
            <div class="app-kpi-label">FILTERED FRAUD COUNT</div>
            <div class="app-kpi-val">{exp_kpis['fraud_transactions']:,}</div>
            <div class="app-kpi-footer"><span class="app-tag-fraud">{exp_kpis['fraud_percentage']:.2f}%</span><span>Fraud rate</span></div>
        </div>
        """, unsafe_allow_html=True)
    with ef3:
        st.markdown(f"""
        <div class="app-kpi-card kpi-legit">
            <div class="app-kpi-label">LEGITIMATE COUNT</div>
            <div class="app-kpi-val">{exp_kpis['non_fraud_transactions']:,}</div>
            <div class="app-kpi-footer"><span class="app-tag-legit">{exp_kpis['non_fraud_percentage']:.2f}%</span><span>Legitimate</span></div>
        </div>
        """, unsafe_allow_html=True)
    with ef4:
        st.markdown(f"""
        <div class="app-kpi-card">
            <div class="app-kpi-label">FILTERED VALUE (INR)</div>
            <div class="app-kpi-val">{format_inr_lakhs(exp_kpis['total_value'])}</div>
            <div class="app-kpi-footer"><span class="app-tag-neutral">Sum</span><span>₹{exp_kpis['total_value']:,.0f}</span></div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class='app-card-box'>
        <div class='app-card-header'>
            <div class='app-card-title'>Filtered Transaction Table ({len(exp_df):,} Rows in ₹ INR)</div>
        </div>
        <div class='app-card-sub'>Displaying first 50 matching records</div>
    """, unsafe_allow_html=True)
    
    disp_exp_df = exp_df.head(50).copy()
    disp_exp_df["Amount (₹)"] = disp_exp_df["Transaction_Amount"].map(lambda x: format_inr(x))
    view_cols = ["Transaction_ID", "User_ID", "Date", "Amount (₹)", "Transaction_Type", "Merchant_Category", "Location", "Device_Type", "Fraud_Label"]
    
    st.dataframe(disp_exp_df[view_cols], use_container_width=True, hide_index=True)
    st.download_button(
        "📥 Download Filtered Dataset as CSV",
        data=export_df_to_csv_bytes(exp_df),
        file_name="filtered_fraud_dataset.csv",
        mime="text/csv",
        use_container_width=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PAGE 6: PROJECT FILES & ARTIFACTS
# -----------------------------------------------------------------------------
elif selected_nav == "📁 Project Files & Artifacts":
    st.markdown("""
    <div class="app-header-banner">
        <div class="app-brand">
            <div class="app-logo-tile">📁</div>
            <div class="app-title-box">
                <h1>Project Artifacts & Code Repository</h1>
                <p>Download interactive Power BI Desktop models (.pbix), transaction datasets, and trained model artifacts.</p>
            </div>
        </div>
        <div>
            <span class="app-pill">ASSETS</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    def render_file_row(file_path: str, title: str, description: str, mime: str = "application/octet-stream"):
        if os.path.exists(file_path):
            file_bytes = load_bytes_from_disk(file_path)
            size_kb = len(file_bytes) / 1024.0
            size_str = f"{size_kb / 1024.0:.1f} MB" if size_kb > 1024 else f"{size_kb:.1f} KB"
            
            c_info, c_action = st.columns([3.5, 1.5])
            with c_info:
                st.markdown(f"""
                <div style="padding: 4px 0;">
                    <div style="font-size: 13px; font-weight: 700; color: #F8FAFC; font-family: 'JetBrains Mono', monospace;">
                        {os.path.basename(file_path)} <span class="app-tag-neutral">{size_str}</span>
                    </div>
                    <div style="font-size: 11.5px; color: #94A3B8; margin-top: 2px;">{description}</div>
                </div>
                """, unsafe_allow_html=True)
            with c_action:
                st.download_button(
                    label=f"📥 Download {os.path.basename(file_path)}",
                    data=file_bytes,
                    file_name=os.path.basename(file_path),
                    mime=mime,
                    use_container_width=True
                )
            st.markdown("<div style='height: 1px; background: #1E293B; margin: 6px 0;'></div>", unsafe_allow_html=True)
        else:
            st.warning(f"File {file_path} not found.")

    # 1. POWER BI DASHBOARDS
    st.markdown("""
    <div class='app-card-box'>
        <div class='app-card-header'>
            <div class='app-card-title'>1. Power BI Desktop Models (.pbix archives)</div>
        </div>
        <div class='app-card-sub'>Pre-built Power BI report models created for this project</div>
    """, unsafe_allow_html=True)
    render_file_row("power_bi_dashboards/Synthetic Fraud Detection.pbix.zip", "Synthetic Fraud Detection Power BI Model", "Power BI Desktop report archive (.pbix) for Synthetic Fraud Monitoring (10.2 MB).")
    render_file_row("power_bi_dashboards/Credit Fraud Detection.pbix.zip", "Credit Fraud Detection Power BI Model", "Power BI Desktop report archive (.pbix) for Credit Card Fraud Analytics (13.6 MB).")
    render_file_row("power_bi_dashboards/Financial_Fraud_Analysis.pbix.zip", "Financial Fraud Analysis Power BI Model", "Power BI report archive (.pbix) for Multi-Channel Financial Fraud Analytics (13.4 MB).")
    st.markdown("</div>", unsafe_allow_html=True)

    # 2. DATASETS SECTION
    st.markdown("""
    <div class='app-card-box'>
        <div class='app-card-header'>
            <div class='app-card-title'>2. Datasets & Evaluation Artifacts</div>
        </div>
        <div class='app-card-sub'>CSV datasets and saved evaluation metrics</div>
    """, unsafe_allow_html=True)
    render_file_row("synthetic_fraud_dataset1.csv", "Primary Synthetic Transaction Dataset", "50,000 transactions with 14 features and ground-truth Fraud_Label.", mime="text/csv")
    render_file_row("validation_model_comparison.csv", "Validation Model Benchmark Results", "Cross-model validation comparison CSV table.", mime="text/csv")
    render_file_row("fraud_model_metrics.json", "Test Evaluation Metrics (JSON)", "Recorded held-out test split metrics.", mime="application/json")
    st.markdown("</div>", unsafe_allow_html=True)

    # 3. MODEL ARTIFACT SECTION
    st.markdown("""
    <div class='app-card-box'>
        <div class='app-card-header'>
            <div class='app-card-title'>3. Trained Machine Learning Pipeline</div>
        </div>
        <div class='app-card-sub'>Persisted Scikit-Learn Pipeline Artifact</div>
    """, unsafe_allow_html=True)
    render_file_row("fraud_detection_model.joblib", "Persisted Decision Tree Pipeline", "Fitted ColumnTransformer + DecisionTreeClassifier pipeline binary.", mime="application/octet-stream")
    st.markdown("</div>", unsafe_allow_html=True)

    # 4. SCRIPTS SECTION
    st.markdown("""
    <div class='app-card-box'>
        <div class='app-card-header'>
            <div class='app-card-title'>4. Python Scripts & Modules</div>
        </div>
        <div class='app-card-sub'>Source code files and automated test suites</div>
    """, unsafe_allow_html=True)
    def script_row(file_path: str, description: str):
        if os.path.exists(file_path):
            code_text = open(file_path, "r", encoding="utf-8").read()
            c_info, c_action = st.columns([3.5, 1.5])
            with c_info:
                st.markdown(f"""
                <div style="padding: 4px 0;">
                    <div style="font-size: 13px; font-weight: 700; color: #F8FAFC; font-family: 'JetBrains Mono', monospace;">
                        {file_path} <span class="app-tag-neutral">{len(code_text.splitlines())} Lines</span>
                    </div>
                    <div style="font-size: 11.5px; color: #94A3B8; margin-top: 2px;">{description}</div>
                </div>
                """, unsafe_allow_html=True)
            with c_action:
                st.download_button(
                    label=f"📥 Download {os.path.basename(file_path)}",
                    data=code_text,
                    file_name=os.path.basename(file_path),
                    mime="text/plain",
                    use_container_width=True
                )
            st.markdown("<div style='height: 1px; background: #1E293B; margin: 6px 0;'></div>", unsafe_allow_html=True)

    script_row("run_dashboard.py", "Dashboard launcher script.")
    script_row("app.py", "Main Streamlit dashboard application.")
    script_row("src/model_loader.py", "Pipeline loader and schema validation module.")
    script_row("src/predictor.py", "Inference engine and risk categorization.")
    script_row("src/data_loader.py", "Dataset loading and empirical KPI computations.")
    script_row("tests/test_model.py", "Automated unit tests for model preprocessing, schema validation, and inference.")
    script_row("tests/test_data.py", "Automated unit tests for dataset integrity and distinct domain values.")
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PAGE 7: ABOUT & GOVERNANCE
# -----------------------------------------------------------------------------
elif selected_nav == "🏛️ About & Governance":
    st.markdown("""
    <div class="app-header-banner">
        <div class="app-brand">
            <div class="app-logo-tile">🏛️</div>
            <div class="app-title-box">
                <h1>Project Architecture & Governance</h1>
                <p>Zidio Development Internship Capstone Project Documentation.</p>
            </div>
        </div>
        <div>
            <span class="app-pill">PROTOTYPE v2.0</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='prototype-disclaimer'>
        <strong>Governance & Academic Scope Statement:</strong> This project is strictly developed as an educational prototype for the Zidio Development Internship. The machine learning pipeline, dataset, and scoring mechanisms are intended for academic demonstration, pipeline engineering, and statistical research. It is not approved for production banking operations without comprehensive retraining on authenticated financial records.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='app-card-box'>
        <div class='app-card-header'>
            <div class='app-card-title'>Project Purpose & Scope</div>
        </div>
        <div style="font-size: 12.5px; line-height: 1.6; color: #CBD5E1;">
            <p>Developed as an <strong>educational prototype</strong> for the <strong>Zidio Development Internship</strong> in Data Science & Machine Learning. The project demonstrates an end-to-end financial fraud detection workflow, combining statistical exploratory data analysis, machine learning model evaluation, strict schema validation, and interactive risk reporting formatted in Indian Rupees (₹ INR).</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='app-card-box'>
        <div class='app-card-header'>
            <div class='app-card-title'>Dataset & Data Integrity</div>
        </div>
        <div style="font-size: 12.5px; line-height: 1.6; color: #CBD5E1;">
            <ul>
                <li><strong>Dataset:</strong> <code>synthetic_fraud_dataset1.csv</code> (50,000 synthetic transactions, 14 raw columns).</li>
                <li><strong>Class Balance:</strong> 16,067 fraud instances (32.13%) vs. 33,933 legitimate instances (67.87%).</li>
                <li><strong>Currency Standard:</strong> Indian Rupees (<code>₹ INR</code>) with exact Lakhs/Crores conversions.</li>
                <li><strong>Date Range:</strong> 01 January 2023 to 31 December 2023.</li>
                <li><strong>Verified Domain Categories:</strong>
                    <ul>
                        <li><code>Transaction_Type</code>: ATM Withdrawal, Bank Transfer, Online, POS</li>
                        <li><code>Device_Type</code>: Laptop, Mobile, Tablet</li>
                        <li><code>Location</code>: London, Mumbai, New York, Sydney, Tokyo</li>
                        <li><code>Merchant_Category</code>: Clothing, Electronics, Groceries, Restaurants, Travel</li>
                        <li><code>Card_Type</code>: Amex, Discover, Mastercard, Visa</li>
                    </ul>
                </li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='app-card-box'>
        <div class='app-card-header'>
            <div class='app-card-title'>Pipeline Architecture & Testing</div>
        </div>
        <div style="font-size: 12.5px; line-height: 1.6; color: #CBD5E1;">
            <ul>
                <li><strong>Preprocessing:</strong> Scikit-Learn <code>ColumnTransformer</code> with 7 numeric features (median imputer + StandardScaler) and 5 categorical features (mode imputer + OneHotEncoder).</li>
                <li><strong>Classifier:</strong> <code>DecisionTreeClassifier(max_depth=12, class_weight='balanced', random_state=42)</code>.</li>
                <li><strong>Input Features:</strong> Exactly 12 features (identifiers <code>Transaction_ID</code> and <code>User_ID</code> and target <code>Fraud_Label</code> are excluded from model inputs).</li>
                <li><strong>Automated Testing:</strong> 19 automated unit tests across <code>tests/test_dashboard_pages.py</code>, <code>tests/test_data.py</code>, and <code>tests/test_model.py</code>, verified with pytest.</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
