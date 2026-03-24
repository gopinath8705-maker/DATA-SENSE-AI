"""
components/ui_helpers.py
Shared UI utilities: custom CSS, insight cards, KPI rendering.
"""

import streamlit as st
from typing import List, Dict


def apply_custom_css() -> None:
    """Inject custom CSS for a modern, polished look."""
    st.markdown("""
    <style>
        /* Main background and font */
        .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }

        /* Metric cards */
        [data-testid="stMetric"] {
            background: linear-gradient(135deg, #f8faff 0%, #eef2ff 100%);
            border: 1px solid #e0e7ff;
            border-radius: 12px;
            padding: 16px 20px;
            box-shadow: 0 2px 8px rgba(99,102,241,0.07);
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.8rem;
            color: #6366f1;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.6rem;
            font-weight: 700;
            color: #1e1b4b;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
            background: #f1f5f9;
            border-radius: 12px;
            padding: 6px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            padding: 8px 16px;
            font-weight: 500;
        }
        .stTabs [aria-selected="true"] {
            background: #6366f1 !important;
            color: white !important;
        }

        /* Insight cards */
        .insight-card {
            padding: 14px 18px;
            border-radius: 12px;
            margin: 6px 0;
            font-size: 0.92rem;
            border-left: 5px solid;
        }
        .insight-success {
            background: #f0fdf4;
            border-color: #22c55e;
            color: #166534;
        }
        .insight-warning {
            background: #fffbeb;
            border-color: #f59e0b;
            color: #92400e;
        }
        .insight-info {
            background: #eff6ff;
            border-color: #3b82f6;
            color: #1e40af;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1e1b4b 0%, #312e81 100%);
        }
        [data-testid="stSidebar"] * {
            color: #e0e7ff !important;
        }
        [data-testid="stSidebar"] .stButton > button {
            background: #6366f1;
            color: white;
            border: none;
            border-radius: 8px;
        }

        /* Upload area */
        [data-testid="stFileUploader"] {
            border: 2px dashed #a5b4fc;
            border-radius: 12px;
            padding: 8px;
        }

        /* Divider */
        hr {
            border-color: #e0e7ff;
            margin: 12px 0;
        }

        /* Chat messages */
        [data-testid="stChatMessage"] {
            border-radius: 12px;
            margin: 4px 0;
        }

        /* Title */
        h1 {
            background: linear-gradient(135deg, #4f46e5, #7c3aed);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
        }
    </style>
    """, unsafe_allow_html=True)


def render_insight_cards(insights: List[Dict[str, str]]) -> None:
    """
    Render a grid of insight cards.
    Each insight: {'title': str, 'value': str, 'type': 'success'|'warning'|'info'}
    """
    if not insights:
        st.info("No insights detected. Upload a dataset with numeric or categorical columns.")
        return

    # Render in 2-column grid
    cols = st.columns(2)
    for i, insight in enumerate(insights):
        css_class = f"insight-{insight.get('type', 'info')}"
        html = f"""
        <div class="insight-card {css_class}">
            <strong>{insight['title']}</strong><br>
            <span style="font-size:1.1rem; font-weight:700;">{insight['value']}</span>
        </div>
        """
        with cols[i % 2]:
            st.markdown(html, unsafe_allow_html=True)


def render_kpi_cards(data: Dict[str, str]) -> None:
    """
    Render KPI metric cards from a dict of {label: value}.
    """
    cols = st.columns(len(data))
    for i, (label, value) in enumerate(data.items()):
        cols[i].metric(label=label, value=value)
