"""
DataSense AI v2 — Advanced AI-Powered Data Analytics
Features: OCR upload, ML predictions, voice query, PDF reports, anomaly detection,
          quality scoring, NL chart generation, chart AI explanations, recommendations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="DataSense AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Module imports ─────────────────────────────────────────────────────────
from utils.data_loader import load_dataset
from utils.ocr_extractor import smart_extract
from utils.analysis import (
    get_dataset_info, get_descriptive_stats,
    identify_column_types, detect_insights
)
from utils.visualizations import (
    plot_bar_chart, plot_line_chart, plot_pie_chart,
    plot_scatter, plot_correlation_heatmap, plot_histogram
)
from utils.ai_engine import explain_dataset, chat_with_data, generate_ai_insights
from utils.ml_engine import run_predictive_analysis, run_forecast
from utils.quality_anomaly import (
    compute_quality_score, detect_anomalies_zscore,
    detect_anomalies_iqr, detect_anomalies_isolation_forest
)
from utils.recommendations import (
    generate_recommendations, parse_chart_request, explain_chart_with_ai
)
from utils.voice_query import render_voice_input, transcribe_audio
from components.dashboard import render_dashboard
from components.ui_helpers import apply_custom_css, render_insight_cards

# ── Styles ─────────────────────────────────────────────────────────────────
apply_custom_css()

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📊 DataSense AI")
    st.caption("v2.0 · Advanced Analytics Platform")
    st.divider()

    # ── File Upload ───────────────────────────────────────────────────────────
    st.markdown("### 📁 Upload Dataset")
    upload_mode = st.radio(
        "Source type",
        ["CSV / Excel", "Image / PDF (OCR)"],
        horizontal=True,
        label_visibility="collapsed"
    )

    df = None

    if upload_mode == "CSV / Excel":
        uploaded_file = st.file_uploader(
            "Upload file",
            type=["csv", "xlsx", "xls"],
            label_visibility="collapsed"
        )
        if uploaded_file:
            df, err = load_dataset(uploaded_file)
            if err:
                st.error(err)
            else:
                st.success(f"✅ {uploaded_file.name} · {df.shape[0]:,}×{df.shape[1]}")

    else:
        uploaded_ocr = st.file_uploader(
            "Upload image/PDF",
            type=["png", "jpg", "jpeg", "pdf", "webp"],
            label_visibility="collapsed",
            key="ocr_upload"
        )
        if uploaded_ocr:
            api_key_ocr = st.session_state.get("openai_key", "")
            with st.spinner("Extracting table via OCR…"):
                df, msg = smart_extract(uploaded_ocr, api_key_ocr)
            if df is not None:
                st.success(msg)
            else:
                st.error(msg)

    st.divider()

    # ── API Key ───────────────────────────────────────────────────────────────
    st.markdown("### 🔑 OpenAI API Key")
    openai_key = st.text_input(
        "Key",
        type="password",
        value=st.session_state.get("openai_key", ""),
        placeholder="sk-… (optional)",
        label_visibility="collapsed",
        help="Enables: AI explanations, GPT chat, voice transcription, chart explanations"
    )
    if openai_key:
        st.session_state["openai_key"] = openai_key
        st.caption("✅ API key active")
    else:
        st.caption("ℹ️ Rule-based mode (no key needed)")

    st.divider()
    st.caption("Built with Streamlit · Plotly · scikit-learn · LangChain")

# ── Guard: no data ─────────────────────────────────────────────────────────
if df is None or df.empty:
    st.title("📊 DataSense AI")
    st.markdown("*Upload a dataset to begin — CSV, Excel, image, or PDF.*")
    c1, c2, c3, c4 = st.columns(4)
    c1.info("**📤 Upload**\nCSV, Excel, or image/PDF tables via OCR")
    c2.info("**🤖 AI Insights**\nAuto-detect trends, anomalies & patterns")
    c3.info("**🔮 Predict**\nML models with feature importance")
    c4.info("**📄 Export**\nGenerate a full PDF analytics report")
    st.stop()

# ── Shared state ────────────────────────────────────────────────────────────
api_key = st.session_state.get("openai_key", "")
numeric_cols, categorical_cols = identify_column_types(df)
insights = detect_insights(df, numeric_cols, categorical_cols)
quality = compute_quality_score(df)

# ── Page header ─────────────────────────────────────────────────────────────
col_h1, col_h2, col_h3 = st.columns([3, 1, 1])
with col_h1:
    st.title("📊 DataSense AI")
with col_h2:
    q_score = quality["total_score"]
    q_grade = quality["grade"]
    q_color = quality["grade_color"]
    st.markdown(f"""
    <div style='background:{q_color}20;border:1.5px solid {q_color};border-radius:10px;
    padding:8px 14px;text-align:center;margin-top:12px'>
    <span style='font-size:1.4rem;font-weight:800;color:{q_color}'>{q_score}/100</span><br>
    <span style='font-size:0.75rem;color:{q_color}'>Quality: {q_grade} · {quality["grade_label"]}</span>
    </div>""", unsafe_allow_html=True)
with col_h3:
    st.metric("Rows × Cols", f"{df.shape[0]:,} × {df.shape[1]}")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "📋 Preview",
    "🔍 AI Insights",
    "📈 Visualize",
    "🔮 Predict",
    "🚨 Anomalies",
    "💡 Recommend",
    "💬 Chat",
    "📊 Dashboard",
    "📄 Export PDF",
])
(tab_preview, tab_insights, tab_viz, tab_ml,
 tab_anomaly, tab_reco, tab_chat, tab_dash, tab_export) = tabs

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_preview:
    st.subheader("📋 Dataset Preview")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Rows", f"{df.shape[0]:,}")
    m2.metric("Columns", df.shape[1])
    m3.metric("Numeric", len(numeric_cols))
    m4.metric("Categorical", len(categorical_cols))
    m5.metric("Missing %", f"{df.isnull().mean().mean()*100:.1f}%")

    st.dataframe(df.head(10), use_container_width=True)

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("**Descriptive Statistics**")
        st.dataframe(get_descriptive_stats(df), use_container_width=True)
    with col_right:
        st.markdown("**Column Information**")
        info = get_dataset_info(df)
        st.dataframe(info["column_info"], use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — AI INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with tab_insights:
    st.subheader("🔍 AI-Powered Insights")

    # Quality Score breakdown
    with st.expander("📊 Data Quality Score Breakdown", expanded=True):
        qs = quality["dimension_scores"]
        max_s = {"Completeness": 25, "Uniqueness": 20, "Consistency": 20, "Validity": 20, "Size Adequacy": 15}
        cols = st.columns(len(qs))
        for i, (dim, score) in enumerate(qs.items()):
            with cols[i]:
                pct = int(score / max_s.get(dim, 20) * 100)
                st.metric(dim, f"{score}/{max_s.get(dim, 20)}")
                st.progress(pct)

        if quality["issues"]:
            st.markdown("**Issues detected:**")
            for issue in quality["issues"]:
                st.markdown(f"- {issue}")

    st.markdown("#### 🎯 Auto-Detected Insights")
    render_insight_cards(insights)
    st.divider()

    st.markdown("#### 🤖 AI Dataset Explanation")
    if st.button("✨ Generate AI Explanation", use_container_width=True, key="btn_explain"):
        with st.spinner("Analyzing with AI…"):
            result = explain_dataset(df, numeric_cols, categorical_cols, api_key)
        st.markdown(result)

    st.divider()
    st.markdown("#### 💡 Detailed AI Insights")
    if st.button("🔮 Generate Detailed Insights", use_container_width=True, key="btn_detail"):
        with st.spinner("Generating insights…"):
            result = generate_ai_insights(df, numeric_cols, categorical_cols, api_key)
        st.markdown(result)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab_viz:
    st.subheader("📈 Interactive Visualizations")

    with st.expander("✨ Generate Chart with Natural Language", expanded=True):
        st.caption("Example: *'Bar chart of sales by region'* or *'Scatter plot of price vs quantity'*")
        nl_prompt = st.text_input(
            "Describe the chart you want:",
            key="nl_chart_prompt",
            placeholder="e.g. show me a bar chart of revenue by category"
        )
        if st.button("📊 Generate Chart", key="nl_chart_btn") and nl_prompt:
            with st.spinner("Building chart…"):
                fig = parse_chart_request(nl_prompt, df, numeric_cols, categorical_cols, api_key)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                with st.spinner("Getting AI explanation…"):
                    explanation = explain_chart_with_ai("custom", nl_prompt, None, df, api_key)
                st.info(f"🤖 **AI Insight:** {explanation}")
            else:
                st.warning("Could not generate a chart. Try being more specific.")

    st.divider()
    st.markdown("#### 🗂️ Manual Chart Builder")

    viz_type = st.selectbox(
        "Chart type",
        ["Bar Chart", "Line Chart", "Pie Chart", "Scatter Plot", "Histogram", "Correlation Heatmap"],
        key="viz_type_select"
    )

    fig = None
    x_col_used, y_col_used = None, None

    if viz_type == "Bar Chart" and categorical_cols and numeric_cols:
        c1, c2 = st.columns(2)
        x_col_used = c1.selectbox("X (category)", categorical_cols, key="bar_x")
        y_col_used = c2.selectbox("Y (value)", numeric_cols, key="bar_y")
        fig = plot_bar_chart(df, x_col_used, y_col_used)

    elif viz_type == "Line Chart" and numeric_cols:
        c1, c2 = st.columns(2)
        x_col_used = c1.selectbox("X-axis", df.columns.tolist(), key="line_x")
        y_col_used = c2.selectbox("Y-axis", numeric_cols, key="line_y")
        fig = plot_line_chart(df, x_col_used, y_col_used)

    elif viz_type == "Pie Chart" and categorical_cols:
        c1, c2 = st.columns(2)
        x_col_used = c1.selectbox("Label", categorical_cols, key="pie_x")
        y_opt = c2.selectbox("Value", ["(Count)"] + numeric_cols, key="pie_y")
        y_col_used = None if y_opt == "(Count)" else y_opt
        fig = plot_pie_chart(df, x_col_used, y_col_used)

    elif viz_type == "Scatter Plot" and len(numeric_cols) >= 2:
        c1, c2, c3 = st.columns(3)
        x_col_used = c1.selectbox("X", numeric_cols, key="scat_x")
        y_col_used = c2.selectbox("Y", numeric_cols, index=1, key="scat_y")
        color = c3.selectbox("Color", ["None"] + categorical_cols, key="scat_c")
        fig = plot_scatter(df, x_col_used, y_col_used, None if color == "None" else color)

    elif viz_type == "Histogram" and numeric_cols:
        x_col_used = st.selectbox("Column", numeric_cols, key="hist_x")
        fig = plot_histogram(df, x_col_used)

    elif viz_type == "Correlation Heatmap" and len(numeric_cols) >= 2:
        fig = plot_correlation_heatmap(df, numeric_cols)

    else:
        st.info("Not enough columns of the required type for this chart.")

    if fig:
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("🤖 AI Chart Explanation"):
            with st.spinner("Generating explanation…"):
                expl = explain_chart_with_ai(viz_type, x_col_used or "", y_col_used, df, api_key)
            st.markdown(expl)
            if not api_key:
                st.caption("Add an OpenAI API key for GPT-powered chart explanations.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PREDICTIVE ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab_ml:
    st.subheader("🔮 Predictive Analytics")

    if not numeric_cols:
        st.warning("No numeric columns available for ML.")
    else:
        c1, c2 = st.columns([1, 2])
        with c1:
            target_col = st.selectbox("🎯 Target Column (predict)", df.columns.tolist(), key="ml_target")
            feature_options = [c for c in df.columns if c != target_col]
            feature_cols = st.multiselect(
                "📥 Feature Columns",
                feature_options,
                default=feature_options[:min(5, len(feature_options))],
                key="ml_features"
            )

        with c2:
            st.markdown("**How it works:**")
            st.markdown(
                "DataSense AI trains 4 models (Linear, Tree, Random Forest, Gradient Boosting), "
                "evaluates on an 80/20 train-test split, and shows the best performer with feature importance."
            )

        if st.button("🚀 Run ML Analysis", use_container_width=True, key="run_ml"):
            if len(feature_cols) < 1:
                st.warning("Select at least one feature column.")
            else:
                with st.spinner("Training models…"):
                    result = run_predictive_analysis(df, target_col, feature_cols)

                if "error" in result:
                    st.error(f"ML Error: {result['error']}")
                else:
                    task = result["task"]
                    best = result["best_model_name"]
                    score = result["best_score"]

                    st.success(f"✅ Task: **{task.capitalize()}** · Best model: **{best}** · Score: **{score:.4f}**")

                    st.markdown("#### 📊 Model Comparison")
                    res_df = pd.DataFrame(result["results"]).T
                    st.dataframe(res_df.style.highlight_max(axis=0, color="#d1fae5"), use_container_width=True)

                    fi = result.get("feature_importance")
                    if fi is not None:
                        st.markdown("#### 🔑 Feature Importance")
                        fig = plot_bar_chart(fi, "Feature", "Importance", title=f"Feature Importance — {best}")
                        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.markdown("#### 📈 Linear Trend Forecast")
        fc_col = st.selectbox("Column to forecast", numeric_cols, key="fc_col")
        fc_periods = st.slider("Forecast periods", 5, 50, 10, key="fc_periods")
        if st.button("📈 Generate Forecast", key="fc_btn"):
            with st.spinner("Forecasting…"):
                fc_df = run_forecast(df, fc_col, fc_periods)
            if fc_df is not None:
                import plotly.express as px
                fig = px.line(
                    fc_df, x="Index", y=fc_col, color="Type",
                    title=f"Forecast: {fc_col} (+{fc_periods} periods)",
                    color_discrete_map={"Historical": "#6366f1", "Forecast": "#f59e0b"},
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption("⚠️ Linear extrapolation for trend indication only.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════════════════════
with tab_anomaly:
    st.subheader("🚨 Anomaly Detection")

    if not numeric_cols:
        st.warning("Anomaly detection requires numeric columns.")
    else:
        method = st.radio(
            "Detection method",
            ["Z-Score (Statistical)", "IQR (Box Plot)", "Isolation Forest (ML)"],
            horizontal=True
        )

        col_select = st.multiselect(
            "Columns to analyze",
            numeric_cols,
            default=numeric_cols[:min(4, len(numeric_cols))],
            key="anomaly_cols"
        )

        if col_select and st.button("🔍 Detect Anomalies", use_container_width=True, key="detect_btn"):

            if method == "Z-Score (Statistical)":
                result_df = detect_anomalies_zscore(df, col_select)
                n_anomalies = result_df["_is_anomaly"].sum()
                st.info(f"Found **{n_anomalies}** anomalous rows ({n_anomalies/len(df)*100:.1f}%) with |Z| > 3")

                if n_anomalies > 0:
                    anomaly_rows = result_df[result_df["_is_anomaly"]].sort_values("_anomaly_score", ascending=False)
                    st.dataframe(anomaly_rows[col_select + ["_anomaly_score", "_n_anomalous_cols"]].head(20),
                                 use_container_width=True)
                    if len(col_select) >= 2:
                        import plotly.express as px
                        plot_df = result_df.copy()
                        plot_df["Status"] = plot_df["_is_anomaly"].map({True: "Anomaly", False: "Normal"})
                        fig = px.scatter(plot_df, x=col_select[0], y=col_select[1], color="Status",
                                         color_discrete_map={"Anomaly": "#ef4444", "Normal": "#6366f1"},
                                         title="Anomaly Scatter", template="plotly_white", opacity=0.7)
                        st.plotly_chart(fig, use_container_width=True)

            elif method == "IQR (Box Plot)":
                iqr_results = detect_anomalies_iqr(df, col_select)
                for col, info in iqr_results.items():
                    with st.expander(f"**{col}** — {info['outlier_count']} outliers ({info['outlier_pct']}%)"):
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Lower Bound", f"{info['lower_bound']:,.2f}")
                        c2.metric("Upper Bound", f"{info['upper_bound']:,.2f}")
                        c3.metric("Outliers", info['outlier_count'])
                        if info['outlier_count'] > 0:
                            st.caption(f"Examples: {info['example_outliers']}")
                        fig = plot_histogram(df, col)
                        fig.add_vline(x=info['lower_bound'], line_dash="dash", line_color="#ef4444",
                                      annotation_text="Lower")
                        fig.add_vline(x=info['upper_bound'], line_dash="dash", line_color="#ef4444",
                                      annotation_text="Upper")
                        st.plotly_chart(fig, use_container_width=True)

            elif method == "Isolation Forest (ML)":
                contamination = st.slider("Expected anomaly ratio", 0.01, 0.2, 0.05, 0.01, key="if_contam")
                with st.spinner("Training Isolation Forest…"):
                    result_df, msg = detect_anomalies_isolation_forest(df, col_select, contamination)
                st.info(msg)
                if "_if_anomaly" in result_df.columns:
                    top = result_df[result_df["_if_anomaly"]].sort_values("_if_score", ascending=False)
                    if len(top) > 0:
                        st.dataframe(top[col_select + ["_if_score"]].head(20), use_container_width=True)
                        if len(col_select) >= 2:
                            import plotly.express as px
                            plot_df = result_df.copy()
                            plot_df["Status"] = plot_df["_if_anomaly"].map({True: "Anomaly", False: "Normal"})
                            fig = px.scatter(plot_df, x=col_select[0], y=col_select[1], color="Status",
                                             title="Isolation Forest Anomalies",
                                             color_discrete_map={"Anomaly": "#ef4444", "Normal": "#6366f1"},
                                             template="plotly_white", opacity=0.7)
                            st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab_reco:
    st.subheader("💡 AI Recommendation Engine")
    st.session_state.setdefault("recommendations", [])

    if st.button("🔮 Generate Recommendations", use_container_width=True, key="gen_reco"):
        with st.spinner("Analyzing dataset…"):
            st.session_state.recommendations = generate_recommendations(df, numeric_cols, categorical_cols, api_key)

    recs = st.session_state.recommendations
    if recs:
        priority_colors = {
            "high": ("#fef2f2", "#991b1b", "🔴"),
            "medium": ("#fffbeb", "#92400e", "🟡"),
            "low": ("#f0fdf4", "#166534", "🟢"),
            "info": ("#eff6ff", "#1e40af", "🔵"),
        }
        for rec in sorted(recs, key=lambda x: {"high": 0, "medium": 1, "low": 2, "info": 3}.get(x.get("priority", "info"), 3)):
            p = rec.get("priority", "info")
            bg, text_color, icon = priority_colors.get(p, priority_colors["info"])
            st.markdown(f"""
            <div style="background:{bg};border-left:4px solid {text_color};border-radius:8px;
                        padding:14px 18px;margin:8px 0">
                <div style="font-weight:700;color:{text_color};font-size:0.9rem">
                    {icon} {rec['category']} · {rec['title']}
                </div>
                <div style="color:#374151;font-size:0.85rem;margin-top:4px">{rec['description']}</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("Click 'Generate Recommendations' to get AI-powered dataset suggestions.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — CHAT (with voice)
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.subheader("💬 Chat with Your Data")

    with st.expander("🎙️ Voice Query", expanded=False):
        st.caption("Upload audio to ask questions hands-free (requires OpenAI API key).")
        audio_path = render_voice_input()
        if audio_path:
            if st.button("📝 Transcribe & Ask", key="transcribe_btn"):
                with st.spinner("Transcribing…"):
                    transcribed, msg = transcribe_audio(audio_path, api_key)
                st.caption(msg)
                if transcribed:
                    st.session_state.setdefault("chat_history", [])
                    st.session_state.chat_history.append({"role": "user", "content": f"🎙️ {transcribed}"})
                    with st.spinner("Analyzing…"):
                        answer = chat_with_data(df, transcribed, api_key)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    st.rerun()

    st.session_state.setdefault("chat_history", [])
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    with st.expander("💡 Example Questions"):
        for ex in ["Which column has the highest average value?", "How many missing values are there?",
                   "What is the most common category?", "Summarize this dataset in 3 sentences."]:
            if st.button(ex, key=f"ex_{ex}"):
                st.session_state.chat_history.append({"role": "user", "content": ex})
                with st.spinner("Thinking…"):
                    answer = chat_with_data(df, ex, api_key)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.rerun()

    user_q = st.chat_input("Ask anything about your dataset…")
    if user_q:
        st.session_state.chat_history.append({"role": "user", "content": user_q})
        with st.spinner("Analyzing…"):
            answer = chat_with_data(df, user_q, api_key)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 8 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab_dash:
    st.subheader("📊 Auto-Generated Dashboard")
    render_dashboard(df, numeric_cols, categorical_cols)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 9 — EXPORT PDF
# ══════════════════════════════════════════════════════════════════════════════
with tab_export:
    st.subheader("📄 Export Analytics Report as PDF")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Report includes:**")
        st.markdown("- Cover page with dataset metadata\n- Data quality score\n- Descriptive statistics table\n- All auto-detected insights")
    with c2:
        st.markdown("&nbsp;")
        st.markdown("- AI recommendations\n- Embedded charts (requires kaleido)\n- Data sample (first 20 rows)\n- Branded footer")

    st.divider()
    st.markdown("**Select charts to embed:**")
    include_bar = st.checkbox("Bar Chart", value=bool(categorical_cols and numeric_cols), key="pdf_bar")
    include_pie = st.checkbox("Pie Chart", value=bool(categorical_cols), key="pdf_pie")
    include_corr = st.checkbox("Correlation Heatmap", value=len(numeric_cols) >= 3, key="pdf_corr",
                                disabled=len(numeric_cols) < 3)

    if st.button("📄 Generate PDF Report", use_container_width=True, type="primary", key="gen_pdf"):
        with st.spinner("Building report…"):
            try:
                from utils.pdf_report import generate_pdf_report

                figures_for_pdf = []
                if include_bar and categorical_cols and numeric_cols:
                    figures_for_pdf.append((
                        plot_bar_chart(df, categorical_cols[0], numeric_cols[0]),
                        f"{numeric_cols[0]} by {categorical_cols[0]}"
                    ))
                if include_pie and categorical_cols:
                    figures_for_pdf.append((
                        plot_pie_chart(df, categorical_cols[0], numeric_cols[0] if numeric_cols else None),
                        f"Distribution of {categorical_cols[0]}"
                    ))
                if include_corr and len(numeric_cols) >= 3:
                    figures_for_pdf.append((
                        plot_correlation_heatmap(df, numeric_cols[:8]),
                        "Correlation Heatmap"
                    ))

                recs = generate_recommendations(df, numeric_cols, categorical_cols, api_key)

                pdf_bytes = generate_pdf_report(
                    df=df,
                    numeric_cols=numeric_cols,
                    categorical_cols=categorical_cols,
                    insights=insights,
                    quality_info=quality,
                    recommendations=recs,
                    figures=figures_for_pdf,
                )

                st.success("✅ PDF report ready!")
                st.download_button(
                    label="⬇️ Download PDF Report",
                    data=pdf_bytes,
                    file_name="datasense_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    type="primary"
                )

            except ImportError:
                st.error("reportlab not installed. Run: `pip install reportlab`")
            except Exception as e:
                st.error(f"Report generation failed: {e}")
                st.caption("For chart embedding in PDFs, also install: `pip install kaleido`")
