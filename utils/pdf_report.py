"""
utils/pdf_report.py
Generate a comprehensive PDF analytics report using ReportLab.
"""

import pandas as pd
import numpy as np
import io
import base64
from datetime import datetime
from typing import List, Dict, Any, Optional

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.colors import HexColor, white, black, grey
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, HRFlowable, Image as RLImage, KeepTogether
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


def fig_to_image_bytes(fig, width=450, height=280) -> Optional[bytes]:
    """Convert a Plotly figure to PNG bytes for embedding in PDF."""
    try:
        return fig.to_image(format="png", width=width, height=height)
    except Exception:
        try:
            import plotly.io as pio
            return pio.to_image(fig, format="png", width=width, height=height)
        except Exception:
            return None


def generate_pdf_report(
    df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    insights: List[Dict],
    quality_info: Optional[Dict] = None,
    recommendations: Optional[List[Dict]] = None,
    figures: Optional[List] = None,
    filename: str = "datasense_report.pdf"
) -> bytes:
    """
    Build a full PDF analytics report.
    Returns raw PDF bytes.
    """
    if not REPORTLAB_AVAILABLE:
        raise ImportError("reportlab not installed. Run: pip install reportlab")

    buffer = io.BytesIO()

    # ── Document setup ────────────────────────────────────────────────────────
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2*cm,
        leftMargin=2*cm,
        topMargin=2.5*cm,
        bottomMargin=2*cm,
        title="DataSense AI Analytics Report"
    )

    # ── Styles ────────────────────────────────────────────────────────────────
    base_styles = getSampleStyleSheet()

    INDIGO = HexColor("#4f46e5")
    LIGHT_BG = HexColor("#f1f5f9")
    SUCCESS = HexColor("#166534")
    WARNING = HexColor("#92400e")
    DANGER = HexColor("#991b1b")

    title_style = ParagraphStyle(
        "CustomTitle",
        parent=base_styles["Title"],
        fontSize=26,
        textColor=INDIGO,
        spaceAfter=6,
        alignment=TA_CENTER,
        fontName="Helvetica-Bold"
    )
    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=base_styles["Normal"],
        fontSize=10,
        textColor=grey,
        spaceAfter=20,
        alignment=TA_CENTER
    )
    h1 = ParagraphStyle("H1", parent=base_styles["Heading1"], fontSize=16, textColor=INDIGO,
                         spaceBefore=20, spaceAfter=8, fontName="Helvetica-Bold")
    h2 = ParagraphStyle("H2", parent=base_styles["Heading2"], fontSize=12, textColor=INDIGO,
                         spaceBefore=12, spaceAfter=6, fontName="Helvetica-Bold")
    normal = ParagraphStyle("N", parent=base_styles["Normal"], fontSize=9, spaceAfter=4, leading=14)
    small = ParagraphStyle("S", parent=base_styles["Normal"], fontSize=8, textColor=grey, spaceAfter=2)
    bullet = ParagraphStyle("B", parent=normal, leftIndent=14, firstLineIndent=-10, spaceAfter=3)

    story = []

    # ── Cover Page ────────────────────────────────────────────────────────────
    story.append(Spacer(1, 2*cm))
    story.append(Paragraph("DataSense AI", title_style))
    story.append(Paragraph("Automated Analytics Report", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=INDIGO, spaceAfter=16))

    # Dataset summary box
    meta_data = [
        ["Generated", datetime.now().strftime("%B %d, %Y  %H:%M")],
        ["Dataset Shape", f"{df.shape[0]:,} rows × {df.shape[1]} columns"],
        ["Numeric Columns", f"{len(numeric_cols)} ({', '.join(numeric_cols[:4])}{'...' if len(numeric_cols)>4 else ''})"],
        ["Categorical Columns", f"{len(categorical_cols)} ({', '.join(categorical_cols[:4])}{'...' if len(categorical_cols)>4 else ''})"],
        ["Missing Values", f"{df.isnull().sum().sum():,} total"],
    ]
    meta_table = Table(meta_data, colWidths=[4*cm, 12*cm])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), LIGHT_BG),
        ("TEXTCOLOR", (0, 0), (0, -1), INDIGO),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#e0e7ff")),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [white, LIGHT_BG]),
        ("PADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(meta_table)

    if quality_info:
        story.append(Spacer(1, 16))
        score = quality_info.get("total_score", 0)
        grade = quality_info.get("grade", "?")
        label = quality_info.get("grade_label", "")
        story.append(Paragraph(
            f"Data Quality Score: <b>{score}/100</b> — Grade <b>{grade}</b> ({label})",
            ParagraphStyle("QS", parent=normal, alignment=TA_CENTER, fontSize=11, textColor=INDIGO)
        ))

    story.append(PageBreak())

    # ── Section 1: Descriptive Statistics ────────────────────────────────────
    story.append(Paragraph("1. Descriptive Statistics", h1))
    story.append(HRFlowable(width="100%", thickness=1, color=HexColor("#e0e7ff"), spaceAfter=10))

    if numeric_cols:
        stats = df[numeric_cols].describe().T.round(3).reset_index()
        stats.columns = ["Column"] + list(stats.columns[1:])
        table_data = [stats.columns.tolist()] + stats.values.tolist()

        col_w = [3.5*cm] + [1.8*cm] * (len(table_data[0]) - 1)
        t = Table(table_data, colWidths=col_w, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), INDIGO),
            ("TEXTCOLOR", (0, 0), (-1, 0), white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7.5),
            ("GRID", (0, 0), (-1, -1), 0.4, HexColor("#e0e7ff")),
            ("ROWBACKGROUNDS", (1, 1), (-1, -1), [white, LIGHT_BG]),
            ("PADDING", (0, 0), (-1, -1), 5),
            ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ]))
        story.append(t)
    else:
        story.append(Paragraph("No numeric columns available for statistics.", normal))

    story.append(PageBreak())

    # ── Section 2: AI Insights ────────────────────────────────────────────────
    story.append(Paragraph("2. Automated Insights", h1))
    story.append(HRFlowable(width="100%", thickness=1, color=HexColor("#e0e7ff"), spaceAfter=10))

    if insights:
        for ins in insights[:12]:
            icon_map = {"success": "✓", "warning": "⚠", "info": "●"}
            color_map = {"success": SUCCESS, "warning": WARNING, "info": INDIGO}
            t = ins.get("type", "info")
            icon = icon_map.get(t, "●")
            color = color_map.get(t, INDIGO)
            row = Table(
                [[Paragraph(f"<b>{ins['title']}</b>", ParagraphStyle("it", parent=normal, textColor=color)),
                  Paragraph(ins['value'], normal)]],
                colWidths=[8*cm, 8*cm]
            )
            row.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BG),
                ("GRID", (0, 0), (-1, -1), 0, white),
                ("PADDING", (0, 0), (-1, -1), 6),
                ("ROUNDEDCORNERS", [4]),
            ]))
            story.append(row)
            story.append(Spacer(1, 4))
    else:
        story.append(Paragraph("No automated insights detected.", normal))

    # ── Section 3: Recommendations ───────────────────────────────────────────
    if recommendations:
        story.append(PageBreak())
        story.append(Paragraph("3. Recommendations", h1))
        story.append(HRFlowable(width="100%", thickness=1, color=HexColor("#e0e7ff"), spaceAfter=10))

        for rec in recommendations[:8]:
            priority = rec.get("priority", "info")
            priority_color = {"high": DANGER, "medium": WARNING, "low": SUCCESS, "info": INDIGO}.get(priority, INDIGO)
            story.append(Paragraph(f"<b>{rec['category']}: {rec['title']}</b>",
                                   ParagraphStyle("rt", parent=h2, textColor=priority_color, fontSize=10)))
            story.append(Paragraph(rec.get("description", ""), normal))
            story.append(Spacer(1, 6))

    # ── Section 4: Charts ────────────────────────────────────────────────────
    if figures:
        story.append(PageBreak())
        story.append(Paragraph("4. Visualizations", h1))
        story.append(HRFlowable(width="100%", thickness=1, color=HexColor("#e0e7ff"), spaceAfter=10))

        for i, (fig, fig_title) in enumerate(figures[:6]):
            try:
                img_bytes = fig_to_image_bytes(fig, width=500, height=300)
                if img_bytes:
                    story.append(Paragraph(fig_title, h2))
                    img_io = io.BytesIO(img_bytes)
                    img = RLImage(img_io, width=15*cm, height=9*cm)
                    story.append(img)
                    story.append(Spacer(1, 12))
            except Exception:
                story.append(Paragraph(f"[Chart: {fig_title} — could not render]", small))

    # ── Section 5: Data Sample ────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("5. Data Sample (first 20 rows)", h1))
    story.append(HRFlowable(width="100%", thickness=1, color=HexColor("#e0e7ff"), spaceAfter=10))

    sample = df.head(20)
    cols_to_show = sample.columns[:8].tolist()  # max 8 cols to fit page
    sample = sample[cols_to_show].fillna("—").astype(str)

    table_data = [cols_to_show] + sample.values.tolist()
    col_w = [16*cm / len(cols_to_show)] * len(cols_to_show)
    t = Table(table_data, colWidths=col_w, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), INDIGO),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("GRID", (0, 0), (-1, -1), 0.4, HexColor("#e0e7ff")),
        ("ROWBACKGROUNDS", (1, 1), (-1, -1), [white, LIGHT_BG]),
        ("PADDING", (0, 0), (-1, -1), 4),
        ("WORDWRAP", (0, 0), (-1, -1), True),
    ]))
    story.append(t)

    # ── Footer note ───────────────────────────────────────────────────────────
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=1, color=HexColor("#e0e7ff")))
    story.append(Paragraph(
        f"Generated by DataSense AI · {datetime.now().strftime('%Y-%m-%d %H:%M')} · Confidential",
        ParagraphStyle("footer", parent=small, alignment=TA_CENTER)
    ))

    # ── Build ─────────────────────────────────────────────────────────────────
    doc.build(story)
    buffer.seek(0)
    return buffer.read()
