# weekly_gbm_full_pipeline.py
# Requires: pip install reportlab matplotlib scikit-learn pandas numpy

import os
import re
import io
import json
import base64
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

# ------------------ (BEGIN) Your create_gbm_report function (copied/adapted) ------------------
# I used your create_gbm_report exactly (signature: output_path, report_data).
# If you already have this function in a separate file, you can import it instead.
import os as _os
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Image
)
from reportlab.pdfgen import canvas

def try_register_censcbk():
    font_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = {
        "Regular": ["CENSCBK.TTF", "censcbk.ttf", "schlbk.ttf"],
        "Bold": ["CENSCBKB.TTF", "censcbk-bd.ttf", "schlbkb.ttf"],
        "Italic": ["CENSCBKI.TTF", "censcbk-i.ttf", "schlbki.ttf"],
        "BoldItalic": ["CENSCBKBI.TTF", "censcbk-bi.ttf", "schlbkbi.ttf"],
    }
    registered = {}
    for face, names in candidates.items():
        for name in names:
            path = os.path.join(font_dir, name)
            if os.path.exists(path):
                try:
                    regname = f"Censcbk-{face}"
                    pdfmetrics.registerFont(TTFont(regname, path))
                    registered[face] = regname
                    break
                except Exception:
                    pass
    return registered

FONT_MAP = try_register_censcbk()
USE_CENS = bool(FONT_MAP)

def face(name):
    if USE_CENS:
        return FONT_MAP.get(name, {
            "Regular": "Times-Roman",
            "Bold": "Times-Bold",
            "Italic": "Times-Italic",
            "BoldItalic": "Times-BoldItalic"
        }[name])
    else:
        return {
            "Regular": "Times-Roman",
            "Bold": "Times-Bold",
            "Italic": "Times-Italic",
            "BoldItalic": "Times-BoldItalic"
        }[name]

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self._draw_page_number()
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def _draw_page_number(self):
        self.saveState()
        self.setFont(face("Regular"), 9)
        page_w = float(self._pagesize[0])
        text = str(self._pageNumber)
        self.drawCentredString(page_w / 2.0, 0.45 * inch, text)
        self.restoreState()

def draw_header(cnv, doc):
    try:
        cnv.setTitle(getattr(doc, "pdf_title", ""))
    except Exception:
        pass

    cnv.saveState()
    cnv.setFont(face("Regular"), 9)
    cnv.setFillColorRGB(0.55, 0.55, 0.55)
    x_offset = doc.leftMargin + 6
    y_pos = doc.height + doc.topMargin - 6
    cnv.drawString(x_offset, y_pos, getattr(doc, "model_date_header", ""))
    cnv.restoreState()

def emphasize_keywords_gray_italic(text):
    italic_face = face("Italic")
    def repl(m):
        return f'<font name="{italic_face}" color="#181717" >{m.group(0)}</font>'
    text = re.sub(r"\b(exploitation)\b", repl, text, flags=re.IGNORECASE)
    text = re.sub(r"\b(exploration)\b", repl, text, flags=re.IGNORECASE)
    return text

def create_gbm_report(output_path, report_data):
    left_margin = 50
    right_margin = 40
    top_margin = 25
    bottom_margin = 35
    column_gap = 12

    doc = BaseDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=left_margin,
        rightMargin=right_margin,
        topMargin=top_margin,
        bottomMargin=bottom_margin,
        title=report_data.get("document_title", ""),
        author=report_data.get("author", "")
    )

    doc.model_date_header = report_data.get("model_dates", "")
    doc.pdf_title = report_data.get("document_title", "")

    # Layout frames
    page_w, page_h = A4
    usable_w = page_w - left_margin - right_margin
    col_w = (usable_w - column_gap) / 2.0
    col_h = page_h - top_margin - bottom_margin
    header_gap = 20

    frame_left = Frame(left_margin, bottom_margin, col_w, col_h - header_gap, id="left", topPadding=header_gap)
    frame_right = Frame(left_margin + col_w + column_gap, bottom_margin, col_w, col_h - header_gap, id="right", topPadding=header_gap)
    doc.addPageTemplates([PageTemplate(id="TwoCol", frames=[frame_left, frame_right], onPage=draw_header)])

    # Styles
    base = getSampleStyleSheet()
    styles = {
        "TitleLeft": ParagraphStyle(name="TitleLeft", parent=base["Heading1"],
            fontName=face("Regular"), fontSize=27, leading=34, alignment=TA_LEFT, spaceAfter=35),
        "Author": ParagraphStyle(name="Author", parent=base["Normal"],
            fontName=face("Italic"), fontSize=11, leading=14, textColor=colors.HexColor("#181717"),
            leftIndent=12, alignment=TA_LEFT, spaceAfter=12),
        "Section": ParagraphStyle(name="Section", parent=base["Heading2"],
            fontName=face("Regular"), fontSize=14, leading=18, spaceBefore=10, spaceAfter=10,
            textColor=colors.HexColor("#111111")),
        "Body": ParagraphStyle(name="Body", parent=base["Normal"],
            fontName=face("Regular"), fontSize=11, leading=15),
        "Bullet": ParagraphStyle(name="Bullet", parent=base["Normal"],
            fontName=face("Regular"), fontSize=11, leading=15, leftIndent=28, firstLineIndent=-12,
            bulletFontName=face("Bold"), bulletFontSize=14, bulletIndent=10,
            spaceBefore=4, spaceAfter=6)
    }

    story = []
    story.append(Paragraph(report_data.get("title", ""), styles["TitleLeft"]))
    if report_data.get("author"):
        story.append(Paragraph(report_data["author"], styles["Author"]))
    story.append(Spacer(1, 8))

    if report_data.get("overview"):
        story.append(Paragraph(report_data["overview"], styles["Body"]))
        story.append(Spacer(1, 8))

    for idx, sec in enumerate(report_data.get("sections", []), start=1):
        if idx > 1:
            story.append(Spacer(1, 20))

        story.append(Paragraph(f"{idx}. {sec.get('model_name','')}", styles["Section"]))
        story.append(Spacer(1, 8))

        for detail in sec.get("details", []):
            story.append(Paragraph(emphasize_keywords_gray_italic(detail), styles["Bullet"], bulletText="•"))
        story.append(Spacer(1, 10))

        img_w = col_w - 8
        img_h = 150
        for chart_key in ("lift_chart", "score_dist"):
            path = sec.get(chart_key)
            if path:
                if os.path.exists(path):
                    story.append(Image(path, width=img_w, height=img_h))
                else:
                    story.append(Paragraph(f"[Missing image: {path}]", styles["Body"]))
                story.append(Spacer(1, 10))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"✅ PDF generated: {output_path}")
# ------------------ (END) create_gbm_report ------------------


# ------------------ Pipeline: fake data, charts, PDFs, audit ------------------

def fetch_data_fake():
    """Generate fake dataset similar to your DB schema."""
    np.random.seed(42)
    experiments = [
        "Referral_English_cMAB_GBM_",
        "Referral_Spanish_cMAB_GBM_",
        "Referral_Portuguese_cMAB_GBM_"
    ]
    start = datetime(2025, 8, 1)
    weeks = [(start + timedelta(weeks=i)).strftime("%Y-W%U") for i in range(6)]
    rows = []
    for week in weeks:
        for exp in experiments:
            n = np.random.randint(800, 1200)
            scores = np.random.beta(2, 5, size=n)
            # create labels correlated with score (noisy)
            probs = 0.2 + 0.6 * scores  # base rate scaled by score
            labels = np.random.rand(n) < probs
            for s, l in zip(scores, labels):
                rows.append({
                    "experiment_name": exp,
                    "week": week,
                    "model_prediction_score": float(s),
                    "actual_success": bool(l)
                })
    df = pd.DataFrame(rows)
    print(f"✅ Fake dataset rows: {len(df)}")
    return df

def create_lift_chart(df_exp, experiment_name, outpath):
    """
    Create lift chart with background bars for volume + line for lift.
    Returns: auroc (float), lift_volume_pairs (list of (lift, volume)), and saved path.
    """
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)

    y_true = df_exp["actual_success"].astype(int)
    y_score = df_exp["model_prediction_score"].astype(float)
    # guard: need at least one class present for roc_auc_score
    try:
        auroc = float(roc_auc_score(y_true, y_score))
    except Exception:
        auroc = float("nan")

    # sort descending by score
    df_sorted = df_exp.sort_values("model_prediction_score", ascending=False).reset_index(drop=True)

    # create deciles (handle duplicates by 'duplicates="drop"' not available reliably for small groups,
    # so we fallback to a rank-based approach)
    try:
        df_sorted["decile"] = pd.qcut(df_sorted["model_prediction_score"], 10, labels=False, duplicates="drop")
        # ensure all 0..9 present: if less, reindex later
    except Exception:
        # fallback: use rank percentile
        df_sorted["decile"] = (df_sorted["model_prediction_score"].rank(method="first", pct=True) * 10).astype(int)
        df_sorted.loc[df_sorted["decile"] == 10, "decile"] = 9

    # compute lift (avg success) and volume per decile, we want decile=0 as top scores (qcut labels are ascending)
    # so map to rank where 1 = top decile
    grouped = df_sorted.groupby("decile").agg(
        lift=("actual_success", "mean"),
        volume=("actual_success", "size")
    ).sort_index(ascending=True)

    # If qcut produced fewer than 10 bins, fill missing deciles with zeros
    all_deciles = list(range(0, 10))
    grouped = grouped.reindex(all_deciles, fill_value=0)

    # We want decile 1 to be top scores, so reverse order for plotting (1..10)
    lift_list = grouped["lift"].tolist()[::-1]      # top decile first
    vol_list = grouped["volume"].tolist()[::-1]

    # Plot: bars (volume scaled) behind line (lift)
    fig, ax1 = plt.subplots(figsize=(5.2, 2.6))
    # scale volume to be visually comparable
    vol = np.array(vol_list)
    if vol.max() > 0:
        vol_scaled = vol / vol.max() * (max(lift_list) if max(lift_list) > 0 else 1)
    else:
        vol_scaled = vol

    x = np.arange(1, len(lift_list) + 1)
    ax1.bar(x, vol_scaled, color="#e0e0e0", alpha=0.7, width=0.8, label="Volume (scaled)")
    ax1.plot(x, lift_list, marker="o", color="#1f77b4", linewidth=2, label="Lift")
    ax1.set_xlabel("Decile (1 = top scores)")
    ax1.set_ylabel("Average success rate")
    ax1.set_title(f"{experiment_name}  (AUROC={np.nan if np.isnan(auroc) else round(auroc,3)})")
    ax1.set_xticks(x)
    ax1.set_xlim(0.5, len(x) + 0.5)

    ax2 = ax1.twinx()
    ax2.set_ylim(0, vol.max() if len(vol)>0 else 1)
    ax2.set_ylabel("Volume (count)")

    ax1.legend(loc="upper right", frameon=False)
    plt.tight_layout()
    plt.savefig(outpath, dpi=120)
    plt.close(fig)

    # return auroc, pairs (lift, volume)
    lift_volume_pairs = [(float(round(l, 6) if not np.isnan(l) else 0.0), int(v)) for l, v in zip(lift_list, vol_list)]
    return auroc, lift_volume_pairs, outpath

def create_score_distribution_chart(df_exp, experiment_name, outpath):
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    plt.figure(figsize=(5.2, 2.6))
    y_true = df_exp["actual_success"].astype(int)
    y_score = df_exp["model_prediction_score"].astype(float)
    plt.hist(y_score[y_true==1], bins=20, alpha=0.6, label="Success", color="#2ca02c")
    plt.hist(y_score[y_true==0], bins=20, alpha=0.4, label="Failure", color="#d62728")
    plt.legend(frameon=False, fontsize=8)
    plt.title(f"Score distribution ({experiment_name})")
    plt.xlabel("Model prediction score")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outpath, dpi=120)
    plt.close()

# ---------------- generate weekly reports pipeline ----------------
def generate_weekly_reports_and_audit():
    df = fetch_data_fake()
    os.makedirs("charts", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    audit_rows = []

    for week, week_df in df.groupby("week"):
        print(f"Generating week: {week} ...")
        sections = []
        lift_chart_data = []

        # group by experiment within the week
        for exp, exp_df in week_df.groupby("experiment_name"):
            # safe filenames
            safe_exp = exp.replace("/", "_").replace(" ", "_")
            lift_path = os.path.join("charts", f"{week}__{safe_exp}__lift.png")
            score_path = os.path.join("charts", f"{week}__{safe_exp}__score.png")

            auroc, lift_pairs, lift_path = create_lift_chart(exp_df, exp, lift_path)
            create_score_distribution_chart(exp_df, exp, score_path)

            # collect lift chart data for audit
            lift_chart_data.append({
                "experiment_name": exp,
                "auroc": round(float(auroc) if not np.isnan(auroc) else 0.0, 4),
                "lift_chart": lift_pairs
            })

            # prepare section for PDF
            details = [
                f"AUROC: {round(float(auroc) if not np.isnan(auroc) else 0.0, 4)}",
                f"Train size: {len(exp_df)}",
                f"Success rate: {exp_df['actual_success'].mean() * 100:.2f}%"
            ]
            sections.append({
                "model_name": exp,
                "details": details,
                "lift_chart": lift_path,
                "score_dist": score_path
            })

        # build report_data structure that your create_gbm_report expects
        report_data = {
            "title": f"Referral GBM Models Report - Week {week}",
            "document_title": f"Referral GBM Models Report - Week {week}",
            "author": "Data Science Team",
            "model_dates": f"Week: {week}",
            "overview": f"This report summarizes GBM model performance and distributions for week {week}.",
            "sections": sections
        }

        pdf_path = os.path.join("reports", f"Referral_GBM_Report_Week_{week}.pdf")
        # CALL your create_gbm_report exactly: (output_path, report_data)
        create_gbm_report(pdf_path, report_data)

        # encode pdf
        with open(pdf_path, "rb") as f:
            pdf_b64 = base64.b64encode(f.read()).decode("utf-8")

        audit_rows.append({
            "week": week,
            "lift_chart_data": json.dumps(lift_chart_data),
            "pdf_binary": pdf_b64
        })

    audit_df = pd.DataFrame(audit_rows)
    audit_csv = os.path.join("reports", "audit_table.csv")
    audit_df.to_csv(audit_csv, index=False)
    print(f"✅ Done. Reports in ./reports, charts in ./charts, audit CSV: {audit_csv}")

# ---------------- main ----------------
if __name__ == "__main__":
    generate_weekly_reports_and_audit()
