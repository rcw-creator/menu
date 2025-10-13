
import os, re
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

# ---------------- Font registration Latex ----------------
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

# ---------------- Page numbers ----------------
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

# ---------------- Header ----------------
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

# ---------------- Text formatting ----------------
def emphasize_keywords_gray_italic(text):
    italic_face = face("Italic")
    def repl(m):
        return f'<font name="{italic_face}" color="#181717" >{m.group(0)}</font>'
    text = re.sub(r"\b(exploitation)\b", repl, text, flags=re.IGNORECASE)
    text = re.sub(r"\b(exploration)\b", repl, text, flags=re.IGNORECASE)
    return text

# ---------------- Report builder ----------------
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

# ---------------- MAIN ----------------
if __name__ == "__main__":
    report_data = {
        "title": "fgs asgsfh saghsrthgtrygh",
        "author": "sfgsg",
        "document_title": "sgfsgh trstryghsrth", 
        "model_dates": "Mosghtghjyh sth srthshtsrhsth", 
        "sections": [
            {
                "model_name": "shtshstrh strgyhs strhst srthgstrh",
                "details": [
                    "Tsgfhg cesssghsh rate: 0.sghtsfh.",
                    "shshhration for new shfgshh."
                ],
                "lift_chart": "english_lift.png",
                "score_dist": "english_score.png"
            },
            {
                "model_name": "sghsfgh tsh stghshs ",
                "details": [
                    "sghfs sg sghfg stgh .",
                    "sghs sgh strghsrtghsrhtsh."
                ],
                "lift_chart": "english_lift.png",
                "score_dist": "english_score.png"
            }
        ]
    }

    create_gbm_report("adfgafg.pdf", report_data)
