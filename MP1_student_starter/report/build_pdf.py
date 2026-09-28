"""Render the six-section Chinese report. Optional: pip install reportlab==5.0.1."""
from pathlib import Path
import argparse
import re
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--font', type=Path, help='Optional CJK TrueType font to embed in the PDF.')
args = parser.parse_args()
FONT = 'ReportFont' if args.font else 'STSong-Light'
pdfmetrics.registerFont(TTFont(FONT, str(args.font)) if args.font else UnicodeCIDFont(FONT))
BODY = ParagraphStyle('body', fontName=FONT, fontSize=10.5, leading=16,
                      spaceAfter=9, wordWrap='CJK', textColor=colors.HexColor('#17242C'))
TITLE = ParagraphStyle('title', parent=BODY, fontSize=21, leading=28, spaceAfter=16)
HEADING = ParagraphStyle('heading', parent=BODY, fontSize=16, leading=23, spaceAfter=15)
CELL = ParagraphStyle('cell', parent=BODY, fontSize=9, leading=13, spaceAfter=0)
SMALL = ParagraphStyle('small', parent=BODY, fontSize=8.5, leading=12)


def inline(text):
    # Plain code spans remain readable with the same CJK-capable typeface.
    text = re.sub(r'`([^`]+)`', r'\1', text)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\1 (\2)', text)
    return escape(text)


def footer(canvas, doc):
    canvas.setStrokeColor(colors.HexColor('#C9D5DD'))
    canvas.line(48, 40, A4[0]-48, 40)
    canvas.setFont(FONT, 8)
    canvas.setFillColor(colors.HexColor('#52616C'))
    canvas.drawString(48, 27, 'DASE7506 / MP1    学号 3036797441    2026-09-28')
    canvas.drawRightString(A4[0]-48, 27, str(doc.page))


def build():
    lines = (ROOT/'REPORT.md').read_text().splitlines()
    flow = []
    index = 0
    section = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line.startswith('## '):
            if section:
                flow.append(PageBreak())
            section += 1
            flow.append(Paragraph(inline(line[3:]), HEADING))
        elif line.startswith('# '):
            flow.append(Paragraph(inline(line[2:]), TITLE))
        elif line.startswith('|'):
            rows = []
            while index < len(lines) and lines[index].strip().startswith('|'):
                cells = [c.strip() for c in lines[index].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[:\- ]+', c) for c in cells):
                    rows.append([Paragraph(inline(c), CELL) for c in cells])
                index += 1
            width = A4[0]-96
            columns = len(rows[0])
            table = Table(rows, colWidths=[width/columns]*columns, repeatRows=1, hAlign='LEFT')
            table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#DDEBF1')),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F4F7F9')]),
                ('LINEBELOW', (0,0), (-1,0), .6, colors.HexColor('#87A3B3')),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('LEFTPADDING', (0,0), (-1,-1), 7), ('RIGHTPADDING', (0,0), (-1,-1), 7),
                ('TOPPADDING', (0,0), (-1,-1), 7), ('BOTTOMPADDING', (0,0), (-1,-1), 7),
            ]))
            flow.extend([table, Spacer(1, 12)])
            continue
        elif line.startswith('### '):
            flow.append(Paragraph(inline(line[4:]), ParagraphStyle('sub', parent=BODY, fontSize=12, leading=18)))
        else:
            paragraph = line
            while index+1 < len(lines) and lines[index+1].strip() and not lines[index+1].startswith(('#', '|', '- ')):
                index += 1
                paragraph += ' ' + lines[index].strip()
            flow.append(Paragraph(inline(paragraph), BODY))
        index += 1
    doc = SimpleDocTemplate(str(ROOT/'REPORT.pdf'), pagesize=A4,
                           rightMargin=48, leftMargin=48, topMargin=45, bottomMargin=55,
                           title='MP1: EMA 参数平均的同预算实验', author='3036797441')
    doc.build(flow, onFirstPage=footer, onLaterPages=footer)
    print(ROOT/'REPORT.pdf')


if __name__ == '__main__':
    build()
