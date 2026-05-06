from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib.utils import simpleSplit


def md_to_pdf(md_path, pdf_path):
    with open(md_path, 'r', encoding='utf-8') as f:
        text = f.read()

    width, height = A4
    margin = 0.5 * inch
    max_width = width - 2 * margin
    y = height - margin
    leading = 12

    c = canvas.Canvas(pdf_path, pagesize=A4)
    c.setFont('Helvetica', 10)

    # Split into paragraphs separated by blank lines
    paragraphs = text.split('\n\n')
    for para in paragraphs:
        # Replace tabs with spaces
        para = para.replace('\t', '    ').strip()
        if not para:
            y -= leading
            continue
        # Wrap paragraph into lines that fit page width.
        lines = simpleSplit(para, 'Helvetica', 10, max_width)
        for line in lines:
            if y < margin + leading:
                c.showPage()
                c.setFont('Helvetica', 10)
                y = height - margin
            c.drawString(margin, y, line)
            y -= leading
        y -= leading  # blank line between paragraphs

    c.save()


if __name__ == '__main__':
    import os
    cwd = os.path.dirname(os.path.abspath(__file__))
    md = os.path.join(cwd, 'VIVA_PREPARATION.md')
    pdf = os.path.join(cwd, 'VIVA_PREPARATION.pdf')
    if not os.path.exists(md):
        print('Markdown file not found:', md)
    else:
        md_to_pdf(md, pdf)
        print('PDF created at', pdf)
