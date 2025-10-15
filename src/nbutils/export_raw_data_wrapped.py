import textwrap
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Preformatted
from reportlab.lib.styles import ParagraphStyle

def export_raw_data_wrapped(all_raw_data: dict, pdf_filename="VQE_Raw_Data_Wrapped.pdf",
                            font_size=8, wrap_width=120):
    """
    Export a Python dictionary to PDF as a wrapped code block.
    This version isolates pprint to avoid shadowing issues in Jupyter.
    """
    # --- Local import of the real pprint module ---
    import importlib, sys
    import pprint as real_pprint
    importlib.reload(real_pprint)

    # Convert dict to pretty string
    raw_text = real_pprint.pformat(all_raw_data, indent=2, width=wrap_width)

    # Wrap each line to wrap_width characters
    import textwrap
    wrapped_lines = []
    for line in raw_text.splitlines():
        wrapped_lines.extend(textwrap.wrap(line, width=wrap_width, replace_whitespace=False) or [''])
    wrapped_text = "\n".join(wrapped_lines)

    # PDF setup with minimal margins
    doc = SimpleDocTemplate(pdf_filename, pagesize=letter,
                            leftMargin=0.2*72, rightMargin=0.2*72,
                            topMargin=0.2*72, bottomMargin=0.2*72)

    # Monospace style for code block
    code_style = ParagraphStyle(
        'codeblock',
        fontName='Courier',
        fontSize=font_size,
        leading=font_size + 2,
        leftIndent=0,
        rightIndent=0,
        spaceBefore=0,
        spaceAfter=0
    )

    # Build PDF
    doc.build([Preformatted(wrapped_text, code_style)])
    print(f"✅ PDF saved as '{pdf_filename}'")
