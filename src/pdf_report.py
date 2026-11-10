
from io import BytesIO
from datetime import datetime
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak
)


def generate_pdf_report(document_name, matches):

    buffer = BytesIO()

    pdf = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()
    content = []

    content.append(
        Paragraph(
            "Plagiarism Detection Report",
            styles["Title"]
        )
    )

    content.append(Spacer(1, 15))

    content.append(
        Paragraph(
            f"Document: {escape(str(document_name))}",
            styles["Normal"]
        )
    )

    content.append(
        Paragraph(
            f"Report Date: {datetime.now().strftime('%d-%m-%Y %H:%M')}",
            styles["Normal"]
        )
    )

    content.append(Spacer(1, 20))

    content.append(
        Paragraph(
            "Detection Summary",
            styles["Heading2"]
        )
    )

    highest_score = max(
        [item["similarity_score"] for item in matches],
        default=0
    )

    summary_data = [
        ["Metric", "Result"],
        ["Potential matches", str(len(matches))],
        [
            "Highest similarity",
            f"{highest_score * 100:.2f}%"
        ]
    ]

    summary_table = Table(
        summary_data,
        colWidths=[3 * inch, 2 * inch]
    )

    summary_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 8),
        ])
    )

    content.append(summary_table)
    content.append(Spacer(1, 20))

    content.append(
        Paragraph(
            "Matched Reference Documents",
            styles["Heading2"]
        )
    )

    if not matches:

        content.append(
            Paragraph(
                "No matches were found above the selected threshold.",
                styles["Normal"]
            )
        )

    for number, result in enumerate(matches, start=1):

        content.append(
            Paragraph(
                f"Match {number}: "
                f"{escape(str(result['source_file']))}",
                styles["Heading3"]
            )
        )

        content.append(
            Paragraph(
                f"Similarity: "
                f"{result['similarity_score'] * 100:.2f}%",
                styles["Normal"]
            )
        )

        content.append(Spacer(1, 8))

        content.append(
            Paragraph(
                "<b>Uploaded Document Text:</b>",
                styles["Normal"]
            )
        )

        content.append(
            Paragraph(
                escape(str(result["suspicious_text"])),
                styles["BodyText"]
            )
        )

        content.append(Spacer(1, 8))

        content.append(
            Paragraph(
                "<b>Reference Document Text:</b>",
                styles["Normal"]
            )
        )

        content.append(
            Paragraph(
                escape(str(result["matched_source_text"])),
                styles["BodyText"]
            )
        )

        content.append(Spacer(1, 20))

    content.append(
        Paragraph(
            "Note: Similarity scores indicate potentially similar "
            "content. They do not automatically prove plagiarism. "
            "Please review the matched passages.",
            styles["Italic"]
        )
    )

    pdf.build(content)

    buffer.seek(0)

    return buffer.getvalue()
