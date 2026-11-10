from io import BytesIO
from PyPDF2 import PdfReader
from docx import Document


def extract_text(file_name, file_bytes):

    try:

        if file_name.lower().endswith(".pdf"):

            pdf = PdfReader(BytesIO(file_bytes))

            return " ".join(
                page.extract_text() or ""
                for page in pdf.pages
            )

        elif file_name.lower().endswith(".docx"):

            doc = Document(BytesIO(file_bytes))

            return "\n".join(
                paragraph.text
                for paragraph in doc.paragraphs
            )

        elif file_name.lower().endswith(".txt"):

            return file_bytes.decode(
                "utf-8",
                errors="replace"
            )

        else:
            return ""

    except Exception as error:

        print(f"Error reading {file_name}: {error}")
        return ""