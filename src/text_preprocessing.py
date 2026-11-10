
import re


def clean_text(text):
    """
    Clean document text before AI processing.
    """

    if not isinstance(text, str):
        return ""

    # Convert text to lowercase
    text = text.lower()

    # Remove extra spaces and newline characters
    text = re.sub(r"\s+", " ", text)

    # Remove unwanted special characters
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Remove extra spaces again
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def preprocess_documents(documents):
    """
    Clean text in all loaded documents.
    """

    processed_documents = []

    for document in documents:

        cleaned_text = clean_text(document["text"])

        if cleaned_text:
            processed_documents.append({
                "file_name": document["file_name"],
                "file_path": document["file_path"],
                "text": cleaned_text
            })

    return processed_documents
