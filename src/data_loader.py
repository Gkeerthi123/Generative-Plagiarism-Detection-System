
import os


def load_text_document(file_path):
    """
    Read the actual text from a TXT document.
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as file:
            return file.read().strip()

    except Exception as error:
        print(f"Error reading {file_path}: {error}")
        return ""


def load_documents(folder_path):
    """
    Load all TXT documents from a folder and its subfolders.
    """
    documents = []

    for root, directories, files in os.walk(folder_path):
        for file in files:

            if file.lower().endswith(".txt"):
                file_path = os.path.join(root, file)

                text = load_text_document(file_path)

                if text:
                    documents.append({
                        "file_name": file,
                        "file_path": file_path,
                        "text": text
                    })

    return documents
