
import os
import json

from src.document_reader import extract_text
from src.text_preprocessing import clean_text
from src.plagiarism_detector import split_into_chunks


def save_reference_files(uploaded_files, folder_path):
    os.makedirs(folder_path, exist_ok=True)

    saved_files = []

    for uploaded_file in uploaded_files:
        file_name = os.path.basename(uploaded_file.name)
        file_path = os.path.join(folder_path, file_name)

        with open(file_path, "wb") as file:
            file.write(uploaded_file.getvalue())

        saved_files.append(file_name)

    # Store only the references selected for the current reference upload
    manifest_path = os.path.join(folder_path, "active_references.json")

    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(saved_files, file, indent=4)

    return saved_files


def load_reference_chunks(folder_path):
    reference_chunks = []

    if not os.path.exists(folder_path):
        return reference_chunks

    manifest_path = os.path.join(folder_path, "active_references.json")

    # Do not automatically load old reference documents
    if not os.path.exists(manifest_path):
        return reference_chunks

    with open(manifest_path, "r", encoding="utf-8") as file:
        active_files = json.load(file)

    for file_name in active_files:
        if not file_name.lower().endswith((".pdf", ".docx", ".txt")):
            continue

        file_path = os.path.join(folder_path, file_name)

        if not os.path.isfile(file_path):
            continue

        with open(file_path, "rb") as file:
            file_bytes = file.read()

        text = extract_text(file_name, file_bytes)
        text = clean_text(text)

        if not text:
            continue

        chunks = split_into_chunks(text)

        for chunk in chunks:
            reference_chunks.append({
                "file_name": file_name,
                "text": chunk
            })

    return reference_chunks
