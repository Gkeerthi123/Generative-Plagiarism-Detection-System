
import numpy as np

from src.embedding import generate_embeddings
from src.semantic_search import create_faiss_index


def split_into_chunks(text, words_per_chunk=100, overlap=20):
    """Split text into overlapping chunks."""

    words = text.split()
    chunks = []
    start = 0
    step = words_per_chunk - overlap

    if step <= 0:
        raise ValueError("Overlap must be smaller than chunk size.")

    while start < len(words):
        end = start + words_per_chunk
        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        start += step

    return chunks


def build_source_index(source_documents, model):
    """Index all chunks from every reference document."""

    source_chunks = []

    for document in source_documents:

        chunks = split_into_chunks(document["text"])

        for chunk in chunks:
            source_chunks.append({
                "file_name": document["file_name"],
                "source_type": document.get(
                    "source_type",
                    "PAN-PC-11 Dataset"
                ),
                "text": chunk
            })

    if not source_chunks:
        raise ValueError("No source document chunks found.")

    texts = [chunk["text"] for chunk in source_chunks]

    embeddings = generate_embeddings(texts, model)

    index = create_faiss_index(embeddings)

    return index, source_chunks


def find_top_matches(
    suspicious_document,
    source_index,
    source_chunks,
    model,
    top_k=5
):
    """Find the closest matching reference passages."""

    suspicious_chunks = split_into_chunks(
        suspicious_document["text"]
    )

    if not suspicious_chunks or not source_chunks:
        return []

    suspicious_embeddings = generate_embeddings(
        suspicious_chunks,
        model
    )

    best_matches = {}

    search_k = min(5, len(source_chunks))

    for chunk, embedding in zip(
        suspicious_chunks,
        suspicious_embeddings
    ):

        query = np.asarray(
            [embedding],
            dtype="float32"
        )

        scores, indices = source_index.search(
            query,
            search_k
        )

        for score, idx in zip(scores[0], indices[0]):

            if idx < 0 or idx >= len(source_chunks):
                continue

            source = source_chunks[idx]

            result = {
                "suspicious_file": suspicious_document["file_name"],
                "source_file": source["file_name"],
                "source_type": source.get(
                    "source_type",
                    "Custom Reference Document"
                ),
                "similarity_score": float(score),
                "suspicious_text": chunk,
                "matched_source_text": source["text"]
            }

            key = (
                source["file_name"],
                chunk
            )

            if (
                key not in best_matches
                or result["similarity_score"]
                > best_matches[key]["similarity_score"]
            ):
                best_matches[key] = result

    sorted_matches = sorted(
        best_matches.values(),
        key=lambda item: item["similarity_score"],
        reverse=True
    )

    return sorted_matches[:top_k]


def detect_plagiarism(
    suspicious_document,
    source_index,
    source_chunks,
    model,
    threshold=0.70
):
    """Return matches that meet the selected threshold."""

    all_matches = find_top_matches(
        suspicious_document,
        source_index,
        source_chunks,
        model,
        top_k=len(suspicious_document["text"].split()) + 5
    )

    return [
        result
        for result in all_matches
        if result["similarity_score"] >= threshold
    ]
