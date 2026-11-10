
import faiss
import numpy as np
import os
import pickle


def create_faiss_index(embeddings):
    embeddings = np.asarray(embeddings, dtype="float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index


def save_faiss_index(index, source_chunks, index_path, chunks_path):
    os.makedirs(os.path.dirname(index_path), exist_ok=True)

    faiss.write_index(index, index_path)

    with open(chunks_path, "wb") as file:
        pickle.dump(source_chunks, file)

    print("FAISS index and source chunks saved successfully.")


def load_faiss_index(index_path, chunks_path):
    if not os.path.exists(index_path) or not os.path.exists(chunks_path):
        return None, None

    index = faiss.read_index(index_path)

    with open(chunks_path, "rb") as file:
        source_chunks = pickle.load(file)

    print("FAISS index and source chunks loaded successfully.")

    return index, source_chunks


def search_similar_documents(query_embedding, index, documents, top_k=5):
    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    ).reshape(1, -1)

    scores, indices = index.search(query_embedding, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        results.append({
            "file_name": documents[idx]["file_name"],
            "similarity_score": float(score),
            "text": documents[idx]["text"]
        })

    return results
