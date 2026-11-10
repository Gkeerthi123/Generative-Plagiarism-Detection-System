
from sentence_transformers import SentenceTransformer


def load_embedding_model():
    """
    Load the pre-trained Sentence Transformer model.
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    model.max_seq_length = 128
    return model


def generate_embeddings(texts, model):
    """
    Convert document texts into numerical embeddings.
    """

    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    return embeddings
