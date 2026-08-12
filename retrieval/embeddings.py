from sentence_transformers import SentenceTransformer
import torch

MODEL_NAME = "all-miniLM-L6-v2"

device = "cuda" if torch.cuda.is_available() else "cpu"

embedding_model = SentenceTransformer(
    MODEL_NAME,
    device=device,
    trust_remote_code=True,
)

def generate_embeddings(texts):
    return embedding_model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )