from sentence_transformers import SentenceTransformer
import os

os.environ.pop("SSL_CERT_FILE", None)

model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_texts(text):
    return model.encode(text)

