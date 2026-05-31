from ollama import Client
from typing import List

EMBED_MODEL = "nomic-embed-text:latest"
OLLAMA_CLIENT = Client(host='http://127.0.0.1:11434')

def embed_text(text: str) -> List[float]:
    response = OLLAMA_CLIENT.embeddings(model=EMBED_MODEL, prompt=text)
    return response["embedding"]


def embed_chunks(chunks: List[dict]) -> List[dict]:
    embedded = []
    for i, chunk in enumerate(chunks):
        print(f"[Embedder] Chunk {i+1}/{len(chunks)} embed ediliyor...")
        emb = embed_text(chunk["text"])
        enriched = dict(chunk)
        enriched["embedding"] = emb
        embedded.append(enriched)
    return embedded
