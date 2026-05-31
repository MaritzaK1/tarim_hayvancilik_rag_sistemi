# pyrefly: ignore [missing-import]
import chromadb
import json
from backend.embedder import embed_text

client = chromadb.HttpClient(host="localhost", port=8000)
collection = client.get_or_create_collection(name="tarim_rag", metadata={"hnsw:space": "cosine"})

query = "yemeklik baklagil hastalık ve zararları"
emb = embed_text(query)

results = collection.query(
    query_embeddings=[emb],
    n_results=10,
    include=["documents", "metadatas", "distances"]
)

print("=== VEKTÖREL ARAMA SONUÇLARI ===")
for i, doc in enumerate(results["documents"][0]):
    meta = results["metadatas"][0][i]
    dist = results["distances"][0][i]
    print(f"\n{i+1}. [{meta['filename']} Sayfa:{meta['page_num']}] mesafe:{dist:.4f}")
    print(f"   {doc[:120]}...")

print("\n\n=== ANAHTAR KELİME ARAMA (baklagil) ===")
kw_results = collection.get(
    where_document={"$contains": "baklagil"},
    limit=5,
    include=["documents", "metadatas"]
)
if kw_results["documents"]:
    for i, doc in enumerate(kw_results["documents"]):
        meta = kw_results["metadatas"][i]
        print(f"\n{i+1}. [{meta['filename']} Sayfa:{meta['page_num']}]")
        print(f"   {doc[:120]}...")
else:
    print("Hiç sonuç yok!")
