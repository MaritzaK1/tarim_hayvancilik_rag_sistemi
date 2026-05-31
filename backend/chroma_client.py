import chromadb
from typing import List, Dict
import json

CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
COLLECTION_NAME = "tarim_rag"

_client = None
_collection = None


def get_client():
    global _client
    if _client is None:
        _client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    return _client


def get_collection():
    global _collection
    if _collection is None:
        client = get_client()
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
    return _collection


def add_chunks(chunks_with_embeddings: List[dict], filename: str):
    collection = get_collection()

    ids = []
    embeddings = []
    documents = []
    metadatas = []

    for chunk in chunks_with_embeddings:
        chunk_id = f"{filename}__chunk_{chunk['chunk_id']}"
        ids.append(chunk_id)
        embeddings.append(chunk["embedding"])
        documents.append(chunk["text"])

        # Entity'leri JSON string olarak sakla
        entities_info = []
        for ent in chunk.get("entities", []):
            ent_data = {"text": ent["text"], "label": ent["label"]}
            wiki = ent.get("wikidata")
            if wiki:
                ent_data["wiki_description"] = wiki.get("description", "")
                ent_data["wiki_id"] = wiki.get("wikidata_id", "")
            entities_info.append(ent_data)

        metadatas.append({
            "filename": filename,
            "page_num": chunk["page_num"],
            "chunk_index": chunk["chunk_index"],
            "entities": json.dumps(entities_info, ensure_ascii=False)
        })

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )
    return len(ids)


def query_similar(query_embedding: List[float], n_results: int = 5, filename_filter: str = None) -> List[Dict]:
    collection = get_collection()

    where = None
    if filename_filter:
        where = {"filename": filename_filter}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    seen_ids = set()
    if results["documents"] and results["documents"][0]:
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            chunk_id = results["ids"][0][i]
            seen_ids.add(chunk_id)
            entities = []
            try:
                entities = json.loads(meta.get("entities", "[]"))
            except Exception:
                pass
            chunks.append({
                "text": doc,
                "filename": meta.get("filename", ""),
                "page_num": meta.get("page_num", 0),
                "chunk_index": meta.get("chunk_index", 0),
                "distance": distance,
                "entities": entities
            })
    return chunks


def _turkish_normalize(text):
    tr_map = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
    return text.translate(tr_map).lower()


def _turkish_stems(word):
    word = word.lower()
    stems = {word, _turkish_normalize(word)}

    suffixes = [
        "ları", "leri", "ları", "lığı", "liği", "lugu", "lüğü",
        "lık", "lik", "luk", "lük",
        "ının", "inin", "unun", "ünün",
        "ında", "inde", "unda", "ünde",
        "ıyla", "iyle",
        "dan", "den", "tan", "ten",
        "lar", "ler",
        "ın", "in", "un", "ün",
        "da", "de", "ta", "te",
        "ya", "ye",
        "ı", "i", "u", "ü",
    ]

    for s in suffixes:
        if word.endswith(s) and len(word) - len(s) >= 3:
            root = word[:-len(s)]
            stems.add(root)
            stems.add(_turkish_normalize(root))

    normalized = _turkish_normalize(word)
    for s in suffixes:
        ns = _turkish_normalize(s)
        if normalized.endswith(ns) and len(normalized) - len(ns) >= 3:
            root = normalized[:-len(ns)]
            stems.add(root)

    return stems


def hybrid_search(query_text: str, query_embedding: List[float], n_results: int = 10, filename_filter: str = None) -> List[Dict]:
    collection = get_collection()

    where = None
    if filename_filter:
        where = {"filename": filename_filter}

    vector_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    seen_ids = set()

    if vector_results["documents"] and vector_results["documents"][0]:
        for i, doc in enumerate(vector_results["documents"][0]):
            meta = vector_results["metadatas"][0][i]
            distance = vector_results["distances"][0][i]
            chunk_id = vector_results["ids"][0][i]
            seen_ids.add(chunk_id)
            entities = []
            try:
                entities = json.loads(meta.get("entities", "[]"))
            except Exception:
                pass
            chunks.append({
                "text": doc,
                "filename": meta.get("filename", ""),
                "page_num": meta.get("page_num", 0),
                "chunk_index": meta.get("chunk_index", 0),
                "distance": distance,
                "entities": entities,
                "search_type": "vector"
            })

    import re
    raw_words = re.findall(r'[a-zA-ZçğıöşüÇĞİÖŞÜ]{3,}', query_text.lower())
    stop_words = {"bir", "ile", "için", "olan", "nasıl", "nedir", "hakkında", "var", "yok", "bana", "anlat", "ver",
                  "neler", "nelerdir", "neden", "hangi", "kaç", "mıdır", "midir", "olan", "olarak"}
    keywords = [k for k in raw_words if k not in stop_words]

    search_terms = set()
    for kw in keywords:
        search_terms.update(_turkish_stems(kw))
    search_terms = [t for t in search_terms if len(t) >= 3]

    if not filename_filter:
        try:
            all_filenames = get_unique_filenames()
            matched_files = []
            for fname in all_filenames:
                fname_normalized = _turkish_normalize(fname)
                for term in search_terms:
                    if term in fname_normalized:
                        matched_files.append(fname)
                        break

            for matched_file in matched_files[:3]:
                try:
                    file_results = collection.get(
                        where={"filename": matched_file},
                        limit=10,
                        include=["documents", "metadatas"]
                    )
                    if file_results and file_results["documents"]:
                        for i, doc in enumerate(file_results["documents"]):
                            chunk_id = file_results["ids"][i]
                            if chunk_id in seen_ids:
                                continue
                            seen_ids.add(chunk_id)
                            meta = file_results["metadatas"][i]
                            entities = []
                            try:
                                entities = json.loads(meta.get("entities", "[]"))
                            except Exception:
                                pass

                            doc_normalized = _turkish_normalize(doc)
                            relevance_score = sum(1 for t in search_terms if t in doc_normalized)

                            chunks.append({
                                "text": doc,
                                "filename": meta.get("filename", ""),
                                "page_num": meta.get("page_num", 0),
                                "chunk_index": meta.get("chunk_index", 0),
                                "distance": max(0.02, 0.3 - (relevance_score * 0.05)),
                                "entities": entities,
                                "search_type": "filename"
                            })
                except Exception:
                    continue
        except Exception:
            pass

    for term in search_terms[:8]:
        try:
            kw_results = collection.get(
                where_document={"$contains": term},
                where=where,
                limit=10,
                include=["documents", "metadatas"]
            )
            if kw_results and kw_results["documents"]:
                for i, doc in enumerate(kw_results["documents"]):
                    chunk_id = kw_results["ids"][i]
                    if chunk_id in seen_ids:
                        continue
                    seen_ids.add(chunk_id)
                    meta = kw_results["metadatas"][i]
                    entities = []
                    try:
                        entities = json.loads(meta.get("entities", "[]"))
                    except Exception:
                        pass

                    doc_lower = doc.lower()
                    doc_normalized = _turkish_normalize(doc)
                    relevance_score = 0
                    for kw in keywords:
                        if kw in doc_lower:
                            relevance_score += 2
                        elif _turkish_normalize(kw) in doc_normalized:
                            relevance_score += 1.5
                        else:
                            for stem in _turkish_stems(kw):
                                if stem in doc_normalized:
                                    relevance_score += 1
                                    break

                    chunks.append({
                        "text": doc,
                        "filename": meta.get("filename", ""),
                        "page_num": meta.get("page_num", 0),
                        "chunk_index": meta.get("chunk_index", 0),
                        "distance": max(0.05, 0.5 - (relevance_score * 0.08)),
                        "entities": entities,
                        "search_type": "keyword"
                    })
        except Exception:
            continue

    chunks.sort(key=lambda x: x.get("distance", 1.0))
    return chunks[:n_results]


def get_collection_info() -> Dict:
    try:
        collection = get_collection()
        count = collection.count()
        return {
            "name": COLLECTION_NAME,
            "count": count,
            "status": "connected"
        }
    except Exception as e:
        return {"name": COLLECTION_NAME, "count": 0, "status": f"error: {str(e)}"}


def delete_by_filename(filename: str):
    collection = get_collection()
    collection.delete(where={"filename": filename})


def get_all_chunks(limit: int = 100, offset: int = 0) -> List[Dict]:
    collection = get_collection()
    results = collection.get(limit=limit, offset=offset, include=["metadatas", "documents", "embeddings"])
    chunks = []
    if results["documents"]:
        for i, doc in enumerate(results["documents"]):
            meta = results["metadatas"][i] if results["metadatas"] else {}
            chunk_id = results["ids"][i]
            
            entities = []
            if "entities" in meta:
                import json
                try:
                    entities = json.loads(meta["entities"])
                except:
                    pass

            emb_list = results.get("embeddings")
            embedding = emb_list[i] if emb_list is not None and len(emb_list) > i else None
            if embedding is not None:
                if hasattr(embedding, "tolist"):
                    embedding = embedding.tolist()
                else:
                    embedding = list(embedding)

            chunks.append({
                "id": chunk_id,
                "text": doc,
                "filename": meta.get("filename", ""),
                "page_num": meta.get("page_num", 0),
                "chunk_index": meta.get("chunk_index", 0),
                "entities": entities,
                "embedding": embedding
            })
    return chunks


def get_unique_filenames() -> List[str]:
    collection = get_collection()
    results = {"metadatas": []}
    offset = 0
    limit = 5000
    while True:
        batch = collection.get(include=["metadatas"], limit=limit, offset=offset)
        if not batch or not batch.get("metadatas"):
            break
        results["metadatas"].extend(batch["metadatas"])
        offset += limit
    filenames = set()
    if results and results.get("metadatas"):
        for meta in results["metadatas"]:
            if meta and "filename" in meta:
                filenames.add(meta["filename"])
    return list(filenames)

