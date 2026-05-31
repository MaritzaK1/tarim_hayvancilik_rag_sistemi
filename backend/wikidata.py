import requests
import time
from typing import List, Dict, Optional


WIKIDATA_API = "https://www.wikidata.org/w/api.php"
HEADERS = {"User-Agent": "TarimRAG/1.0 (educational project)"}


def search_wikidata(entity_text: str, language: str = "tr") -> Optional[Dict]:
    try:
        params = {
            "action": "wbsearchentities",
            "format": "json",
            "search": entity_text,
            "language": language,
            "limit": 1,
            "type": "item"
        }
        response = requests.get(WIKIDATA_API, params=params, headers=HEADERS, timeout=5)
        data = response.json()
        
        if data.get("search"):
            item = data["search"][0]
            return {
                "entity": entity_text,
                "wikidata_id": item.get("id", ""),
                "label": item.get("label", entity_text),
                "description": item.get("description", ""),
                "url": f"https://www.wikidata.org/wiki/{item.get('id', '')}"
            }
    except Exception as e:
        print(f"[Wikidata] '{entity_text}' aranamadı: {e}")
    return None


from concurrent.futures import ThreadPoolExecutor

def enrich_chunks_with_wikidata(chunks: List[dict]) -> List[dict]:
    important_labels = {"ORG", "PER", "PERSON", "LOC", "GPE", "MISC", "PRODUCT"}
    
    unique_entities = set()
    for chunk in chunks:
        if chunk.get("entities"):
            for ent in chunk["entities"]:
                if ent.get("label") in important_labels:
                    unique_entities.add(ent["text"])
                    
    unique_entities = list(unique_entities)[:50]
    
    global_cache = {}
    
    def fetch_wiki(entity_text):
        res = search_wikidata(entity_text)
        return (entity_text.lower(), res)

    with ThreadPoolExecutor(max_workers=5) as executor:
        results = executor.map(fetch_wiki, unique_entities)
        for (key, res) in results:
            global_cache[key] = res

    enriched_chunks = []
    for chunk in chunks:
        enriched_chunk = dict(chunk)
        if chunk.get("entities"):
            enriched_entities = []
            for ent in chunk["entities"]:
                enriched_ent = dict(ent)
                entity_key = ent["text"].lower()
                if entity_key in global_cache and global_cache[entity_key] is not None:
                    enriched_ent["wikidata"] = global_cache[entity_key]
                enriched_entities.append(enriched_ent)
            enriched_chunk["entities"] = enriched_entities
        enriched_chunks.append(enriched_chunk)
        
    return enriched_chunks
