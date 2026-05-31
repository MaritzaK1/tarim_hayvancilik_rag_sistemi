import re
import spacy
from typing import List, Dict

_nlp = None
TARLA_PATTERNS = {
    "PESTICIDE": re.compile(
        r'\b(?:fungusit|insektisit|herbisit|pestisit|nematisit|akarisit|'
        r'gliosat|sulfur|bakır|mankozeb|kloroprifos|imidakloprit)\w*\b',
        re.IGNORECASE
    ),
    "CROP": re.compile(
        r'\b(?:buğday|arpa|mısır|çeltik|pirinç|soya|ayçiçeği|pamuk|'
        r'tütün|domates|patates|soğan|doğan|biber|salatalık|kavun|'
        r'karpuz|elma|armut|kiraz|vişne|şeftali|erik|üzüm|incir|'
        r'zeytin|fındık|çay|portakal|limon|mandalina|greyfurt)\w*\b',
        re.IGNORECASE
    ),
    "DISEASE": re.compile(
        r'\b(?:yanıklık|külleme|mildiyö|pas hastalığı|çürüklük|antraknoz|'
        r'septoria|fusarium|botrytis|alternaria|sclerotinia)\w*\b',
        re.IGNORECASE
    ),
    "ORG": re.compile(
        r'\b(?:tarım bakanlığı|ziraat odası|köy hizmetleri|'
        r'tarım kredi|ziraat bankası|EPPO|FAO|TUIK|TRT|TOBB)\b',
        re.IGNORECASE
    )
}

def _get_nlp():
    global _nlp
    if _nlp is not None:
        return _nlp
    
    try:
        _nlp = spacy.load("xx_ent_wiki_sm")
        print("[NER] Yüklenen model: xx_ent_wiki_sm")
        return _nlp
    except OSError:
        print("[NER] Uyarı: xx_ent_wiki_sm modeli bulunamadı. Sadece Regex kullanılacak.")
        _nlp = spacy.blank("xx")
        return _nlp

def extract_entities_regex(text: str) -> List[Dict]:
    entities = []
    seen = set()
    for label, pattern in TARLA_PATTERNS.items():
        for m in pattern.finditer(text):
            key = m.group().lower()
            if key not in seen:
                seen.add(key)
                entities.append({
                    "text": m.group(),
                    "label": label,
                    "start": m.start(),
                    "end": m.end()
                })
    return entities

def extract_entities_spacy(text: str) -> List[Dict]:
    nlp = _get_nlp()
    
    if len(text) > 50000:
        text = text[:50000]
    
    doc = nlp(text)
    entities = []
    seen = set()
    
    for ent in doc.ents:
        key = ent.text.strip().lower()
        if key not in seen and len(ent.text.strip()) > 1:
            seen.add(key)
            entities.append({
                "text": ent.text.strip(),
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })
    
    return entities

def extract_entities(text: str) -> List[Dict]:
    regex_entities = extract_entities_regex(text)
    spacy_entities = extract_entities_spacy(text)
    all_entities = regex_entities[:]
    seen = {e["text"].lower() for e in regex_entities}
    
    for e in spacy_entities:
        if e["text"].lower() not in seen:
            seen.add(e["text"].lower())
            all_entities.append(e)
            
    return all_entities

def extract_entities_from_chunks(chunks: List[dict]) -> List[dict]:
    enriched = []
    for chunk in chunks:
        entities = extract_entities(chunk["text"])
        enriched_chunk = dict(chunk)
        enriched_chunk["entities"] = entities
        enriched.append(enriched_chunk)
    return enriched
