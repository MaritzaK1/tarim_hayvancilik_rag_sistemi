from typing import List


def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    if not text or not text.strip():
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(para) > chunk_size:
            sentences = _split_sentences(para)
            for sent in sentences:
                if len(current_chunk) + len(sent) + 1 <= chunk_size:
                    current_chunk = (current_chunk + " " + sent).strip()
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    overlap_text = _get_overlap(current_chunk, chunk_overlap)
                    current_chunk = (overlap_text + " " + sent).strip()
        else:
            if len(current_chunk) + len(para) + 2 <= chunk_size:
                current_chunk = (current_chunk + "\n" + para).strip()
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                overlap_text = _get_overlap(current_chunk, chunk_overlap)
                current_chunk = (overlap_text + "\n" + para).strip()

    if current_chunk:
        chunks.append(current_chunk)

    return [c for c in chunks if len(c.strip()) > 20]


def _split_sentences(text: str) -> List[str]:
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def _get_overlap(text: str, overlap_size: int) -> str:
    if not text or overlap_size <= 0:
        return ""
    words = text.split()
    overlap_words = []
    total = 0
    for word in reversed(words):
        if total + len(word) > overlap_size:
            break
        overlap_words.insert(0, word)
        total += len(word) + 1
    return " ".join(overlap_words)


def chunk_pages(pages: list, chunk_size: int = 500, chunk_overlap: int = 50) -> List[dict]:
    all_chunks = []
    chunk_id = 0

    for page in pages:
        page_chunks = split_text(page["text"], chunk_size, chunk_overlap)
        for idx, chunk_text in enumerate(page_chunks):
            all_chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "page_num": page["page_num"],
                "chunk_index": idx
            })
            chunk_id += 1

    return all_chunks
