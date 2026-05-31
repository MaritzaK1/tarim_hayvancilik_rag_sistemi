import os
import ollama
from dotenv import load_dotenv
from typing import List, Dict, Generator
from backend.embedder import embed_text
from backend.chroma_client import query_similar, hybrid_search

load_dotenv("rag.env")

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
LLM_MODEL = os.environ.get("LLM_MODEL", "openrouter/auto")

SYSTEM_PROMPT = """Sen tarım alanında uzman bir yapay zeka asistanısın.

KURALLAR:
1. Sana verilen BAĞLAM bilgilerini kullanarak soruyu yanıtla.
2. Bağlamda soruyla DOĞRUDAN veya DOLAYLI olarak ilgili herhangi bir bilgi varsa (kısmen ilgili olsa bile), bu bilgileri DERLEYEREK MUTLAKA bir yanıt oluştur. Eksik olan kısımları belirt ama elindeki bilgiyi paylaş.
3. Genel bir soru sorulduysa (örn: "domates hastalıkları nelerdir", "bitki hastalıkları nelerdir") bağlamda geçen TÜM hastalık/zararlı/belirti/mücadele bilgilerini topla, isimlerini listele ve özetle.
4. Her zaman Türkçe yanıt ver.
5. Yanıtında hangi kaynaktan (dosya adı ve sayfa numarası) bilgi aldığını belirt.
6. "Bu konuda belgelerde bilgi bulamadım." cümlesini SADECE ve SADECE bağlamdaki tüm chunk'lar konuyla TAMAMEN alakasız olduğunda kullan. Sana kaynaklar sunulmuşsa, içlerinde sorunun konusuyla (anahtar kelime, varlık, başlık) en ufak bir bağlantı bile varsa, bilgi bulamadın deme — eldeki bilgiyle yanıt üret.
7. Yanıtını açık, anlaşılır ve maddeler halinde ver. Hastalık/zararlı isimlerini kalın yaz.
8. Bağlamdaki teknik terimleri ve verileri olduğu gibi aktar.
9. Eğer bağlamdaki bilgi sınırlıysa, "Belgelerde tespit edebildiğim kadarıyla şunlar yer alıyor: ..." gibi bir giriş yap ve elindeki bilgiyi paylaş."""


def rag_query(
    question: str,
    n_results: int = 15,
    filename_filter: str = None
) -> Generator:
    question_embedding = embed_text(question)
    relevant_chunks = hybrid_search(question, question_embedding, n_results, filename_filter)

    if not relevant_chunks:
        yield {"type": "error", "content": "Veritabanında ilgili belge bulunamadı."}
        return

    context_parts = []
    for i, chunk in enumerate(relevant_chunks):
        source = f"[Kaynak {i+1}: {chunk['filename']}, Sayfa {chunk['page_num']}]"
        entities_text = ""
        if chunk.get("entities"):
            ent_names = [e["text"] for e in chunk["entities"][:5]]
            entities_text = f"\nİlgili varlıklar: {', '.join(ent_names)}"
        context_parts.append(f"{source}{entities_text}\n{chunk['text']}")

    context = "\n\n---\n\n".join(context_parts)
    yield {
        "type": "sources",
        "content": [
            {
                "filename": c["filename"],
                "page_num": c["page_num"],
                "text_preview": c["text"][:200] + "..." if len(c["text"]) > 200 else c["text"],
                "distance": round(c["distance"], 4),
                "entities": c.get("entities", [])
            }
            for c in relevant_chunks
        ]
    }
    prompt = f"""Aşağıda tarım belgelerinden alınan bağlam bilgileri ve bir kullanıcı sorusu var.

BAĞLAM BİLGİLERİ:
{context}

KULLANICI SORUSU: {question}

Yukarıdaki bağlam bilgilerini kullanarak bu soruya kapsamlı bir Türkçe yanıt ver. Bağlamda ilgili bilgi varsa mutlaka yanıtla. Kaynak dosya adı ve sayfa numarasını belirt."""

    from openai import OpenAI
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )
    
    stream = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        stream=True,
        temperature=0.3,
        max_tokens=2048
    )

    for part in stream:
        if part.choices and part.choices[0].delta.content:
            yield {"type": "token", "content": part.choices[0].delta.content}

    yield {"type": "done", "content": ""}


def rag_query_sync(question: str, n_results: int = 5, filename_filter: str = None) -> Dict:
    sources = []
    answer_parts = []

    for chunk in rag_query(question, n_results, filename_filter):
        if chunk["type"] == "sources":
            sources = chunk["content"]
        elif chunk["type"] == "token":
            answer_parts.append(chunk["content"])
        elif chunk["type"] == "error":
            return {"answer": chunk["content"], "sources": []}

    return {
        "answer": "".join(answer_parts),
        "sources": sources
    }
