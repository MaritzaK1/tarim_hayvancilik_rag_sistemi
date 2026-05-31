import os
import shutil
import asyncio
import json
from dotenv import load_dotenv
load_dotenv("rag.env")

from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.pdf_processor import extract_text_from_pdf
from backend.chunker import chunk_pages
from backend.ner import extract_entities_from_chunks
from backend.wikidata import enrich_chunks_with_wikidata
from backend.embedder import embed_chunks
from backend.chroma_client import add_chunks, get_collection_info, delete_by_filename, hybrid_search
from backend.rag import rag_query, embed_text, query_similar

app = FastAPI(title="Tarım RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

pipeline_status = {}


class QueryRequest(BaseModel):
    question: str
    n_results: Optional[int] = 5
    filename_filter: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/collection-info")
def collection_info():
    return get_collection_info()


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Sadece PDF dosyaları kabul edilir.")

    filename = file.filename
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    async def event_stream():
        try:
            yield _sse("step", {"step": "extract", "status": "running", "message": "PDF okunuyor..."})
            await asyncio.sleep(0.1)

            loop = asyncio.get_event_loop()
            pdf_data = await loop.run_in_executor(None, extract_text_from_pdf, file_path)
            total_pages = pdf_data["total_pages"]
            yield _sse("step", {
                "step": "extract",
                "status": "done",
                "message": f"{total_pages} sayfa okundu",
                "data": {"total_pages": total_pages}
            })

            yield _sse("step", {"step": "chunk", "status": "running", "message": "Metin parçalanıyor..."})
            await asyncio.sleep(0.1)

            chunks = await loop.run_in_executor(None, chunk_pages, pdf_data["pages"])
            yield _sse("step", {
                "step": "chunk",
                "status": "done",
                "message": f"{len(chunks)} chunk oluşturuldu",
                "data": {
                    "chunk_count": len(chunks),
                    "sample_chunks": [c["text"][:150] for c in chunks[:3]]
                }
            })

            yield _sse("step", {"step": "ner", "status": "running", "message": "Entity'ler tespit ediliyor..."})
            await asyncio.sleep(0.1)

            chunks_with_ner = await loop.run_in_executor(None, extract_entities_from_chunks, chunks)
            all_entities = []
            for c in chunks_with_ner:
                all_entities.extend(c.get("entities", []))

            seen = set()
            unique_entities = []
            for e in all_entities:
                k = e["text"].lower()
                if k not in seen:
                    seen.add(k)
                    unique_entities.append(e)

            yield _sse("step", {
                "step": "ner",
                "status": "done",
                "message": f"{len(unique_entities)} entity tespit edildi",
                "data": {"entities": unique_entities[:50]}
            })

            yield _sse("step", {"step": "wikidata", "status": "running", "message": "Wikidata'dan bilgi çekiliyor..."})
            await asyncio.sleep(0.1)

            chunks_with_wiki = await loop.run_in_executor(None, enrich_chunks_with_wikidata, chunks_with_ner)

            wiki_count = sum(
                1 for c in chunks_with_wiki
                for e in c.get("entities", [])
                if e.get("wikidata")
            )
            yield _sse("step", {
                "step": "wikidata",
                "status": "done",
                "message": f"{wiki_count} entity Wikidata ile zenginleştirildi",
                "data": {"wiki_enriched_count": wiki_count}
            })

            yield _sse("step", {"step": "embed", "status": "running", "message": f"{len(chunks_with_wiki)} chunk embed ediliyor..."})
            await asyncio.sleep(0.1)

            chunks_embedded = await loop.run_in_executor(None, embed_chunks, chunks_with_wiki)
            yield _sse("step", {
                "step": "embed",
                "status": "done",
                "message": f"{len(chunks_embedded)} embedding oluşturuldu"
            })

            yield _sse("step", {"step": "vectordb", "status": "running", "message": "ChromaDB'ye kaydediliyor..."})
            await asyncio.sleep(0.1)

            count = await loop.run_in_executor(None, add_chunks, chunks_embedded, filename)
            collection = get_collection_info()

            yield _sse("step", {
                "step": "vectordb",
                "status": "done",
                "message": f"{count} chunk ChromaDB'ye eklendi",
                "data": {"added": count, "total_in_db": collection.get("count", 0)}
            })

            yield _sse("done", {
                "filename": filename,
                "total_chunks": len(chunks_embedded),
                "total_entities": len(unique_entities),
                "wiki_enriched": wiki_count
            })

        except Exception as e:
            yield _sse("error", {"message": str(e)})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/query")
def query(req: QueryRequest):
    try:
        question_embedding = embed_text(req.question)
        relevant_chunks = hybrid_search(req.question, question_embedding, req.n_results or 10, req.filename_filter)

        if not relevant_chunks:
            return {"answer": "Veritabanında ilgili belge bulunamadı.", "sources": []}

        context_parts = []
        for i, chunk in enumerate(relevant_chunks):
            source = f"[Kaynak {i+1}: {chunk['filename']}, Sayfa {chunk['page_num']}]"
            entities_text = ""
            if chunk.get("entities"):
                ent_names = [e["text"] for e in chunk["entities"][:5]]
                entities_text = f"\nİlgili varlıklar: {', '.join(ent_names)}"
            context_parts.append(f"{source}{entities_text}\n{chunk['text']}")
        context = "\n\n---\n\n".join(context_parts)

        from openai import OpenAI
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        )

        system_prompt = """Sen tarım alanında uzman bir yapay zeka asistanısın.

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

        response = client.chat.completions.create(
            model=os.environ.get("LLM_MODEL", "google/gemma-3-27b-it:free"),
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": f"Aşağıda tarım belgelerinden alınan bağlam bilgileri ve bir kullanıcı sorusu var.\n\nBAĞLAM BİLGİLERİ:\n{context}\n\nKULLANICI SORUSU: {req.question}\n\nYukarıdaki bağlam bilgilerini kullanarak bu soruya kapsamlı bir Türkçe yanıt ver. Bağlamda ilgili bilgi varsa mutlaka yanıtla. Kaynak dosya adı ve sayfa numarasını belirt."
                }
            ],
            temperature=0.3,
            max_tokens=2048
        )
        answer = response.choices[0].message.content

        sources = [
            {
                "filename": c["filename"],
                "page_num": c["page_num"],
                "text_preview": c["text"][:200] + "..." if len(c["text"]) > 200 else c["text"],
                "distance": round(c["distance"], 4),
                "entities": c.get("entities", [])
            }
            for c in relevant_chunks
        ]

        return {"answer": answer, "sources": sources}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query-stream")
def query_stream(req: QueryRequest):
    def event_generator():
        try:
            for chunk in rag_query(req.question, req.n_results, req.filename_filter):
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.delete("/document/{filename}")
def delete_document(filename: str):
    delete_by_filename(filename)
    
    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        
    return {"message": f"{filename} başarıyla silindi"}


@app.get("/documents")
def get_documents():
    from backend.chroma_client import get_unique_filenames
    filenames = get_unique_filenames()
    return {"documents": filenames}


@app.get("/chromadb-chunks")
def get_chromadb_chunks(limit: int = 100, offset: int = 0):
    from backend.chroma_client import get_all_chunks
    chunks = get_all_chunks(limit=limit, offset=offset)
    return {"chunks": chunks}



def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


if os.path.exists("frontend"):
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
