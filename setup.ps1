Write-Host "=== Tarım RAG Sistemi Kurulum ===" -ForegroundColor Green

Write-Host "`n[1/3] Python bağımlılıkları yükleniyor..." -ForegroundColor Cyan
pip install -r requirements.txt

Write-Host "`n[2/3] spaCy çok dilli (Türkçe) modeli yükleniyor..." -ForegroundColor Cyan
python -m spacy download xx_ent_wiki_sm

Write-Host "`n[3/3] Ollama modelleri kontrol ediliyor..." -ForegroundColor Cyan
ollama list

Write-Host "`n=== Kurulum Tamamlandı ===" -ForegroundColor Green
Write-Host "Backend'i başlatmak için: uvicorn main:app --port 8001 --reload" -ForegroundColor Yellow
