# Tarım & Hayvancılık RAG Paneli

Bu proje; tarım, besicilik ve hayvancılık alanlarındaki yerel rehberleri, eylem planlarını ve bilimsel dokümanları işleyerek akıllı bir soru-cevap asistanı sunan, yapay zeka tabanlı bir **Retrieval-Augmented Generation (RAG)** sistemidir.

Proje, kullanıcıların yükledikleri PDF dokümanlarını otomatik olarak parçalara ayırır, anlamsal varlıkları (NER - Named Entity Recognition) çıkarır, bunları Wikidata ile zenginleştirir, vektör veri tabanına kaydeder ve ardından bu veriler üzerinde anlamsal hibrit arama yaparak kullanıcılara doğru ve kaynak odaklı yanıtlar üretir.

---

## 🚀 Temel Özellikler

* **Gelişmiş RAG Akışı (Pipeline)**:
  * **PDF Metin Çıkarma**: PDF dosyalarındaki metinlerin sayfa sayfa okunması.
  * **Chunking (Parçalama)**: Metinlerin anlamsal bütünlüğü korunarak parçalara ayrılması.
  * **Varlık Tanıma (NER)**: spaCy ile bitki, hayvan, hastalık, konum ve organizasyon gibi anlamsal varlıkların tespit edilmesi.
  * **Wikidata Zenginleştirmesi**: Tespit edilen varlıkların Wikidata API'si üzerinden ek açıklamalarla zenginleştirilmesi.
  * **Vektör Ekleme (Embedding)**: Ollama (`nomic-embed-text`) kullanılarak her parçanın vektör karşılığının çıkarılması.
  * **Vektör Depolama**: Vektörlerin ve zengin metin metadata bilgilerinin ChromaDB'de saklanması.
* **Akıllı Sohbet Arayüzü**:
  * Markdown biçimlendirmesi ile kalın yazılar, listeler ve kod blokları desteği.
  * Belirli bir yüklenen belgeye göre filtreleme yapabilen akıllı RAG sorgusu.
  * ChromaDB'den çekilecek benzer parça limitinin (`k` parametresi) dinamik olarak ayarlanabilmesi.
  * Tek tıkla cevapları kopyalama ve sohbet geçmişini temizleme.
  * Çiftçiler ve besiciler için özel olarak tasarlanmış hazır öneri soru kartları.
* **Modern Tarım & Hayvancılık Temalı Dashboard**:
  * Derin orman yeşili, başak sarısı ve zeytin tonlarında tasarlanmış, Plus Jakarta Sans yazı tipiyle güçlendirilmiş profesyonel bir karanlık mod tasarımı.

---

## 🛠️ Sistem Mimarisi

Sistem arka planda şu adımlarla çalışır:

1. **PDF Okuyucu**: PyMuPDF (`fitz`) kütüphanesi kullanılarak yüklenen PDF dosyaları okunur.
2. **Chunker**: Metinler sayfa yapısına sadık kalınarak RAG uyumlu chunk'lara bölünür.
3. **NER Modülü**: Türkçe ve çok dilli spaCy modeli ile metindeki kritik kelimeler (hastalık, böcek, gübre, bölge vb.) varlık olarak etiketlenir.
4. **Wikidata Entegrasyonu**: Tespit edilen varlıklar Wikidata üzerinde sorgulanarak ansiklopedik açıklamalar metadatalara eklenir.
5. **Ollama Embedding**: Yerelde çalışan Ollama istemcisi ile anlamsal vektörler üretilir.
6. **ChromaDB**: Tüm metadata (dosya adı, sayfa no, Wikidata bilgileri) ve vektörler ChromaDB veritabanında indekslenir.
7. **Hibrit Arama (Hybrid Search)**: Arama yapıldığında hem vektör benzerliği hem de Türkçe kelime köklerine (stemming) dayalı anahtar kelime araması birlikte koşturulur.
8. **LLM Cevap Üretici**: Çekilen en alakalı kaynaklar OpenRouter API'si aracılığıyla LLM modeline iletilerek kaynak destekli yanıtlar oluşturulur.

---

## 📦 Kurulum ve Çalıştırma

### 1. Gereksinimler
* Python 3.12 veya üzeri (Önerilen Python 3.14)
* Ollama (nomic-embed-text modeli yüklü olmalıdır)
* ChromaDB Sunucusu (Docker veya lokal yerel kurulum)

### 2. Sanal Ortam Oluşturma ve Bağımlılıkları Yükleme
Proje dizininde bir sanal ortam oluşturun ve bağımlılıkları kurun:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. spaCy Dil Modelini İndirme
Varlık tanıma için gerekli olan spaCy modelini yükleyin:

```bash
python -m spacy download xx_ent_wiki_sm
```

### 4. Gerekli Modelleri Ollama ile İndirme
Yerel embedding üretimi için Ollama uygulamasını açın ve modeli çekin:

```bash
ollama pull nomic-embed-text
```

### 5. Çevresel Değişkenleri Ayarlama
`rag.env` adında bir dosya oluşturarak OpenRouter API anahtarınızı ve kullanmak istediğiniz LLM modelini tanımlayın:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
LLM_MODEL=openai/gpt-3.5-turbo (veya tercih ettiğiniz herhangi bir model)
```

### 6. ChromaDB Sunucusunu Başlatma
ChromaDB sunucusunu Docker ile veya yerel terminalde başlatın:

```bash
docker run -p 8000:8000 chromadb/chroma
```
veya yerel kurulum yaptıysanız:
```bash
chroma run --port 8000
```

### 7. Uygulamayı Başlatma
FastAPI backend sunucusunu ayağa kaldırın:

```bash
uvicorn main:app --port 8001 --reload
```

Uygulama çalıştıktan sonra tarayıcınızdan `http://localhost:8001` adresine giderek arayüze erişebilirsiniz.

---

## 📚 Veri Kaynakları & Kullanılan PDF Dokümanları

Sistemde kullanılan resmî rehberler, kitaplar ve eylem planları T.C. Tarım ve Orman Bakanlığı platformlarından elde edilmiştir:

* [BKU Kaynak Listesi](https://bku.tarimorman.gov.tr/YararlanilacakKaynak/Liste)
* [Bitki Sağlığı Hizmetleri](https://www.tarimorman.gov.tr/Konular/Bitki-Sagligi-Hizmetleri)
* [Tarım Havzaları](https://www.tarimorman.gov.tr/Konular/Bitkisel-Uretim/Tarim-Havzalari)
* [Su Ürünleri](https://www.tarimorman.gov.tr/Konular/Su-Urunleri)
* [Hayvancılık Konuları](https://www.tarimorman.gov.tr/Konular/Hayvancilik)

### Projede İşlenen Doküman Listesi (129 Adet)

| Sıra | Dosya Adı |
|---|---|
| 1 | 2025_Bitki_Sagligi_Uygulama_Programi.pdf |
| 2 | ANA ARI YETİŞTİRİCİLİĞİ.pdf |
| 3 | Avian_Influenza_Acil_Eylem_Plani.pdf |
| 4 | biber_hastalik_ve_zararlilari_ile_mucadele.pdf |
| 5 | bitkikoruma.pdf |
| 6 | biyolojik mücadele kitabı 2018.pdf |
| 7 | biyoteknik mücadele kitabı 2018.pdf |
| 8 | Bombus Ari Kolonisi Uretimi Yapan Isletmeler Nisan 2025.pdf |
| 9 | Buyukbas_Hayvan_Yetistiriciligi.pdf |
| 10 | domates_hastalik_ve_zararlilari_ile_mucadele.pdf |
| 11 | Egzotik Hayvan Hastaliklari El Kitabi.pdf |
| 12 | GUBRELEME_REHBERI_ADANA_S.pdf |
| 13 | GUBRELEME_REHBERI_ADIYAMAN_S.pdf |
| 14 | GUBRELEME_REHBERI_AFYONKARAHISAR_S.pdf |
| 15 | GUBRELEME_REHBERI_AĞRI_S.pdf |
| 16 | GUBRELEME_REHBERI_AKSARAY_S.pdf |
| 17 | GUBRELEME_REHBERI_AMASYA_S.pdf |
| 18 | GUBRELEME_REHBERI_ANKARA_S.pdf |
| 19 | GUBRELEME_REHBERI_ANTALYA_S.pdf |
| 20 | GUBRELEME_REHBERI_ARDAHAN_S.pdf |
| 21 | GUBRELEME_REHBERI_ARTVİN_S.pdf |
| 22 | GUBRELEME_REHBERI_AYDIN_S.pdf |
| 23 | GUBRELEME_REHBERI_BALIKESIR_S.pdf |
| 24 | GUBRELEME_REHBERI_BARTIN_S.pdf |
| 25 | GUBRELEME_REHBERI_BATMAN_S.pdf |
| 26 | GUBRELEME_REHBERI_BAYBURT_S.pdf |
| 27 | GUBRELEME_REHBERI_BİLECİK_S.pdf |
| 28 | GUBRELEME_REHBERI_BİNGÖL_S.pdf |
| 29 | GUBRELEME_REHBERI_BİTLİS_S.pdf |
| 30 | GUBRELEME_REHBERI_BOLU_S.pdf |
| 31 | GUBRELEME_REHBERI_BURDUR_S.pdf |
| 32 | GUBRELEME_REHBERI_BURSA_S.pdf |
| 33 | GUBRELEME_REHBERI_ÇANAKKALE_S.pdf |
| 34 | GUBRELEME_REHBERI_ÇANKIRI_S.pdf |
| 35 | GUBRELEME_REHBERI_ÇORUM_S.pdf |
| 36 | GUBRELEME_REHBERI_DENİZLİ_S.pdf |
| 37 | GUBRELEME_REHBERI_DİYARBAKIR_S.pdf |
| 38 | GUBRELEME_REHBERI_DÜZCE_S.pdf |
| 39 | GUBRELEME_REHBERI_EDİRNE_S.pdf |
| 40 | GUBRELEME_REHBERI_ELAZIĞ_S.pdf |
| 41 | GUBRELEME_REHBERI_ERZİNCAN_S.pdf |
| 42 | GUBRELEME_REHBERI_ERZURUM_S.pdf |
| 43 | GUBRELEME_REHBERI_ESKİŞEHİR_S.pdf |
| 44 | GUBRELEME_REHBERI_GAZİANTEP_S.pdf |
| 45 | GUBRELEME_REHBERI_GİRESUN_S.pdf |
| 46 | GUBRELEME_REHBERI_GÜMÜŞHANE_S.pdf |
| 47 | GUBRELEME_REHBERI_HAKKARİ_S.pdf |
| 48 | GUBRELEME_REHBERI_HATAY_S.pdf |
| 49 | GUBRELEME_REHBERI_IĞDIR_S.pdf |
| 50 | GUBRELEME_REHBERI_ISPARTA_S.pdf |
| 51 | GUBRELEME_REHBERI_İSTANBUL_S.pdf |
| 52 | GUBRELEME_REHBERI_İZMİR_S.pdf |
| 53 | GUBRELEME_REHBERI_KAHRAMANMARAŞ_S.pdf |
| 54 | GUBRELEME_REHBERI_KARABÜK_S.pdf |
| 55 | GUBRELEME_REHBERI_KARAMAN_S.pdf |
| 56 | GUBRELEME_REHBERI_KARS_S.pdf |
| 57 | GUBRELEME_REHBERI_KASTAMONU_S.pdf |
| 58 | GUBRELEME_REHBERI_KAYSERİ_S.pdf |
| 59 | GUBRELEME_REHBERI_KİLİS_S.pdf |
| 60 | GUBRELEME_REHBERI_KIRIKKALE_S.pdf |
| 61 | GUBRELEME_REHBERI_KIRKLARELİ_S.pdf |
| 62 | GUBRELEME_REHBERI_KIRŞEHİR_S.pdf |
| 63 | GUBRELEME_REHBERI_KOCAELİ_S.pdf |
| 64 | GUBRELEME_REHBERI_KONYA_S.pdf |
| 65 | GUBRELEME_REHBERI_KÜTAHYA_S.pdf |
| 66 | GUBRELEME_REHBERI_MALATYA_S.pdf |
| 67 | GUBRELEME_REHBERI_MANİSA_S.pdf |
| 68 | GUBRELEME_REHBERI_MARDİN_S.pdf |
| 69 | GUBRELEME_REHBERI_MERSİN_S.pdf |
| 70 | GUBRELEME_REHBERI_MUĞLA_S.pdf |
| 71 | GUBRELEME_REHBERI_MUŞ_S.pdf |
| 72 | GUBRELEME_REHBERI_NEVŞEHİR_S.pdf |
| 73 | GUBRELEME_REHBERI_NİĞDE_S.pdf |
| 74 | GUBRELEME_REHBERI_ORDU_S.pdf |
| 75 | GUBRELEME_REHBERI_OSMANİYE_S.pdf |
| 76 | GUBRELEME_REHBERI_RİZE_S.pdf |
| 77 | GUBRELEME_REHBERI_SAKARYA_S.pdf |
| 78 | GUBRELEME_REHBERI_SAMSUN_S.pdf |
| 79 | GUBRELEME_REHBERI_ŞANLIURFA_S.pdf |
| 80 | GUBRELEME_REHBERI_SİİRT_S.pdf |
| 81 | GUBRELEME_REHBERI_SİNOP_S.pdf |
| 82 | GUBRELEME_REHBERI_ŞIRNAK_S.pdf |
| 83 | GUBRELEME_REHBERI_SİVAS_S.pdf |
| 84 | GUBRELEME_REHBERI_TEKİRDAĞ_S.pdf |
| 85 | GUBRELEME_REHBERI_TOKAT_S.pdf |
| 86 | GUBRELEME_REHBERI_TRABZON_S.pdf |
| 87 | GUBRELEME_REHBERI_TUNCELİ_S.pdf |
| 88 | GUBRELEME_REHBERI_UŞAK_S.pdf |
| 89 | GUBRELEME_REHBERI_VAN_S.pdf |
| 90 | GUBRELEME_REHBERI_YALOVA_S.pdf |
| 91 | GUBRELEME_REHBERI_YOZGAT_S.pdf |
| 92 | GUBRELEME_REHBERI_ZONGULDAK_S.pdf |
| 93 | Hastalik_ve_Zararlilarla_Mucadele_antepfıstığı.pdf |
| 94 | Hastalik_ve_Zararlilarla_Mucadele_armut-ayva.pdf |
| 95 | Hastalik_ve_Zararlilarla_Mucadele_bag.pdf |
| 96 | Hastalik_ve_Zararlilarla_Mucadele_celtik.pdf |
| 97 | Hastalik_ve_Zararlilarla_Mucadele_ceviz.pdf |
| 98 | Hastalik_ve_Zararlilarla_Mucadele_cilek.pdf |
| 99 | Hastalik_ve_Zararlilarla_Mucadele_elma.pdf |
| 100 | Hastalik_ve_Zararlilarla_Mucadele_erik.pdf |
| 101 | Hastalik_ve_Zararlilarla_Mucadele_findik.pdf |
| 102 | Hastalik_ve_Zararlilarla_Mucadele_havuc.pdf |
| 103 | Hastalik_ve_Zararlilarla_Mucadele_hıyar-kabak.pdf |
| 104 | Hastalik_ve_Zararlilarla_Mucadele_hububat.pdf |
| 105 | Hastalik_ve_Zararlilarla_Mucadele_kavun-karpuz.pdf |
| 106 | Hastalik_ve_Zararlilarla_Mucadele_kayisi.pdf |
| 107 | Hastalik_ve_Zararlilarla_Mucadele_kiraz.pdf |
| 108 | Hastalik_ve_Zararlilarla_Mucadele_lahanagiller.pdf |
| 109 | Hastalik_ve_Zararlilarla_Mucadele_misir.pdf |
| 110 | Hastalik_ve_Zararlilarla_Mucadele_patates.pdf |
| 111 | Hastalik_ve_Zararlilarla_Mucadele_patlican.pdf |
| 112 | Hastalik_ve_Zararlilarla_Mucadele_seftali.pdf |
| 113 | Hastalik_ve_Zararlilarla_Mucadele_sogan-sarımsak.pdf |
| 114 | Hastalik_ve_Zararlilarla_Mucadele_zeytin.pdf |
| 115 | koyun_keci_vebasi_ppr_acil_eylem_plani.pdf |
| 116 | Kriterler_BU.pdf |
| 117 | Kriterler_KY.pdf |
| 118 | Kriterler_MS.pdf |
| 119 | Kriterler_TC.pdf |
| 120 | LSD+Acil+Eylem+Plani+2023.pdf |
| 121 | newcastle_acil_eylem_plani.pdf |
| 122 | Pamuk Kontrol Noktaları.pdf |
| 123 | Sap_Hastaligi_Acil_Eylem_Plani.pdf |
| 124 | Sigirlarda+Dis+Gorunuse+Gore+Siniflandirma.pdf |
| 125 | sogan_patates_domates_mildiyosu.pdf |
| 126 | TİCARİ AMAÇLI SU ÜRÜNLERİ AVCILIĞININ.pdf |
| 127 | turuncgil_hastalik_ve_zararlilari_ile_mucadele.pdf |
| 128 | yapragi_yenen_sebzeler.pdf |
| 129 | yemeklik_baklagil.pdf |