# Synapse — Local GraphRAG Engine

> **Tamamen yerel/offline çalışan, iki aşamalı retrieval ve bilgi grafıyla desteklenmiş RAG motoru.**  
> İnternet bağlantısı yok, bulut API'si yok, veri gizliliği tam.

---

## Nedir?

Synapse, kendi dokümanlarına soru sormanı sağlayan, **sıfır dış bağımlılıkla** çalışan bir Retrieval-Augmented Generation (RAG) motorudur.  
Sıradan RAG motorlarından farkı, **iki aşamalı retrieval** (bi-encoder → cross-encoder reranking) ve **bilgi grafı** (knowledge graph) katmanının birlikte çalışmasıdır — vektör aramanın kaçırdığı ilişkisel sorular için graftan yararlanır.

---

## Mimari

```
Kullanıcı Sorusu
       │
       ├─── Vektör Arama (ChromaDB)
       │         └─ multilingual-e5-large (bi-encoder)
       │                   └─ cross-encoder reranker (mMiniLMv2)
       │
       └─── Graf Araması (NetworkX)
                 └─ find_entities_by_name → get_relationships
                           │
                           ▼
                    LLM Generation (Foundry Local)
                    qwen3.5-2b-text (cevap üretimi)
                           │
                    Answer Quality Check
                    (groundedness skoru ≥ 0.25)
                           │
                           ▼
                    API Cevabı (FastAPI)
```

### Doküman İndeksleme Akışı

```
PDF / TXT yükleme
       │
  Metin Çıkarma
       │
  Chunking (900 karakter, 140 overlap)
       │
       ├─── Embedding → ChromaDB (vektör arama için)
       │
       └─── Graf Çıkarımı (best-effort)
                 └─ phi-4-mini → varlık + ilişki → NetworkX MultiDiGraph
```

---

## Özellikler

| Katman | Teknoloji | Açıklama |
|--------|-----------|----------|
| **Embedding** | `intfloat/multilingual-e5-large` | Çok dilli bi-encoder, CPU'da çalışır |
| **Vector DB** | ChromaDB | Yerel, kalıcı vektör deposu |
| **Reranker** | `mmarco-mMiniLMv2-L12-H384-v1` | Cross-encoder, bi-encoder adaylarını yeniden sıralar |
| **Knowledge Graph** | NetworkX `MultiDiGraph` | Yönlendirilmiş çoklu ilişki grafı, pickle ile kalıcı |
| **Graf Çıkarımı** | phi-4-mini (Foundry Local) | JSON formatında varlık/ilişki çıkarımı |
| **LLM Generation** | qwen3.5-2b-text (Foundry Local) | Tamamen yerel, Foundry Local üzerinden |
| **Answer QC** | Heuristik groundedness skoru | Kelime örtüşmesi tabanlı hallüsinasyon filtresi |
| **API** | FastAPI + Pydantic v2 | Swagger UI dahil |

---

## API Endpoint'leri

### Doküman Yönetimi
| Method | Endpoint | Açıklama |
|--------|----------|----------|
| `POST` | `/api/v1/documents/upload` | PDF veya TXT yükle, indeksle |
| `GET` | `/api/v1/documents/{id}` | Doküman durumunu sorgula |
| `GET` | `/api/v1/documents/` | Tüm dokümanları listele |

### Sorgulama
| Method | Endpoint | Açıklama |
|--------|----------|----------|
| `POST` | `/api/v1/query/ask` | **Tam GraphRAG akışı** — vektör + graf + LLM |
| `POST` | `/api/v1/query/search` | Sadece retrieval (LLM çağırmadan) |

### Graf
| Method | Endpoint | Açıklama |
|--------|----------|----------|
| `GET` | `/api/v1/graph/stats` | Grafttaki toplam varlık ve ilişki sayısı |
| `GET` | `/api/v1/graph/entities/{name}/neighbors` | Bir varlığın komşuları ve ilişkileri |

---

## Kurulum

### Gereksinimler
- Python 3.11+
- [Foundry Local](https://github.com/microsoft/foundry-local) kurulu ve çalışıyor olmalı
- Foundry'de `qwen3.5-2b-text` ve `phi-4-mini` modelleri yüklü olmalı

### Adımlar

```bash
# 1. Sanal ortam oluştur
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Linux/Mac

# 2. Bağımlılıkları yükle
pip install -r requirements.txt

# 3. Ortam değişkenlerini ayarla (isteğe bağlı)
copy .env.example .env

# 4. Sunucuyu başlat
uvicorn app.main:app --reload --port 8000
```

Sunucu ayağa kalktıktan sonra Swagger UI: **http://localhost:8000/docs**

---

## Proje Yapısı

```
synapse-graph-rag/
├── app/
│   ├── api/
│   │   ├── routers/          # FastAPI router'ları (query, ingestion, graph)
│   │   └── schemas/          # Pydantic request/response modelleri
│   ├── core/
│   │   ├── config.py         # Merkezi ayar yönetimi (pydantic-settings)
│   │   ├── llm_manager.py    # Çok-model Foundry Local yöneticisi
│   │   └── prompts.py        # Tüm LLM prompt şablonları
│   ├── rag/
│   │   ├── chunking.py       # Metin parçalama
│   │   ├── embedder.py       # Bi-encoder embedding
│   │   ├── reranker.py       # Cross-encoder reranking
│   │   ├── graph_builder.py  # LLM tabanlı varlık/ilişki çıkarımı
│   │   └── answer_quality.py # Groundedness skoru + format temizliği
│   ├── repositories/
│   │   ├── document_repository.py  # SQLite doküman meta verisi
│   │   ├── vector_repository.py    # ChromaDB vektör deposu
│   │   └── graph_repository.py     # NetworkX graf deposu
│   └── services/
│       ├── ingestion_service.py    # Doküman işleme orkestrasyon
│       ├── retrieval_service.py    # İki aşamalı retrieval + graf araması
│       ├── generation_service.py   # LLM cevap üretimi
│       └── extraction_service.py   # Graf çıkarım orkestrasyon
├── data/
│   ├── raw_documents/        # Yüklenen ham dosyalar
│   └── processed/            # ChromaDB, graf ve SQLite verileri
├── tests/
└── requirements.txt
```

---

## Tasarım Kararları

**Neden iki aşamalı retrieval?**  
Bi-encoder (embedding) hızlı ama kabadır — geniş bir aday havuzu getirir. Cross-encoder soru ve dokümanı birlikte okuyarak daha isabetli sıralar; sadece az sayıda aday üzerinde çalıştığı için performans kaybı minimumdur.

**Neden Knowledge Graph?**  
"Ahmet ile Ayşe arasındaki bağlantı nedir?" gibi **ilişkisel sorular** vektör aramayla zor cevaplanır. Graf, dokümanlardan çıkarılan varlık-ilişki yapısını muhafaza eder ve bu tür sorularda doğrudan yanıt sağlar.

**Neden best-effort graf çıkarımı?**  
Küçük modeller bazen geçersiz JSON üretebilir. Graf çıkarımı başarısız olsa bile temel vektör arama çalışmaya devam eder — kullanıcı her durumda cevap alabilir.

**Neden groundedness skoru?**  
Ekstra LLM çağrısı yapmadan ucuz bir hallüsinasyon filtresi. Cevaptaki anlamlı kelimelerin %25'inden azı kaynaklarda geçiyorsa model muhtemelen uydurmuştur — fallback mesajı gösterilir.

---

## Commit Geçmişi / Geliştirme Fazları

| Faz | İçerik |
|-----|--------|
| **Faz 1 — Temel RAG** | FastAPI iskeleti, chunking, embedding, ChromaDB, bi-encoder retrieval, Foundry Local LLM entegrasyonu |
| **Faz 2 — Üretim Kalitesi** | Cross-encoder reranking, answer quality kontrolü (groundedness), format temizliği, iki aşamalı retrieval pipeline |
| **Faz 3 — GraphRAG** | NetworkX bilgi grafı, LLM tabanlı varlık/ilişki çıkarımı, graf-destekli retrieval, çok-model yönetimi |

---

## Lisans

MIT
