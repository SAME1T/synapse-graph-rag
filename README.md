# Synapse — Local GraphRAG Engine

> **Tamamen yerel/offline çalışan, iki aşamalı retrieval ve bilgi grafıyla desteklenmiş RAG motoru.**
> İnternet bağlantısı yok, bulut API'si yok, veri gizliliği tam.

---

## Nedir?

Synapse, kendi dokümanlarına soru sormanı sağlayan, **sıfır dış bağımlılıkla** çalışan bir Retrieval-Augmented Generation (RAG) motorudur.
Sıradan RAG motorlarından farkı, **iki aşamalı retrieval** (bi-encoder → cross-encoder reranking) ve **bilgi grafı** (knowledge graph) katmanının birlikte çalışmasıdır — vektör aramanın kaçırdığı ilişkisel sorular için graftan yararlanır. Framework'süz, tek dosyalık bir web arayüzü ile birlikte gelir.

---

## Mimari

Kullanıcı Sorusu
│
├─── Vektör Arama (ChromaDB)
│ └─ multilingual-e5-large (bi-encoder)
│ └─ cross-encoder reranker (mMiniLMv2)
│
└─── Graf Araması (NetworkX)
└─ find_entities_by_name → get_relationships / find_path_between
│
▼
LLM Generation (Foundry Local)
qwen3.5-2b-text, ayarlı decoding parametreleriyle
(temperature, max_tokens, frequency_penalty)
│
Deterministik Prompt-Leak Temizliği
+ Tekrar Döngüsü Dedektörü
+ Groundedness Kontrolü (skor ≥ 0.25)
│
▼
API Cevabı (is_fallback ile net ayrım) → Web Arayüzü


### Doküman İndeksleme Akışı

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

Doküman Silme
│
SQLite kaydı + ChromaDB chunk'ları + graf kenarları silinir
(yalnızca yetim kalan graf düğümleri kaldırılır — başka dokümana
bağlı varlıklar korunur)


---

## Özellikler

| Katman | Teknoloji | Açıklama |
|--------|-----------|----------|
| **Embedding** | `intfloat/multilingual-e5-large` | Çok dilli bi-encoder, CPU'da çalışır |
| **Vector DB** | ChromaDB | Yerel, kalıcı vektör deposu |
| **Reranker** | `mmarco-mMiniLMv2-L12-H384-v1` | Cross-encoder, bi-encoder adaylarını yeniden sıralar |
| **Knowledge Graph** | NetworkX `MultiDiGraph` | Yönlendirilmiş çoklu ilişki grafı, pickle ile kalıcı |
| **Graf Çıkarımı** | phi-4-mini (Foundry Local) | JSON formatında varlık/ilişki çıkarımı |
| **LLM Generation** | qwen3.5-2b-text (Foundry Local) | Ayarlı decoding parametreleri (temperature/max_tokens/frequency_penalty) ile |
| **Answer QC** | Heuristik groundedness + tekrar dedektörü | Kelime örtüşmesi ve trigram tekrar oranı tabanlı çift katmanlı filtre |
| **Prompt-Leak Koruması** | Deterministik post-processing | Modelin kendi prompt'unu geri üretmesi durumunda metni kesin olarak temizler |
| **API** | FastAPI + Pydantic v2 | Swagger UI dahil |
| **Web UI** | Vanilla JS, tek dosya | Chat arayüzü, doküman yönetimi, graf istatistikleri |
| **Evaluation** | Özel black-box test script'i | Kategori bazlı (factual / out-of-domain / graph) otomatik raporlama |

---

## API Endpoint'leri

### Doküman Yönetimi
| Method | Endpoint | Açıklama |
|--------|----------|----------|
| `POST` | `/api/v1/documents/upload` | PDF veya TXT yükle, indeksle |
| `GET` | `/api/v1/documents/{id}` | Doküman durumunu sorgula |
| `GET` | `/api/v1/documents/` | Tüm dokümanları listele |
| `DELETE` | `/api/v1/documents/{id}` | Dokümanı ve ona ait vektör/graf verisini sil |

### Sorgulama
| Method | Endpoint | Açıklama |
|--------|----------|----------|
| `POST` | `/api/v1/query/ask` | **Tam GraphRAG akışı** — vektör + graf + LLM, `is_fallback` ile net durum ayrımı |
| `POST` | `/api/v1/query/search` | Sadece retrieval (LLM çağırmadan) |

### Graf
| Method | Endpoint | Açıklama |
|--------|----------|----------|
| `GET` | `/api/v1/graph/stats` | Grafttaki toplam varlık ve ilişki sayısı |
| `GET` | `/api/v1/graph/entities/{name}/neighbors` | Bir varlığın komşuları ve ilişkileri |
| `GET` | `/api/v1/graph/dump` | Debug: grafın tamamını (tüm düğüm/kenar) döker |

### Web Arayüzü
| Method | Endpoint | Açıklama |
|--------|----------|----------|
| `GET` | `/` | Chat arayüzü (statik, framework'süz) |

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

Sunucu ayağa kalktıktan sonra:
- **Web Arayüzü:** http://localhost:8000/
- **Swagger UI:** http://localhost:8000/docs

### Testleri Çalıştırma

```bash
# Birim testleri (graf gezinti algoritması, LLM'den bağımsız)
python -m pytest tests/ -v

# Uçtan uca değerlendirme (sunucu ayakta ve bir doküman yüklü olmalı)
python -m evaluation.evaluate
```

---

## Proje Yapısı

synapse-graph-rag/
├── app/
│ ├── api/
│ │ ├── routers/ # FastAPI router'ları (query, ingestion, graph)
│ │ └── schemas/ # Pydantic request/response modelleri
│ ├── core/
│ │ ├── config.py # Merkezi ayar yönetimi (pydantic-settings)
│ │ ├── llm_manager.py # Çok-model Foundry Local yöneticisi (decoding ayarları dahil)
│ │ └── prompts.py # Tüm LLM prompt şablonları
│ ├── rag/
│ │ ├── chunking.py # Metin parçalama
│ │ ├── embedder.py # Bi-encoder embedding
│ │ ├── reranker.py # Cross-encoder reranking
│ │ ├── graph_builder.py # LLM tabanlı varlık/ilişki çıkarımı
│ │ └── answer_quality.py # Groundedness skoru + tekrar dedektörü + format temizliği
│ ├── repositories/
│ │ ├── document_repository.py # SQLite doküman meta verisi
│ │ ├── vector_repository.py # ChromaDB vektör deposu
│ │ └── graph_repository.py # NetworkX graf deposu (+ find_path_between)
│ └── services/
│ ├── ingestion_service.py # Doküman işleme + silme orkestrasyonu
│ ├── retrieval_service.py # İki aşamalı retrieval + graf araması
│ ├── generation_service.py # LLM cevap üretimi + is_fallback ayrımı
│ └── extraction_service.py # Graf çıkarım orkestrasyon
├── static/
│ └── index.html # Tek dosyalık web arayüzü
├── evaluation/
│ ├── test_questions.json # Kategori bazlı test soruları
│ └── evaluate.py # Black-box değerlendirme script'i
├── data/
│ ├── raw_documents/ # Yüklenen ham dosyalar
│ └── processed/ # ChromaDB, graf ve SQLite verileri
├── tests/
│ └── test_graph_repository.py # Graf gezinti algoritması birim testleri
└── requirements.txt


---

## Tasarım Kararları

**Neden iki aşamalı retrieval?**
Bi-encoder (embedding) hızlı ama kabadır — geniş bir aday havuzu getirir. Cross-encoder soru ve dokümanı birlikte okuyarak daha isabetli sıralar; sadece az sayıda aday üzerinde çalıştığı için performans kaybı minimumdur.

**Neden Knowledge Graph?**
"Ahmet ile Ayşe arasındaki bağlantı nedir?" gibi **ilişkisel sorular** vektör aramayla zor cevaplanır. Graf, dokümanlardan çıkarılan varlık-ilişki yapısını muhafaza eder ve `find_path_between` ile çok-adımlı gezinti yaparak bu tür sorularda doğrudan yanıt sağlar.

**Neden best-effort graf çıkarımı?**
Küçük modeller bazen geçersiz JSON üretebilir veya zaman aşımına uğrayabilir. Graf çıkarımı başarısız olsa bile temel vektör arama çalışmaya devam eder — kullanıcı her durumda cevap alabilir.

**Neden groundedness skoru + tekrar dedektörü (iki ayrı kontrol)?**
Groundedness, cevabın kaynaklarla kelime düzeyinde örtüşüp örtüşmediğini ölçer — ama bu ölçüt, modelin bir tekrar döngüsüne girip aynı kaynak kelimelerini defalarca tekrarladığı durumları yakalayamaz (tekrarlanan kelimeler skoru yapay olarak şişirir). Bu yüzden ayrı bir trigram tekrar oranı kontrolü eklendi — ikisi birlikte, tek başına hiçbirinin yakalayamayacağı hata sınıflarını kapsar.

**Neden deterministik prompt-leak temizliği?**
Küçük yerel modeller, cevaplarının bittiğini işaretleyen durma (stop/EOS) sinyalini her zaman güvenilir vermiyor; bazen kendi aldığı prompt'u ("user KAYNAK METİNLER: ...") yeniden üretmeye başlıyor. SDK'nın stop-sequence desteğine güvenmek yerine, bu izleri regex ile kesin olarak tespit edip cevabı o noktada kesiyoruz — SDK sürümünden bağımsız, garantili bir çözüm.

**Neden `is_fallback` ayrı bir alan, `used_llm` yetmiyor mu?**
`used_llm=True` iken cevap yine de kalite kontrolünden geçemeyip reddedilebiliyor — bu durumda kullanıcıya bir fallback mesajı gösterilir ama `used_llm` bunu yansıtmaz. `is_fallback`, "kullanıcıya gösterilen şey gerçek bir cevap mı, yoksa bir ret mesajı mı" sorusuna net cevap verir; arayüz artık doğru rozeti gösterebiliyor.

**Neden görev bazlı model routing (generation ≠ extraction modeli)?**
Yapılandırılmış JSON çıkarımı, serbest metin üretiminden daha fazla instruction-following hassasiyeti gerektirir. Extraction için ayrı ve daha güçlü bir model (`phi-4-mini`), farklı decoding ayarlarıyla (düşük temperature, daha yüksek max_tokens) kullanılır.

---

## Bilinen Sınırlamalar

Bu proje, küçük yerel modellerle (2-4B parametre) çalışmanın gerçek zorluklarını bilerek gizlemeden belgelemektedir; aşağıdaki bulguların çoğu `evaluation/evaluate.py` ile ölçülmüş ve `evaluation/results.json` içinde kayıtlıdır:

- **Entity resolution yok:** Aynı varlık dokümanın farklı yerlerinde farklı adlarla geçtiğinde (örn. "Mustafa Kemal" / "Yarbay" / "Kemal"), graf bunları ayrı düğümler olarak kaydedebilir.
- **Predicate doğruluğu, entity doğruluğundan düşük:** Extraction modeli varlıkları genelde doğru tespit eder, ancak aralarındaki ilişkinin türünü bazen yanlış çıkarabilir.
- **Devrik cümle yapısında özne kayması:** Türkçenin özne-sonda kalıplarında, model bazen gerçek özneyi yanlış tespit edebilir. Few-shot örneklerle kısmen iyileştirilmiştir.
- **Reranker, tanım-formatlı sorularda aşırı temkinli olabilir:** Bi-encoder skoru yüksek olsa bile, cross-encoder bazı tanım/kısaltma soruları için tüm adaylara düşük skor verebilir. İki ayrı dokümanda, iki farklı örnekle doğrulanmıştır: "Anzak kısaltması hangi ülkeleri temsil eder?" (evaluation raporunda q5) ve "SLS kısaltması neyi ifade eder?" — ikisinde de kaynak metinde bilgi açıkça mevcutken retrieval hiçbir sonuç döndürmedi.
- **İsim/alias uyuşmazlığı:** Sorguda kullanılan bir isim dokümanda hiç geçmiyorsa (örn. "Atatürk" yerine dokümanda sadece "Mustafa Kemal" yazıyorsa), sistem doğru şekilde "bulunamadı" der — ama bu, iki ismin aynı varlığa işaret ettiğini bilemediği için olur.
- **CPU-only ortamda extraction gecikmesi ve hata oranı ölçekle artıyor:** 5 chunk'lık küçük bir dokümanda ~%0-20 olan chunk başarısızlık oranı, 38 chunk'lık büyük bir dokümanda ~%29'a çıktığı gözlemlenmiştir. Best-effort tasarım sayesinde bu durum dokümanın tamamen başarısız sayılmasına yol açmaz.
- **Küçük modellerde tekrar döngüsü (repetition loop) riski:** Model bazen aynı ifadeyi (örn. "son derece önemli") onlarca kez tekrarlayarak anlamsız, uzun bir çıktı üretebiliyordu. `temperature`, `max_tokens` ve `frequency_penalty` ayarlarıyla büyük ölçüde azaltıldı ve ayrıca bir trigram tekrar dedektörü eklendi — ancak küçük modellerde bu riskin sıfıra indiği garanti edilemez.
- **Prompt sızıntısı (EOS güvenilirliği):** Model, cevabını bitirdiğini işaretleyen durma sinyalini bazen vermeyip kendi aldığı prompt'u yeniden üretmeye başlıyordu. Deterministik post-processing ile bu izler kesin olarak temizleniyor, ancak bu SDK/model seviyesindeki güvenilirlik sorununun kök nedenini çözmüyor, sonucunu telafi ediyor.
- **Çoklu doküman kirliliği:** Vektör veritabanında birbirinden tamamen farklı konularda dokümanlar bulunduğunda, gerçekte alakasız bir soru bile retrieval eşiğini (0.3) az farkla geçebiliyor (ölçülmüş örnek: birden fazla teknoloji temalı doküman eklendikten sonra "Python'un avantajları nedir" sorusu, groundedness kontrolü tarafından reddedilmeden önce retrieval'i geçmişti). Groundedness kontrolü bunu ikinci savunma hattı olarak yakalıyor.
- **Evaluation script'inin kendi sınırı:** `out_of_domain` kategorisindeki test sorularında `expected_keywords` boş liste olduğu için, bu kategorideki testler yapısal olarak neredeyse hiç "FAIL" olamıyor; asıl güvenilir sinyal `passed` alanı değil, ham `groundedness_score` değeridir. Bu, henüz düzeltilmemiş bilinen bir test tasarımı zayıflığıdır.
- **Fixed-size chunking, kısa metinlerde konu sınırlarını önemsemiyor:** `chunk_size=900` ayarı, art arda gelen kısa ve alakasız paragrafları (örn. üç farklı konudan birer paragraf içeren bir test dokümanında bileşik faiz, Python ve fotosentez) tek bir chunk'ta birleştirebiliyor. Bu durumda retrieval doğru chunk'ı bulsa bile, LLM'e verilen bağlam üç farklı konuyu karıştırıyor ve cevap gereksiz yere birden fazla konuyu art arda dökebiliyor. Uzun, tek-konulu dokümanlarda (örn. 9 sayfalık tarihsel test dokümanı) bu sorun gözlemlenmedi — riski en çok kısa, çok konulu dosyalarda artıyor.

---

## Geliştirme Fazları

| Faz | İçerik |
|-----|--------|
| **Faz 1 — Temel RAG** | FastAPI iskeleti, chunking, embedding, ChromaDB, bi-encoder retrieval, Foundry Local LLM entegrasyonu |
| **Faz 2 — Üretim Kalitesi** | Cross-encoder reranking, answer quality kontrolü (groundedness), format temizliği, iki aşamalı retrieval pipeline |
| **Faz 3 — GraphRAG** | NetworkX bilgi grafı, LLM tabanlı varlık/ilişki çıkarımı, çok-adımlı graf gezintisi, çok-model yönetimi |
| **Faz 4 — Değerlendirme ve Arayüz** | Birim testleri, kategori bazlı otomatik evaluation, doküman silme, framework'süz web arayüzü |
| **Faz 5 — Üretim Stabilizasyonu** | Decoding parametre ayarı (temperature/max_tokens/frequency_penalty), deterministik prompt-leak temizliği, tekrar döngüsü dedektörü, `is_fallback` durum ayrımı |
