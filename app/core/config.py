"""
TODO: Projenin tüm ayarlarını tek merkezden yöneten dosya.
Diğer tüm dosyalar değerleri buradan okur, hiçbir dosyada "sabit değer" (magic number) olmaz.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # .env dosyası varsa, buradaki varsayılan değerlerin üzerine yazar
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # --- Genel ---
    PROJECT_NAME: str = "Synapse - Local GraphRAG Engine"

    # --- Dosya Yolları (işletim sisteminden bağımsız çalışsın diye Path kullanıyoruz) ---
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    RAW_DOCUMENTS_DIR: Path = BASE_DIR / "data" / "raw_documents"
    PROCESSED_DIR: Path = BASE_DIR / "data" / "processed"
    VECTOR_DB_DIR: Path = PROCESSED_DIR / "vector_store"
    GRAPH_STORE_PATH: Path = PROCESSED_DIR / "graph_store.gpickle"
    DOCUMENT_DB_PATH: Path = PROCESSED_DIR / "documents.db"

    # --- Chunking (metin parçalama) Ayarları ---
    CHUNK_SIZE: int = 900          # her parçanın yaklaşık karakter uzunluğu
    CHUNK_OVERLAP: int = 140       # parçalar arası örtüşme (bağlam kaybını önler)

    # --- Embedding Modeli (yerel, sentence-transformers ile çalışır) ---
    EMBEDDING_MODEL_NAME: str = "intfloat/multilingual-e5-large"
    EMBEDDING_DIMENSION: int = 1024

    # --- Yerel LLM (Foundry Local üzerinden çalışan model) ---
    LLM_MODEL_ALIAS: str = "qwen3.5-2b-text"
    EXTRACTION_MODEL_ALIAS: str = "phi-4-mini"

    # --- Retrieval (bilgi getirme) Ayarları ---
    TOP_K: int = 4                     # kaç parça getirileceği
    MIN_RETRIEVAL_SCORE: float = 0.55  # bu skorun altındaki sonuçlar "yetersiz kaynak" sayılır
    RERANKER_MODEL_NAME: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    MIN_RERANK_SCORE: float = 0.3
    RERANK_CANDIDATE_POOL: int = 10  # bi-encoder'dan kaç aday alınıp reranker'a verilecek

    # --- API ---
    API_V1_PREFIX: str = "/api/v1"


# Tek bir "settings" nesnesi oluşturuyoruz, diğer dosyalar bunu import edip kullanacak
settings = Settings()