"""
TODO: FastAPI uygulamasının giriş noktası. Router'ları, başlangıç/kapanış
işlemlerini (model yükleme vb.) burada birbirine bağlıyoruz.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routers import graph_router, ingestion_router, query_router
from app.core.config import settings
from app.core.llm_manager import llm_manager
from app.rag.embedder import embedder
from app.rag.reranker import reranker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Açılışta: embedding modeli + yerel LLM belleğe yüklenir.
    Kapanışta: LLM bellekten temiz şekilde kaldırılır.
    """
    logger.info(f"{settings.PROJECT_NAME} başlatılıyor...")

    logger.info("Embedding modeli yükleniyor (ilk çalıştırmada biraz sürebilir)...")
    embedder.load()

    logger.info("Reranker modeli yükleniyor...")
    reranker.load()

    try:
        logger.info("Yerel LLM (Foundry Local) başlatılıyor...")
        llm_manager.load_model(
            settings.LLM_MODEL_ALIAS,
            temperature=settings.GENERATION_TEMPERATURE,
            max_tokens=settings.GENERATION_MAX_TOKENS,
            frequency_penalty=settings.GENERATION_FREQUENCY_PENALTY,
        )
        llm_manager.load_model(
            settings.EXTRACTION_MODEL_ALIAS,
            temperature=settings.EXTRACTION_TEMPERATURE,
            max_tokens=settings.EXTRACTION_MAX_TOKENS,
        )
    except Exception as exc:
        logger.warning(
            f"Yerel LLM başlatılamadı, LLM gerektiren endpoint'ler şimdilik çalışmayacak: {exc}"
        )

    logger.info("Uygulama hazır.")
    yield
    logger.info("Uygulama kapatılıyor...")
    llm_manager.shutdown()


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Tamamen yerel/offline çalışan, graf destekli RAG motoru.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(ingestion_router.router, prefix=settings.API_V1_PREFIX)
app.include_router(query_router.router, prefix=settings.API_V1_PREFIX)
app.include_router(graph_router.router, prefix=settings.API_V1_PREFIX)


@app.get("/health", tags=["System"])
async def health_check():
    """Basit ayakta mıyım kontrolü - manuel test ve ileride izleme için kullanılır."""
    return {"status": "ok", "project": settings.PROJECT_NAME}


app.mount("/", StaticFiles(directory="static", html=True), name="static")