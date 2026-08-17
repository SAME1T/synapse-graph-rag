"""
TODO: Metni yerel embedding modeliyle vektöre çeviren dosya.
Sorgu (query) ve doküman (passage) embedding'leri AYRI metodlarla yapılır,
çünkü E5 tabanlı modeller bu ikisini farklı önekle (prefix) işler.
"""

import logging
from typing import List, Optional

from sentence_transformers import SentenceTransformer

from app.core.config import settings

logger = logging.getLogger(__name__)


class LocalEmbedder:
    """
    E5 ailesi (intfloat/multilingual-e5-large) modelle metin embedding'i üretir.
    """

    def __init__(self, model_name: str = settings.EMBEDDING_MODEL_NAME):
        self._model_name = model_name
        self._model: Optional[SentenceTransformer] = None

    def load(self) -> None:
        """
        Modeli belleğe yükler. İlk çalıştırmada Hugging Face Hub'dan indirilir,
        sonraki çalıştırmalarda yerel önbellekten (cache) okunur - internet gerekmez.
        """
        if self._model is not None:
            return  # zaten yüklü

        try:
            logger.info(f"Embedding modeli yükleniyor: {self._model_name}")
            self._model = SentenceTransformer(self._model_name)
            logger.info("Embedding modeli hazır.")
        except Exception as exc:
            logger.error(f"Embedding modeli yüklenemedi: {exc}")
            raise RuntimeError(
                f"'{self._model_name}' embedding modeli yüklenemedi. "
                f"İnternet bağlantısını veya model adını kontrol edin."
            ) from exc

    def embed_query(self, text: str) -> List[float]:
        """
        Kullanıcının sorusunu vektöre çevirir. "query: " öneki eklenir.
        """
        self._ensure_loaded()
        prefixed = f"query: {text}"
        vector = self._model.encode(prefixed, normalize_embeddings=True)
        return vector.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Doküman chunk'larını toplu (batch) halde vektöre çevirir - "passage: " öneki eklenir.
        Toplu işlem, tek tek işlemekten çok daha hızlıdır.
        """
        self._ensure_loaded()
        prefixed = [f"passage: {t}" for t in texts]
        vectors = self._model.encode(
            prefixed,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=True,
        )
        return vectors.tolist()

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self.load()


# Tüm uygulamanın kullanacağı tek nesne
embedder = LocalEmbedder()