"""
TODO: ChromaDB vektör veritabanına yazma/okuma işlemlerini yöneten repository.
Başka hiçbir dosya chromadb kütüphanesini doğrudan import etmez, hepsi buradan geçer.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import chromadb

from app.core.config import settings

logger = logging.getLogger(__name__)


class VectorRepository:
    """
    Chunk embedding'lerini kaydeden ve benzerlik araması yapan katman.
    """

    def __init__(
        self,
        persist_directory: Path = settings.VECTOR_DB_DIR,
        collection_name: str = "synapse_chunks",
    ):
        self._persist_directory = persist_directory
        self._collection_name = collection_name
        self._client = None
        self._collection = None

    def _ensure_connected(self) -> None:
        """
        Chroma istemcisini ilk kullanımda kurar (embedder'daki load() ile aynı mantık).
        """
        if self._client is not None:
            return

        try:
            self._persist_directory.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self._persist_directory))
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},  # benzerlik ölçütü: kosinüs
            )
            logger.info(f"Vektör veritabanına bağlanıldı: {self._persist_directory}")
        except Exception as exc:
            logger.error(f"Vektör veritabanına bağlanılamadı: {exc}")
            raise RuntimeError("Vektör veritabanı başlatılamadı.") from exc

    def add_chunks(
        self,
        chunk_ids: List[str],
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[dict],
    ) -> None:
        """
        Yeni chunk'ları embedding'leriyle birlikte veritabanına ekler.
        """
        self._ensure_connected()
        try:
            self._collection.add(
                ids=chunk_ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            logger.info(f"{len(chunk_ids)} chunk veritabanına eklendi.")
        except Exception as exc:
            logger.error(f"Chunk eklenirken hata oluştu: {exc}")
            raise RuntimeError("Chunk'lar veritabanına yazılamadı.") from exc

    def search(
        self,
        query_embedding: List[float],
        top_k: int = settings.TOP_K,
        where: Optional[Dict] = None,
    ) -> List[dict]:
        """
        Verilen sorgu vektörüne en benzer chunk'ları getirir.
        'where' parametresi ile metadata bazlı filtreleme de yapılabilir (örn. sadece belirli bir doküman).
        """
        self._ensure_connected()
        try:
            raw_results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where,
            )
            return self._format_results(raw_results)
        except Exception as exc:
            logger.error(f"Arama sırasında hata oluştu: {exc}")
            raise RuntimeError("Vektör araması başarısız oldu.") from exc

    def _format_results(self, raw_results: dict) -> List[dict]:
        """
        Chroma'nın ham çıktısını, geri kalan kodun kolayca kullanacağı
        sade bir listeye çevirir. Uzaklığı (distance) benzerlik skoruna dönüştürür.
        """
        ids = raw_results.get("ids", [[]])[0]
        documents = raw_results.get("documents", [[]])[0]
        metadatas = raw_results.get("metadatas", [[]])[0]
        distances = raw_results.get("distances", [[]])[0]

        formatted = []
        for i in range(len(ids)):
            similarity_score = 1 - distances[i]  # kosinüs uzaklığı -> benzerlik skoru
            formatted.append(
                {
                    "chunk_id": ids[i],
                    "text": documents[i],
                    "metadata": metadatas[i],
                    "score": round(similarity_score, 4),
                }
            )
        return formatted

    def count(self) -> int:
        """Veritabanındaki toplam chunk sayısını döner - test/debug için kullanışlı."""
        self._ensure_connected()
        return self._collection.count()


# Tüm uygulamanın kullanacağı tek nesne
vector_repository = VectorRepository()