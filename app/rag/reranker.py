"""
TODO: Bi-encoder'ın getirdiği aday chunk'ları cross-encoder ile yeniden sıralayan modül.
Bi-encoder soru ve dokümanı AYRI AYRI vektöre çevirir (hızlı, kaba).
Cross-encoder soru ve dokümanı BİRLİKTE okur (yavaş, isabetli).
Bu yüzden cross-encoder'ı sadece bi-encoder'ın getirdiği az sayıda aday üzerinde çalıştırıyoruz -
tüm veritabanında çalıştırmak çok yavaş olurdu.
"""

import logging
from typing import List, Optional

import torch
from sentence_transformers import CrossEncoder

from app.core.config import settings

logger = logging.getLogger(__name__)


class LocalReranker:
    """
    Çok dilli bir cross-encoder ile soru-chunk çiftlerinin gerçek alaka
    düzeyini 0-1 arası bir olasılık olarak puanlar.
    """

    def __init__(self, model_name: str = settings.RERANKER_MODEL_NAME):
        self._model_name = model_name
        self._model: Optional[CrossEncoder] = None

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            logger.info(f"Reranker modeli yükleniyor: {self._model_name}")
            self._model = CrossEncoder(self._model_name)
            logger.info("Reranker modeli hazır.")
        except Exception as exc:
            logger.error(f"Reranker modeli yüklenemedi: {exc}")
            raise RuntimeError(f"'{self._model_name}' reranker modeli yüklenemedi.") from exc

    def rerank(self, query: str, candidates: List[dict]) -> List[dict]:
        """
        candidates: vector_repository.search()'den dönen liste.
        Her adaya 'rerank_score' alanı ekler ve listeyi bu yeni skora göre
        büyükten küçüğe sıralanmış olarak döner.
        """
        if not candidates:
            return candidates

        self._ensure_loaded()
        pairs = [(query, c["text"]) for c in candidates]

        try:
            # Sigmoid uyguluyoruz ki skor 0-1 arası, yorumlanabilir bir olasılık olsun
            raw_scores = self._model.predict(pairs, activation_fct=torch.nn.Sigmoid())
        except Exception as exc:
            logger.error(f"Reranking sırasında hata oluştu: {exc}")
            raise RuntimeError("Yeniden sıralama başarısız oldu.") from exc

        for candidate, score in zip(candidates, raw_scores):
            candidate["rerank_score"] = float(score)

        return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self.load()


reranker = LocalReranker()
