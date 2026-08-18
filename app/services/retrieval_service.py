"""
TODO: Kullanıcının sorusunu alıp en alakalı chunk'ları getiren servis.
İki aşamalı çalışır: (1) bi-encoder ile geniş bir aday havuzu bulunur (hızlı, kaba),
(2) cross-encoder reranker ile bu havuz yeniden puanlanır (yavaş ama isabetli).
Tek başına bi-encoder skorunun alakasız sonuçları da "alakalı" gösterebildiğini test ederek
kanıtladık - bu yüzden reranker aşaması zorunlu hale getirildi.
"""

import logging
from typing import List

from app.core.config import settings
from app.rag.embedder import embedder
from app.rag.reranker import reranker
from app.repositories.graph_repository import graph_repository
from app.repositories.vector_repository import vector_repository

logger = logging.getLogger(__name__)


class RetrievalService:
    def retrieve(self, query: str, top_k: int = settings.TOP_K) -> List[dict]:
        if not query or not query.strip():
            raise ValueError("Sorgu boş olamaz.")

        try:
            query_embedding = embedder.embed_query(query)
            # Aşama 1: bi-encoder ile geniş bir aday havuzu getir (final top_k'dan fazla)
            candidate_pool = vector_repository.search(
                query_embedding, top_k=settings.RERANK_CANDIDATE_POOL
            )
        except Exception as exc:
            logger.error(f"Retrieval sırasında hata oluştu: {exc}")
            raise RuntimeError("Doküman araması başarısız oldu.") from exc

        if not candidate_pool:
            return []

        try:
            # Aşama 2: cross-encoder ile havuzu yeniden sırala
            reranked = reranker.rerank(query, candidate_pool)
        except Exception as exc:
            logger.error(f"Reranking sırasında hata oluştu: {exc}")
            raise RuntimeError("Yeniden sıralama başarısız oldu.") from exc

        filtered = [r for r in reranked if r["rerank_score"] >= settings.MIN_RERANK_SCORE]

        if not filtered:
            best_score = reranked[0]["rerank_score"] if reranked else "yok"
            logger.info(
                f"'{query}' için yeterli skorda kaynak bulunamadı "
                f"(eşik: {settings.MIN_RERANK_SCORE}, en iyi skor: {best_score})."
            )

        return filtered[:top_k]

    def retrieve_graph_facts(self, query: str) -> List[dict]:
        """
        Sorguda birden fazla varlık birlikte geçiyorsa, önce aralarındaki DOĞRUDAN
        BAĞLANTI ZİNCİRİNİ arar (find_path_between) - "X ile Y arasında ne var" gibi
        sorular için bu, her varlığın ilişkilerini ayrı ayrı dökmekten çok daha isabetli.
        Yol bulunamazsa (ya da tek varlık eşleşirse) eski davranışa döner: her varlığın
        kendi doğrudan ilişkilerini listeler.
        """
        matched_entities = graph_repository.find_entities_by_name(query)
        if not matched_entities:
            return []

        entity_names = [m["entity"] for m in matched_entities]

        path_facts: List[dict] = []
        if len(entity_names) >= 2:
            for i in range(len(entity_names)):
                for j in range(i + 1, len(entity_names)):
                    path = graph_repository.find_path_between(entity_names[i], entity_names[j])
                    path_facts.extend(path)

        logger.info(f"Eşleşen varlıklar: {entity_names} | Bulunan yol: {path_facts}")
        if path_facts:
            return self._dedupe_facts(path_facts)

        seen = set()
        graph_facts = []
        for match in matched_entities:
            relationships = graph_repository.get_relationships_for_entity(match["entity"])
            for rel in relationships:
                key = (rel["subject"], rel["predicate"], rel["object"])
                if key in seen:
                    continue
                seen.add(key)
                graph_facts.append(rel)
        return graph_facts

    @staticmethod
    def _dedupe_facts(facts: List[dict]) -> List[dict]:
        seen = set()
        deduped = []
        for f in facts:
            key = (f["subject"], f["predicate"], f["object"])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(f)
        return deduped


retrieval_service = RetrievalService()