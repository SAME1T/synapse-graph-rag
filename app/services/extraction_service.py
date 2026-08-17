"""
TODO: Chunk'lardan çıkarılan varlık/ilişkileri graph_repository'ye yazan orkestrasyon servisi.
graph_builder.py (LLM ile çıkarım) ile graph_repository.py (NetworkX'e yazma) arasındaki köprü.
"""

import logging
from typing import List

from app.rag.graph_builder import extract_graph_elements
from app.repositories.graph_repository import graph_repository

logger = logging.getLogger(__name__)


class ExtractionService:
    def process_chunks_for_graph(self, chunks: List[dict], document_id: str) -> dict:
        """
        chunks: [{"chunk_id": ..., "text": ...}, ...] biçiminde liste.
        Bir chunk'ın çıkarımı başarısız olsa bile diğer chunk'ların işlenmesi durmaz.
        """
        entity_mentions = 0
        relationship_mentions = 0
        failed_chunks = 0

        for chunk in chunks:
            try:
                result = extract_graph_elements(chunk["text"])
            except Exception as exc:
                logger.error(f"Chunk {chunk['chunk_id']} için graf çıkarımı başarısız: {exc}")
                failed_chunks += 1
                continue

            for entity in result.entities:
                graph_repository.add_entity(
                    name=entity.name, entity_type=entity.type, document_id=document_id
                )
                entity_mentions += 1

            for rel in result.relationships:
                graph_repository.add_relationship(
                    subject=rel.subject,
                    predicate=rel.predicate,
                    obj=rel.object,
                    document_id=document_id,
                    chunk_id=chunk["chunk_id"],
                )
                relationship_mentions += 1

        graph_repository.commit()

        logger.info(
            f"Graf çıkarımı tamamlandı: {entity_mentions} varlık bahsi, "
            f"{relationship_mentions} ilişki bahsi, {failed_chunks} başarısız chunk "
            f"(doküman: {document_id})."
        )

        return {
            "entity_mentions": entity_mentions,
            "relationship_mentions": relationship_mentions,
            "failed_chunks": failed_chunks,
        }


extraction_service = ExtractionService()
