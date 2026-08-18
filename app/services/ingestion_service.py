"""
TODO: Doküman işleme iş akışını yöneten servis.
NOT: Bu servis doküman kaydını OLUŞTURMAZ - sadece var olan bir kaydı işler.

Akış: metni çıkar -> chunk'la -> embed et -> vektör DB'ye yaz -> [graf çıkarımı] -> durumu güncelle.
Graf çıkarımı BEST-EFFORT'tur: başarısız olsa bile doküman "completed" sayılır, çünkü
temel RAG (vektör arama) zaten çalışır durumda - graf sadece ek bir katman, kritik yol değil.
"""

import logging
from pathlib import Path

from pypdf import PdfReader

from app.rag.chunking import chunk_text
from app.rag.embedder import embedder
from app.repositories.document_repository import document_repository
from app.repositories.graph_repository import graph_repository
from app.repositories.vector_repository import vector_repository
from app.services.extraction_service import extraction_service

logger = logging.getLogger(__name__)


class IngestionService:
    def process_document(self, document_id: str, file_path: Path, filename: str) -> None:
        try:
            document_repository.update_status(document_id, status="processing")

            raw_text = self._extract_text(file_path)
            if not raw_text.strip():
                raise ValueError(
                    "Dokümandan hiç metin çıkarılamadı (boş ya da taranmış görsel PDF olabilir)."
                )

            chunks = chunk_text(raw_text)
            if not chunks:
                raise ValueError("Metin hiçbir chunk'a bölünemedi.")

            chunk_ids = [f"{document_id}_{c.chunk_index}" for c in chunks]
            texts = [c.text for c in chunks]
            embeddings = embedder.embed_documents(texts)
            metadatas = [
                {"document_id": document_id, "filename": filename, "chunk_index": c.chunk_index}
                for c in chunks
            ]

            vector_repository.add_chunks(
                chunk_ids=chunk_ids, texts=texts, embeddings=embeddings, metadatas=metadatas
            )

            try:
                chunk_records = [
                    {"chunk_id": cid, "text": text} for cid, text in zip(chunk_ids, texts)
                ]
                extraction_service.process_chunks_for_graph(chunk_records, document_id=document_id)
            except Exception as exc:
                logger.error(
                    f"Graf çıkarımı başarısız oldu, doküman yine de 'completed' işaretlenecek: {exc}"
                )

            document_repository.update_status(
                document_id, status="completed", chunk_count=len(chunks)
            )
            logger.info(f"Doküman başarıyla indekslendi: {filename} ({len(chunks)} chunk)")

        except Exception as exc:
            logger.error(f"Doküman işlenirken hata oluştu ({filename}): {exc}")
            document_repository.update_status(
                document_id, status="failed", error_message=str(exc)
            )
            raise

    def _extract_text(self, file_path: Path) -> str:
        suffix = file_path.suffix.lower()
        if suffix == ".txt":
            return file_path.read_text(encoding="utf-8", errors="ignore")
        if suffix == ".pdf":
            return self._extract_text_from_pdf(file_path)
        raise ValueError(f"Desteklenmeyen dosya tipi: {suffix}")

    def _extract_text_from_pdf(self, file_path: Path) -> str:
        try:
            reader = PdfReader(str(file_path))
            pages_text = [page.extract_text() or "" for page in reader.pages]
            return "\n".join(pages_text)
        except Exception as exc:
            raise ValueError(f"PDF okunamadı: {exc}") from exc

    def delete_document(self, document_id: str) -> bool:
        """
        Bir dokümanı üç katmandan da (SQLite, vektör DB, graf) ve fiziksel
        dosya sisteminden siler. Doküman bulunamazsa False döner.
        Alt katmanlardan biri başarısız olsa bile diğerlerinin silinmesi denenir -
        kısmi bir silme, hiç silmemekten iyidir.
        """
        file_path = document_repository.delete_document(document_id)
        if file_path is None:
            return False

        try:
            vector_repository.delete_by_document(document_id)
        except Exception as exc:
            logger.error(f"Vektör verisi silinirken hata oluştu: {exc}")

        try:
            graph_repository.remove_document(document_id)
        except Exception as exc:
            logger.error(f"Graf verisi silinirken hata oluştu: {exc}")

        try:
            Path(file_path).unlink(missing_ok=True)
        except Exception as exc:
            logger.error(f"Fiziksel dosya silinirken hata oluştu: {exc}")

        return True


ingestion_service = IngestionService()