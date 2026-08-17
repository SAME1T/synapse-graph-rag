"""
TODO: Doküman kayıtlarını (yükleme durumu, chunk sayısı vb.) SQLite'ta tutan repository.
Vektör verisi burada YOK - o vector_repository.py'de. Burada sadece "hangi doküman,
ne durumda" bilgisi var.
"""

import logging
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import List, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)


class DocumentRepository:
    """
    Yüklenen dokümanların yaşam döngüsünü (pending -> processing -> completed/failed)
    SQLite üzerinde takip eden katman.
    """

    def __init__(self, db_path=settings.DOCUMENT_DB_PATH):
        self._db_path = db_path
        self._initialized = False

    def _ensure_schema(self) -> None:
        """
        Tablo yoksa oluşturur. Her repository metodundan önce çağrılır,
        ama tablo zaten varsa hiçbir şey yapmaz (IF NOT EXISTS sayesinde).
        """
        if self._initialized:
            return

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS documents (
                        id TEXT PRIMARY KEY,
                        filename TEXT NOT NULL,
                        file_path TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'pending',
                        chunk_count INTEGER DEFAULT 0,
                        error_message TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
            self._initialized = True
        except Exception as exc:
            logger.error(f"Doküman tablosu oluşturulamadı: {exc}")
            raise RuntimeError("Doküman veritabanı hazırlanamadı.") from exc

    @contextmanager
    def _get_connection(self):
        """
        Her işlemde yeni bağlantı açıp işi bitince otomatik kapatan yardımcı fonksiyon.
        'with' ile kullanılınca hata olsa bile bağlantının açık kalmamasını garantiler.
        """
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row  # sonuçları dict gibi okuyabilmek için
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def create_document(self, filename: str, file_path: str) -> str:
        """
        Yeni bir doküman kaydı oluşturur, durumu 'pending' olarak başlar.
        Geriye o dokümanın benzersiz id'sini döner.
        """
        self._ensure_schema()
        document_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO documents (id, filename, file_path, status, created_at, updated_at)
                    VALUES (?, ?, ?, 'pending', ?, ?)
                    """,
                    (document_id, filename, file_path, now, now),
                )
            logger.info(f"Doküman kaydı oluşturuldu: {filename} ({document_id})")
            return document_id
        except Exception as exc:
            logger.error(f"Doküman kaydı oluşturulamadı: {exc}")
            raise RuntimeError("Doküman kaydedilemedi.") from exc

    def update_status(
        self,
        document_id: str,
        status: str,
        chunk_count: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """
        Doküman işlenirken/işlem bitince durumunu günceller.
        status: 'processing' | 'completed' | 'failed'
        """
        self._ensure_schema()
        now = datetime.now(timezone.utc).isoformat()

        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    UPDATE documents
                    SET status = ?, chunk_count = COALESCE(?, chunk_count),
                        error_message = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (status, chunk_count, error_message, now, document_id),
                )
            logger.info(f"Doküman durumu güncellendi: {document_id} -> {status}")
        except Exception as exc:
            logger.error(f"Doküman durumu güncellenemedi: {exc}")
            raise RuntimeError("Doküman durumu güncellenemedi.") from exc

    def get_document(self, document_id: str) -> Optional[dict]:
        """Tek bir dokümanın güncel kaydını döner, yoksa None."""
        self._ensure_schema()
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
            return dict(row) if row else None

    def list_documents(self) -> List[dict]:
        """Tüm dokümanları en yeniden eskiye sıralı döner - API'de listeleme için kullanılır."""
        self._ensure_schema()
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM documents ORDER BY created_at DESC"
            ).fetchall()
            return [dict(row) for row in rows]


# Tüm uygulamanın kullanacağı tek nesne
document_repository = DocumentRepository()