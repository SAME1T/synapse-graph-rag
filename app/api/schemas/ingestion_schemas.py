"""
TODO: Doküman yükleme/durum sorgulama isteklerinin ve cevaplarının veri şekillerini tanımlar.
"""

from typing import List, Optional

from pydantic import BaseModel


class DocumentUploadResponse(BaseModel):
    """Yükleme isteği hemen dönerken kullanılan cevap - işlem arka planda sürer."""
    document_id: str
    filename: str
    status: str
    message: str


class DocumentStatusResponse(BaseModel):
    """Bir dokümanın güncel durumunu sorgularken dönen cevap."""
    id: str
    filename: str
    status: str
    chunk_count: int
    error_message: Optional[str] = None
    created_at: str
    updated_at: str


class DocumentListResponse(BaseModel):
    """Tüm dokümanların listesini dönerken kullanılan cevap."""
    documents: List[DocumentStatusResponse]
    total: int