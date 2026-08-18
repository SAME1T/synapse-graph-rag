"""
TODO: Doküman yükleme ve durum sorgulama API endpoint'lerini tanımlar.
"""

import logging
import shutil
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile

from app.api.schemas.ingestion_schemas import (
    DocumentListResponse,
    DocumentStatusResponse,
    DocumentUploadResponse,
)
from app.core.config import settings
from app.repositories.document_repository import document_repository
from app.services.ingestion_service import ingestion_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])

ALLOWED_EXTENSIONS = {".txt", ".pdf"}


@router.post("/upload", response_model=DocumentUploadResponse, status_code=202)
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Dosyayı diske kaydeder, kaydı oluşturur ve indekslemeyi ARKA PLANDA başlatır.
    202 Accepted = "isteğini aldım, işleniyor" demek, "bitti" değil.
    """
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Desteklenmeyen dosya tipi: {suffix}. Desteklenenler: {ALLOWED_EXTENSIONS}",
        )

    try:
        settings.RAW_DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        destination = settings.RAW_DOCUMENTS_DIR / file.filename
        with destination.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as exc:
        logger.error(f"Dosya kaydedilirken hata oluştu: {exc}")
        raise HTTPException(status_code=500, detail="Dosya sunucuya kaydedilemedi.") from exc

    document_id = document_repository.create_document(
        filename=file.filename, file_path=str(destination)
    )

    background_tasks.add_task(
        ingestion_service.process_document,
        document_id=document_id,
        file_path=destination,
        filename=file.filename,
    )

    return DocumentUploadResponse(
        document_id=document_id,
        filename=file.filename,
        status="pending",
        message="Doküman alındı, işleniyor. Durumu /documents/{id} ile takip edebilirsin.",
    )


@router.get("/{document_id}", response_model=DocumentStatusResponse)
async def get_document_status(document_id: str):
    """Belirli bir dokümanın işlenme durumunu döner."""
    document = document_repository.get_document(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Doküman bulunamadı.")
    return DocumentStatusResponse(**document)


@router.get("/", response_model=DocumentListResponse)
async def list_documents():
    """Yüklenmiş tüm dokümanları listeler."""
    documents = document_repository.list_documents()
    return DocumentListResponse(
        documents=[DocumentStatusResponse(**doc) for doc in documents], total=len(documents)
    )


@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """Bir dokümanı ve ona ait tüm vektör/graf verisini kalıcı olarak siler."""
    deleted = ingestion_service.delete_document(document_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Doküman bulunamadı.")
    return {"message": "Doküman silindi.", "document_id": document_id}