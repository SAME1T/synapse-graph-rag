"""
TODO: Soru sorma / arama API endpoint'lerini tanımlar.
"""

import logging

from fastapi import APIRouter, HTTPException

from app.api.schemas.query_schemas import (
    AskRequest,
    AskResponse,
    GraphFactReference,
    RetrievedChunk,
    SearchRequest,
    SearchResponse,
    SourceReference,
)
from app.core.config import settings
from app.services.generation_service import generation_service
from app.services.retrieval_service import retrieval_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/query", tags=["Query"])


@router.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    try:
        top_k = request.top_k or settings.TOP_K
        results = retrieval_service.retrieve(request.query, top_k=top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return SearchResponse(
        query=request.query,
        results=[
            RetrievedChunk(
                chunk_id=r["chunk_id"],
                text=r["text"],
                bi_encoder_score=r["score"],
                rerank_score=r["rerank_score"],
                filename=r["metadata"].get("filename", "bilinmiyor"),
            )
            for r in results
        ],
        sufficient_context=len(results) > 0,
    )


@router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    Tam GraphRAG akışı: vektör+rerank retrieval + graf ilişki araması + generation.
    İkisinden de hiçbir şey bulunamazsa LLM çağrılmaz.
    """
    try:
        top_k = request.top_k or settings.TOP_K
        retrieved_chunks = retrieval_service.retrieve(request.query, top_k=top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    graph_facts = retrieval_service.retrieve_graph_facts(request.query)

    result = generation_service.generate_answer(request.query, retrieved_chunks, graph_facts)

    return AskResponse(
        query=request.query,
        answer=result["answer"],
        used_llm=result["used_llm"],
        is_fallback=result["is_fallback"],
        groundedness_score=result["groundedness_score"],
        sources=[SourceReference(**s) for s in result["sources"]],
        graph_facts_used=[GraphFactReference(**f) for f in result["graph_facts_used"]],
    )
    