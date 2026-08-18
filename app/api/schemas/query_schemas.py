"""
TODO: Soru sorma ve arama sonucu cevaplarının veri şekillerini tanımlar.
"""

from typing import List, Optional

from pydantic import BaseModel


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    bi_encoder_score: float
    rerank_score: float
    filename: str


class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = None


class SearchResponse(BaseModel):
    query: str
    results: List[RetrievedChunk]
    sufficient_context: bool


class SourceReference(BaseModel):
    chunk_id: str
    filename: str
    rerank_score: float


class GraphFactReference(BaseModel):
    subject: str
    predicate: str
    object: str


class AskRequest(BaseModel):
    query: str
    top_k: Optional[int] = None


class AskResponse(BaseModel):
    query: str
    answer: str
    used_llm: bool
    is_fallback: bool
    groundedness_score: float
    sources: List[SourceReference]
    graph_facts_used: List[GraphFactReference]