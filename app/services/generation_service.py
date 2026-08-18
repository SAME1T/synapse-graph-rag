"""
TODO: Retrieval ile bulunan chunk'ları VE graf ilişkilerini context olarak kullanıp
LLM'den cevap üreten servis. RAG'in "Generation" kısmı. İki katmanlı groundedness
kontrolü içerir:
(1) Kaynak yoksa LLM hiç çağrılmaz.
(2) LLM cevap üretse bile, cevabın kaynaklarla örtüşmesi (answer_quality.py) kontrol
    edilir - düşük örtüşme varsa cevap gösterilmez, güvenli bir fallback'e düşülür.

is_fallback alanı: "used_llm" tek başına yanıltıcı olabiliyordu, çünkü LLM çağrılıp
kalite kontrolünden geçemeyen bir cevap da used_llm=True dönüyordu - arayüz bunu
"gerçek bir cevap" gibi gösterebiliyordu. is_fallback, kullanıcıya gösterilen
metnin bir ret/fallback mesajı mı yoksa gerçek bir cevap mı olduğunu net ayırır.
"""

import logging
from typing import List, Optional

from app.core.llm_manager import llm_manager
from app.core.prompts import (
    LOW_QUALITY_FALLBACK_MESSAGE,
    NO_CONTEXT_FALLBACK_MESSAGE,
    RAG_SYSTEM_PROMPT,
    build_user_prompt,
)
from app.rag.answer_quality import check_answer_quality, clean_answer_formatting

logger = logging.getLogger(__name__)


class GenerationService:
    def generate_answer(
        self,
        query: str,
        retrieved_chunks: List[dict],
        graph_facts: Optional[List[dict]] = None,
    ) -> dict:
        graph_facts = graph_facts or []

        if not retrieved_chunks and not graph_facts:
            logger.info(f"'{query}' için hiçbir kaynak (metin/graf) yok, fallback dönülüyor.")
            return {
                "answer": NO_CONTEXT_FALLBACK_MESSAGE,
                "used_llm": False,
                "is_fallback": True,
                "groundedness_score": 0.0,
                "sources": [],
                "graph_facts_used": [],
            }

        context_texts = [c["text"] for c in retrieved_chunks]
        user_prompt = build_user_prompt(query, context_texts, graph_facts)

        try:
            raw_answer = llm_manager.generate(
                system_prompt=RAG_SYSTEM_PROMPT, user_prompt=user_prompt
            )
        except Exception as exc:
            logger.error(f"LLM cevap üretemedi, fallback'e düşülüyor: {exc}")
            return {
                "answer": NO_CONTEXT_FALLBACK_MESSAGE,
                "used_llm": False,
                "is_fallback": True,
                "groundedness_score": 0.0,
                "sources": [],
                "graph_facts_used": [],
            }

        cleaned_answer = clean_answer_formatting(raw_answer)

        graph_facts_as_text = [f"{f['subject']} {f['predicate']} {f['object']}" for f in graph_facts]
        quality_result = check_answer_quality(cleaned_answer, context_texts + graph_facts_as_text)

        if not quality_result.is_acceptable:
            logger.warning(f"Cevap kalite kontrolünden geçemedi: {quality_result.reason}")
            return {
                "answer": LOW_QUALITY_FALLBACK_MESSAGE,
                "used_llm": True,
                "is_fallback": True,
                "groundedness_score": quality_result.groundedness_score,
                "sources": [],
                "graph_facts_used": [],
            }

        return {
            "answer": cleaned_answer,
            "used_llm": True,
            "is_fallback": False,
            "groundedness_score": quality_result.groundedness_score,
            "sources": [
                {
                    "chunk_id": c["chunk_id"],
                    "filename": c["metadata"].get("filename", "bilinmiyor"),
                    "rerank_score": c["rerank_score"],
                }
                for c in retrieved_chunks
            ],
            "graph_facts_used": graph_facts,
        }


generation_service = GenerationService()