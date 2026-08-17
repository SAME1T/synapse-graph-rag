"""
TODO: Metinden varlık (entity) ve ilişki (relationship) çıkaran modül.
Extraction işlemi için, generation'dan FARKLI ve daha güçlü bir model kullanılır
(settings.EXTRACTION_MODEL_ALIAS) - çünkü yapılandırılmış JSON çıkarımı, serbest
metin üretiminden daha fazla instruction-following hassasiyeti gerektirir.
Küçük modeller bazen geçersiz JSON üretebildiği için, parse hatası durumunda tüm süreci
durdurmak yerine boş bir sonuç döner ve durumu loglar.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import List

from app.core.config import settings
from app.core.llm_manager import llm_manager
from app.core.prompts import GRAPH_EXTRACTION_SYSTEM_PROMPT, build_extraction_prompt

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    name: str
    type: str


@dataclass
class Relationship:
    subject: str
    predicate: str
    object: str


@dataclass
class ExtractionResult:
    entities: List[Entity] = field(default_factory=list)
    relationships: List[Relationship] = field(default_factory=list)


def extract_graph_elements(chunk_text: str) -> ExtractionResult:
    prompt = build_extraction_prompt(chunk_text)

    try:
        raw_response = llm_manager.generate(
            system_prompt=GRAPH_EXTRACTION_SYSTEM_PROMPT,
            user_prompt=prompt,
            model_alias=settings.EXTRACTION_MODEL_ALIAS,
        )
    except Exception as exc:
        logger.error(f"Graf çıkarımı için LLM çağrısı başarısız oldu: {exc}")
        return ExtractionResult()

    return _parse_extraction_response(raw_response)


def _parse_extraction_response(raw_response: str) -> ExtractionResult:
    json_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
    if not json_match:
        logger.warning("LLM cevabında JSON bulunamadı, boş sonuç dönülüyor.")
        return ExtractionResult()

    try:
        data = json.loads(json_match.group(0))
    except json.JSONDecodeError as exc:
        logger.warning(f"LLM'in ürettiği JSON parse edilemedi: {exc}")
        return ExtractionResult()

    entities = [
        Entity(name=e.get("name", "").strip(), type=e.get("type", "unknown").strip())
        for e in data.get("entities", [])
        if e.get("name")
    ]
    relationships = [
        Relationship(
            subject=r.get("subject", "").strip(),
            predicate=r.get("predicate", "").strip(),
            object=r.get("object", "").strip(),
        )
        for r in data.get("relationships", [])
        if r.get("subject") and r.get("object")
    ]

    return ExtractionResult(entities=entities, relationships=relationships)
