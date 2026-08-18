"""
TODO: Metinden varlık (entity) ve ilişki (relationship) çıkaran modül.
Extraction işlemi için, generation'dan FARKLI ve daha güçlü bir model kullanılır.

İKİ SAVUNMA KATMANI:
1. LLM çağrısı başarısız olursa artık hata YUTULMUYOR, yukarıya fırlatılıyor -
   böylece extraction_service'teki "başarısız chunk" sayacı gerçeği yansıtıyor.
   (Önceki haliyle, sessiz başarısızlıklar "0 başarısız chunk" diye raporlanıyordu - yanıltıcıydı.)
2. Parse edilen entity/relationship alanları, kelime sayısı eşiğiyle doğrulanıyor -
   modelin bazen tam bir cümleyi "varlık" sanıp JSON'a koyduğu durumlar (gerçek bir dokümanla
   test ederken gözlemlendi) grafa hiç yazılmadan eleniyor. Bu KESİN bir çözüm değil,
   ucuz bir sezgisel filtre - ama gerçek dokümanla test edilip etkisi ölçülecek.
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

MAX_ENTITY_NAME_WORDS = 6  # bu eşiğin üstündeki bir "varlık adı" muhtemelen bir cümle parçasıdır


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
    discarded_count: int = 0  # doğrulamadan geçemeyip elenen öğe sayısı - şeffaflık için


def extract_graph_elements(chunk_text: str) -> ExtractionResult:
    prompt = build_extraction_prompt(chunk_text)

    # LLM çağrısı burada bilerek try/except İÇİNDE DEĞİL - başarısız olursa hata
    # yukarıya (extraction_service'e) iletilsin ki gerçek bir "başarısız chunk" olarak sayılsın.
    raw_response = llm_manager.generate(
        system_prompt=GRAPH_EXTRACTION_SYSTEM_PROMPT,
        user_prompt=prompt,
        model_alias=settings.EXTRACTION_MODEL_ALIAS,
    )

    return _parse_extraction_response(raw_response)


def _is_valid_entity_text(text: str) -> bool:
    """
    Basit bir sezgisel: bir 'varlık adı' ya da ilişki ucu, MAX_ENTITY_NAME_WORDS'ten
    fazla kelime içeriyorsa muhtemelen model bir cümleyi/açıklamayı varlık sanmıştır.
    Bu KESİN bir dilbilimsel çözüm değil, ama gerçek dokümanlarda gözlemlenen
    "tam cümle varlık" hatasını ucuza büyük ölçüde eler.
    """
    if not text:
        return False
    return len(text.split()) <= MAX_ENTITY_NAME_WORDS


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

    discarded = 0

    raw_entities = data.get("entities", [])
    entities = []
    for e in raw_entities:
        name = _safe_str(e.get("name", ""))
        if name and _is_valid_entity_text(name):
            entities.append(Entity(name=name, type=_safe_str(e.get("type", "unknown"))))
        elif name:
            discarded += 1
            logger.info(f"Geçersiz (muhtemelen cümle parçası) varlık elendi: '{name[:60]}...'")

    raw_relationships = data.get("relationships", [])
    relationships = []
    for r in raw_relationships:
        subject = _safe_str(r.get("subject", ""))
        obj = _safe_str(r.get("object", ""))
        if subject and obj and _is_valid_entity_text(subject) and _is_valid_entity_text(obj):
            relationships.append(
                Relationship(subject=subject, predicate=_safe_str(r.get("predicate", "")), object=obj)
            )
        elif subject and obj:
            discarded += 1
            logger.info(f"Geçersiz (muhtemelen cümle parçası) ilişki elendi: '{subject[:40]}' -> '{obj[:40]}'")

    return ExtractionResult(entities=entities, relationships=relationships, discarded_count=discarded)


def _safe_str(value) -> str:
    """
    LLM bazen bir alanı (özellikle 'X ve Y' gibi bileşik ifadelerde) liste olarak
    dönebiliyor - beklenen string yerine. Bu fonksiyon her türlü tipi güvenle
    string'e çevirir, böylece .strip() gibi metodlar asla çökmez.
    """
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return ", ".join(_safe_str(v) for v in value if v)
    if value is None:
        return ""
    return str(value).strip()
