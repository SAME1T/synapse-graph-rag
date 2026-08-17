"""
TODO: LLM cevabının kalitesini ve kaynaklara dayanıklılığını (groundedness) kontrol eden modül.
Ekstra bir LLM çağrısı yapmadan, hızlı sezgisel (heuristic) kontrollerle çalışır.
Bu KESİN bir doğruluk kanıtı değildir - kelime örtüşmesi anlam örtüşmesiyle aynı şey değildir,
ama LLM'in kaynaklardan tamamen kopup kendi bilgisini uydurduğu bariz durumları
yakalamakta ucuz ve hızlı bir ilk savunma hattı olarak işe yarar.
"""

import re
from dataclasses import dataclass
from typing import List

MIN_ANSWER_LENGTH = 15
MIN_GROUNDEDNESS_SCORE = 0.25  # cevaptaki anlamlı kelimelerin en az %25'i kaynaklarda geçmeli

REFUSAL_PHRASES = [
    "yeterli bilgi bulunmuyor",
    "yeterli bilgi yok",
    "kaynaklarda bu soruyu",
]


@dataclass
class QualityCheckResult:
    is_acceptable: bool
    reason: str
    groundedness_score: float


def clean_answer_formatting(answer: str) -> str:
    """
    Küçük modellerin sık ürettiği gereksiz markdown başlıklarını (**Sonuç:** gibi)
    ve fazla boş satırları temizler. İçeriği DEĞİŞTİRMEZ, sadece görünümü sadeleştirir.
    """
    cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", answer)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _tokenize(text: str) -> set:
    """Basit kelime ayırıcı - küçük harfe çevirip kısa/anlamsız kelimeleri (ve, ile, bu vb.) eler."""
    words = re.findall(r"[a-zA-ZçğıöşüÇĞİÖŞÜ0-9]+", text.lower())
    return {w for w in words if len(w) > 3}


def calculate_groundedness(answer: str, source_texts: List[str]) -> float:
    """
    Cevaptaki anlamlı kelimelerin ne kadarının kaynak metinlerde de geçtiğini ölçer (0-1 arası).
    """
    answer_words = _tokenize(answer)
    if not answer_words:
        return 0.0

    source_words = set()
    for text in source_texts:
        source_words |= _tokenize(text)

    overlapping = answer_words & source_words
    return len(overlapping) / len(answer_words)


def check_answer_quality(answer: str, source_texts: List[str]) -> QualityCheckResult:
    """
    Cevabı hızlı kontrollerden geçirir. is_acceptable=False dönerse,
    generation_service bu durumda fallback mesajına düşer.
    """
    if not answer or len(answer.strip()) < MIN_ANSWER_LENGTH:
        return QualityCheckResult(False, "Cevap çok kısa veya boş.", 0.0)

    lowered = answer.lower()
    if any(phrase in lowered for phrase in REFUSAL_PHRASES):
        # Model kendisi "yeterli bilgi yok" demişse, bu bir HATA değil, doğru bir davranıştır.
        return QualityCheckResult(True, "Model bilinçli olarak bilgi yetersizliğini belirtti.", 1.0)

    score = calculate_groundedness(answer, source_texts)
    if score < MIN_GROUNDEDNESS_SCORE:
        return QualityCheckResult(
            False,
            f"Groundedness skoru çok düşük ({score:.2f}) - cevap kaynaklardan kopmuş olabilir.",
            score,
        )

    return QualityCheckResult(True, "Cevap kabul edildi.", score)
