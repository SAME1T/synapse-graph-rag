"""
TODO: Ham metni chunk'lara (parçalara) bölen modül.
Sabit karakter sayısına göre rastgele kesmek yerine, cümle sınırlarına saygı göstererek böler.
"""

import re
from dataclasses import dataclass
from typing import List

from app.core.config import settings


@dataclass
class Chunk:
    """Tek bir metin parçasını temsil eden basit veri yapısı."""
    text: str
    chunk_index: int
    char_count: int


def clean_text(text: str) -> str:
    """
    Fazla boşlukları/satır başlarını sadeleştirir.
    PDF'lerden çıkan metinlerde sık görülen bozuk boşluklama sorununu temizler.
    """
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_into_sentences(text: str) -> List[str]:
    """
    Basit bir cümle ayırıcı: nokta/ünlem/soru işaretinden sonra büyük harfle
    başlayan yeni cümleyi ayırır. Mükemmel değil ama chunk'ların cümle
    ortasından kesilmesini büyük oranda engeller.
    """
    sentence_endings = re.compile(r"(?<=[.!?])\s+(?=[A-ZÇĞİÖŞÜ0-9])")
    sentences = sentence_endings.split(text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(
    text: str,
    chunk_size: int = settings.CHUNK_SIZE,
    chunk_overlap: int = settings.CHUNK_OVERLAP,
) -> List[Chunk]:
    """
    Temizlenmiş metni, cümle sınırlarına saygı göstererek chunk_size civarında
    parçalara böler. Parçalar arasında chunk_overlap kadar örtüşme bırakır.
    """
    cleaned = clean_text(text)
    sentences = split_into_sentences(cleaned)

    chunks: List[Chunk] = []
    current_sentences: List[str] = []
    current_length = 0
    chunk_index = 0

    for sentence in sentences:
        sentence_length = len(sentence)

        # Bu cümleyi eklersek limiti aşacaksak - mevcut chunk'ı kapat
        if current_length + sentence_length > chunk_size and current_sentences:
            chunk_text_value = " ".join(current_sentences)
            chunks.append(
                Chunk(
                    text=chunk_text_value,
                    chunk_index=chunk_index,
                    char_count=len(chunk_text_value),
                )
            )
            chunk_index += 1

            # Overlap: yeni chunk'ı sıfırdan değil, öncekinin son cümleleriyle başlat
            current_sentences = _get_overlap_sentences(current_sentences, chunk_overlap)
            current_length = sum(len(s) for s in current_sentences)

        current_sentences.append(sentence)
        current_length += sentence_length

    # Döngü bitince elde kalan son parçayı da ekle
    if current_sentences:
        chunk_text_value = " ".join(current_sentences)
        chunks.append(
            Chunk(
                text=chunk_text_value,
                chunk_index=chunk_index,
                char_count=len(chunk_text_value),
            )
        )

    return chunks


def _get_overlap_sentences(sentences: List[str], overlap_chars: int) -> List[str]:
    """
    Önceki chunk'ın sonundan overlap_chars kadar karakter tutacak şekilde
    cümleleri sondan başa doğru toplar. Bunlar yeni chunk'ın başına eklenecek.
    """
    overlap_sentences = []
    accumulated = 0

    for sentence in reversed(sentences):
        if accumulated >= overlap_chars:
            break
        overlap_sentences.insert(0, sentence)
        accumulated += len(sentence)

    return overlap_sentences