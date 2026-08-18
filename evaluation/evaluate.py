"""
TODO: Sistemin uçtan uca (black-box) başarı oranını ölçen değerlendirme scripti.
API'ye gerçek bir kullanıcı gibi HTTP isteği atar, cevapları beklenen anahtar
kelimelerle karşılaştırır. Kodun iç mantığını değil, kullanıcının gerçekte
aldığı sonucu test eder.

ÇALIŞTIRMADAN ÖNCE: Sunucunun ayakta olması ve test edilecek dokümanın
(örn. canakkale.txt) zaten yüklenmiş olması gerekir.

Kullanım: python -m evaluation.evaluate
"""

import json
import sys
from pathlib import Path

import httpx

from app.core.config import settings

QUESTIONS_PATH = Path(__file__).parent / "test_questions.json"
RESULTS_PATH = Path(__file__).parent / "results.json"


def load_questions() -> list[dict]:
    with QUESTIONS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def ask_question(client: httpx.Client, question: str) -> dict:
    response = client.post(
        f"{settings.EVALUATION_API_BASE_URL}{settings.API_V1_PREFIX}/query/ask",
        json={"query": question},
        timeout=120.0,  # extraction/generation yavaş olabiliyor, geniş tuttuk
    )
    response.raise_for_status()
    return response.json()


def evaluate_answer(test_case: dict, api_response: dict) -> dict:
    """
    Tek bir test sorusunun geçip geçmediğini belirler.
    should_have_answer=True  -> LLM kullanılmalı VE en az bir anahtar kelime cevapta geçmeli.
    should_have_answer=False -> LLM kullanılmamalı (fallback dönmeli) - ya da kullanıldıysa
                                 bile beklenen bir keyword olmamalı (q15 gibi durumlar için).
    """
    answer_lower = api_response["answer"].lower()
    keyword_hit = any(kw.lower() in answer_lower for kw in test_case["expected_keywords"])

    if test_case["should_have_answer"]:
        passed = api_response["used_llm"] and keyword_hit
    else:
        passed = not api_response["used_llm"] or not keyword_hit

    return {
        "id": test_case["id"],
        "category": test_case["category"],
        "question": test_case["question"],
        "passed": passed,
        "used_llm": api_response["used_llm"],
        "groundedness_score": api_response["groundedness_score"],
        "keyword_hit": keyword_hit,
        "answer_preview": api_response["answer"][:150],
    }


def print_report(results: list[dict]) -> None:
    print("\n" + "=" * 90)
    print(f"{'ID':<5}{'Kategori':<15}{'Sonuç':<8}{'Groundedness':<14}{'Soru'}")
    print("=" * 90)

    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(
            f"{r['id']:<5}{r['category']:<15}{status:<8}"
            f"{r['groundedness_score']:<14.2f}{r['question'][:50]}"
        )

    print("=" * 90)

    total = len(results)
    passed = sum(1 for r in results if r["passed"])

    print(f"\nGENEL: {passed}/{total} test geçti ({passed / total * 100:.1f}%)\n")

    categories = sorted(set(r["category"] for r in results))
    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        cat_passed = sum(1 for r in cat_results if r["passed"])
        print(f"  {cat:<15}: {cat_passed}/{len(cat_results)} geçti")

    answered = [r for r in results if r["used_llm"]]
    if answered:
        avg_groundedness = sum(r["groundedness_score"] for r in answered) / len(answered)
        print(f"\nOrtalama groundedness skoru (cevaplanan sorular): {avg_groundedness:.3f}")

    print()


def main() -> None:
    questions = load_questions()
    results = []

    with httpx.Client() as client:
        for i, test_case in enumerate(questions, 1):
            print(f"[{i}/{len(questions)}] Soruluyor: {test_case['question'][:60]}...")
            try:
                api_response = ask_question(client, test_case["question"])
            except Exception as exc:
                print(f"  HATA: {exc}", file=sys.stderr)
                continue

            result = evaluate_answer(test_case, api_response)
            results.append(result)

    print_report(results)

    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Detaylı sonuçlar kaydedildi: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
