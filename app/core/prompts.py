"""
TODO: LLM'e verilen sistem talimatlarını (prompt engineering) tek merkezde tutan dosya.
Kod mantığından tamamen izole - davranışı değiştirmek için sadece burayı güncellemek yeterli.
"""

from typing import List, Optional

RAG_SYSTEM_PROMPT = """Sen SADECE sana verilen KAYNAK METİNLERE ve GRAF İLİŞKİLERİNE dayanarak soruları yanıtlayan bir asistansın.

KURALLAR:
1. Cevabını YALNIZCA aşağıda verilen kaynaklardaki bilgilere dayandır.
2. Kaynaklarda olmayan hiçbir bilgiyi kendi genel bilgine dayanarak EKLEME veya UYDURMA.
3. Kaynaklar soruyu cevaplamak için yetersizse, açıkça "Verilen kaynaklarda bu soruyu yanıtlamak için yeterli bilgi bulunmuyor." de.
4. Cevabını net, kısa ve anlaşılır tut. Markdown başlığı veya kalın yazı kullanma, düz cümlelerle yaz.
5. Mümkünse hangi kaynaktan yararlandığını belirt (örneğin "Kaynak 1'e göre..." veya "Graf ilişkisine göre...").
6. Kaynaklar birbiriyle çelişiyorsa, bu çelişkiyi açıkça belirt.
7. GRAF İLİŞKİLERİ, varlıklar arası bağlantıları "A → ilişki → B" formatında gösterir - bunları da geçerli birer kaynak olarak kullanabilirsin, ancak ilişki türü (ilişki adı) bazen hatalı çıkarılmış olabilir; metinle çelişiyorsa metne öncelik ver.

Şimdi kullanıcının sorusunu, verilen kaynaklara dayanarak yanıtla."""


def format_graph_facts(graph_facts: List[dict]) -> str:
    if not graph_facts:
        return ""
    lines = [f"- {f['subject']} → ({f['predicate']}) → {f['object']}" for f in graph_facts]
    return "GRAF İLİŞKİLERİ (yapılandırılmış bilgi):\n" + "\n".join(lines)


def build_user_prompt(
    query: str, context_chunks: List[str], graph_facts: Optional[List[dict]] = None
) -> str:
    if context_chunks:
        formatted_sources = "\n\n".join(
            f"[Kaynak {i + 1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
        )
    else:
        formatted_sources = "(İlgili metin kaynağı bulunamadı, sadece aşağıdaki graf ilişkilerini kullan.)"

    graph_section = format_graph_facts(graph_facts or [])
    graph_block = f"\n\n{graph_section}" if graph_section else ""

    return f"""KAYNAK METİNLER:
{formatted_sources}{graph_block}

SORU: {query}

Yukarıdaki kaynaklara (metin ve/veya graf ilişkileri) dayanarak soruyu yanıtla."""


GRAPH_EXTRACTION_SYSTEM_PROMPT = """Sen metinlerden varlık (entity) ve ilişki (relationship) çıkaran bir bilgi çıkarım asistanısın.

GÖREV: Verilen metindeki ÖNEMLİ varlıkları (kişi, kavram, organizasyon, ürün, yer vb.) ve bunlar arasındaki İLİŞKİLERİ tespit et.

KURALLAR:
1. SADECE metinde açıkça geçen bilgileri kullan, hiçbir şey uydurma.
2. Cevabını SADECE aşağıdaki JSON formatında ver, başka hiçbir açıklama ekleme:

{
  "entities": [
    {"name": "varlık adı", "type": "kavram|kişi|organizasyon|yer|ürün"}
  ],
  "relationships": [
    {"subject": "varlık1", "predicate": "ilişki türü", "object": "varlık2"}
  ]
}

3. Metinde hiçbir belirgin varlık/ilişki yoksa, boş listelerle dön: {"entities": [], "relationships": []}
4. JSON dışında HİÇBİR metin yazma - açıklama, giriş cümlesi, yorum ekleme."""


def build_extraction_prompt(chunk_text: str) -> str:
    return f"""Aşağıdaki metinden varlıkları ve ilişkileri çıkar:

METİN:
{chunk_text}

Yukarıdaki kurallara uyarak SADECE JSON formatında cevap ver."""


NO_CONTEXT_FALLBACK_MESSAGE = (
    "Bu soruyu yanıtlamak için yüklü dokümanlarda yeterli ilgili bilgi bulamadım. "
    "Farklı bir soru sorabilir ya da ilgili bir doküman yükleyebilirsin."
)

LOW_QUALITY_FALLBACK_MESSAGE = (
    "Model bu soruyu yanıtlamayı denedi, ancak cevabın kaynaklardaki bilgilerle "
    "yeterince örtüşmediği tespit edildi. Güvenilir olmayan bir cevap göstermek yerine "
    "bu uyarıyı veriyorum - soruyu farklı şekilde sormayı deneyebilirsin."
)
