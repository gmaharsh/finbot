from openai import OpenAI

from finbot.settings import get_settings


def _client() -> OpenAI:
    s = get_settings()
    kwargs = {"api_key": s.openai_api_key or None}
    if s.openai_base_url:
        kwargs["base_url"] = s.openai_base_url
    return OpenAI(**kwargs)


def answer_from_context(question: str, contexts: list[dict]) -> str:
    s = get_settings()
    blocks = []
    for i, c in enumerate(contexts, 1):
        src = c.get("source_document", "unknown")
        page = c.get("page_number", "?")
        blocks.append(f"[{i}] Source: {src} | Page: {page}\n{c.get('content', '')}")
    ctx = "\n\n".join(blocks)
    sys = (
        "You are FinBot, FinSolve's internal assistant. Answer ONLY using the provided context. "
        "If the answer is not in the context, say you do not have that information. "
        "Every factual claim must end with a citation like (Source: filename, Page: N). "
        "Do not invent figures."
    )
    user = f"Context:\n{ctx}\n\nQuestion: {question}\n"
    client = _client()
    r = client.chat.completions.create(
        model=s.chat_model,
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return (r.choices[0].message.content or "").strip()
