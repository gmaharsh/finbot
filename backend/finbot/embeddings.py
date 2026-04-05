from openai import OpenAI

from finbot.settings import get_settings


def _client() -> OpenAI:
    s = get_settings()
    kwargs = {"api_key": s.openai_api_key or None}
    if s.openai_base_url:
        kwargs["base_url"] = s.openai_base_url
    return OpenAI(**kwargs)


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    s = get_settings()
    client = _client()
    resp = client.embeddings.create(model=s.embedding_model, input=texts)
    return [d.embedding for d in sorted(resp.data, key=lambda x: x.index)]
