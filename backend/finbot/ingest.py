"""
Ingest documents with Docling HybridChunker and upsert to Qdrant with RBAC metadata.
"""

from __future__ import annotations

import argparse
import logging
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter

from finbot.access_matrix import access_roles_for_collection
from finbot.embeddings import embed_texts
from finbot.qdrant_store import delete_by_source_document, upsert_points
from finbot.settings import get_settings

logger = logging.getLogger(__name__)


def _page_from_meta(meta: Any) -> int:
    if meta is None:
        return 1
    items = getattr(meta, "doc_items", None) or []
    for it in items:
        prov = getattr(it, "prov", None)
        if prov is None and hasattr(it, "meta") and getattr(it.meta, "doc_items", None):
            continue
        if prov is not None:
            for p in prov if isinstance(prov, (list, tuple)) else [prov]:
                pn = getattr(p, "page_no", None)
                if pn is not None:
                    return int(pn)
    return 1


def _headings_from_meta(meta: Any) -> list[str]:
    if meta is None:
        return []
    h = getattr(meta, "headings", None)
    if h:
        return [str(x) for x in h]
    return []


def _infer_chunk_type(text: str, meta: Any) -> str:
    t = text.strip()
    if not t:
        return "text"
    low = t.lower()
    items = getattr(meta, "doc_items", None) or [] if meta else []
    for it in items:
        name = type(it).__name__.lower()
        if "table" in name:
            return "table"
        if "code" in name:
            return "code"
    if t.startswith("|") and "|" in t[1:]:
        return "table"
    if t.startswith("```") or "```" in t:
        return "code"
    if len(t) < 200 and not t.endswith(".") and "\n" not in t and t.isupper():
        return "heading"
    return "text"


def _build_chunker() -> HybridChunker:
    # Align token limits with embedding model when possible (optional).
    return HybridChunker()


def parse_and_chunk(path: Path) -> list[dict[str, Any]]:
    converter = DocumentConverter()
    result = converter.convert(source=str(path))
    dl_doc = result.document
    chunker = _build_chunker()
    rows: list[dict[str, Any]] = []
    for ch in chunker.chunk(dl_doc=dl_doc):
        meta = getattr(ch, "meta", None)
        text = (ch.text or "").strip()
        if not text:
            continue
        headings = _headings_from_meta(meta)
        section_title = " > ".join(headings) if headings else "(document)"
        page_number = _page_from_meta(meta)
        ctype = _infer_chunk_type(text, meta)
        ctx = ""
        try:
            ctx = chunker.contextualize(chunk=ch)
        except Exception:
            ctx = text
        rows.append(
            {
                "text": text,
                "embed_text": ctx or text,
                "section_title": section_title,
                "page_number": page_number,
                "chunk_type": ctype,
            }
        )
    return rows


def ingest_file(
    file_path: Path,
    collection: str,
    access_roles: list[str] | None = None,
) -> int:
    access_roles = access_roles or access_roles_for_collection(collection)
    source_document = file_path.name
    delete_by_source_document(source_document)

    leaf_rows = parse_and_chunk(file_path)
    if not leaf_rows:
        logger.warning("No chunks for %s", file_path)
        return 0

    by_section: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in leaf_rows:
        by_section[r["section_title"]].append(r)

    parent_ids: dict[str, str] = {}
    parent_points: list[dict[str, Any]] = []
    for section_title, secs in by_section.items():
        pid = str(uuid.uuid4())
        parent_ids[section_title] = pid
        preview = "\n".join(s["text"][:400] for s in secs[:3])
        parent_text = f"Section: {section_title}\nSummary: {preview[:1200]}"
        parent_points.append(
            {
                "id": pid,
                "vector": None,  # fill later
                "payload": {
                    "content": parent_text,
                    "source_document": source_document,
                    "collection": collection,
                    "access_roles": access_roles,
                    "section_title": section_title,
                    "page_number": min(s["page_number"] for s in secs),
                    "chunk_type": "heading",
                    "parent_chunk_id": None,
                },
            }
        )

    leaf_points: list[dict[str, Any]] = []
    for r in leaf_rows:
        pid = parent_ids[r["section_title"]]
        leaf_points.append(
            {
                "id": str(uuid.uuid4()),
                "vector": None,
                "payload": {
                    "content": r["text"],
                    "source_document": source_document,
                    "collection": collection,
                    "access_roles": access_roles,
                    "section_title": r["section_title"],
                    "page_number": r["page_number"],
                    "chunk_type": r["chunk_type"],
                    "parent_chunk_id": pid,
                },
                "_embed": r["embed_text"],
            }
        )

    all_embed_texts = [p["payload"]["content"] for p in parent_points] + [p["_embed"] for p in leaf_points]
    vectors = embed_texts(all_embed_texts)
    if len(vectors) != len(all_embed_texts):
        raise RuntimeError("Embedding batch size mismatch")

    i = 0
    for p in parent_points:
        p["vector"] = vectors[i]
        i += 1
    for p in leaf_points:
        p["vector"] = vectors[i]
        i += 1
        del p["_embed"]

    upsert_points(parent_points + leaf_points)
    return len(parent_points) + len(leaf_points)


def discover_data_files(root: Path) -> list[tuple[Path, str]]:
    """Map path -> collection from folder name."""
    out: list[tuple[Path, str]] = []
    for sub in ("general", "finance", "engineering", "marketing"):
        d = root / sub
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("*")):
            if p.is_file() and p.suffix.lower() in {".md", ".markdown", ".pdf", ".docx", ".html"}:
                out.append((p, sub))
    return out


def ingest_all_data_dir() -> None:
    s = get_settings()
    root = Path(s.data_dir).resolve()
    pairs = discover_data_files(root)
    for path, collection in pairs:
        logger.info("Ingesting %s as %s", path, collection)
        ingest_file(path, collection)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="FinBot Docling ingest")
    parser.add_argument("--all", action="store_true", help="Ingest all files under data_dir subfolders")
    parser.add_argument("--file", type=str, help="Single file path")
    parser.add_argument("--collection", type=str, default="general", help="Collection for --file")
    args = parser.parse_args()
    if args.all:
        ingest_all_data_dir()
    elif args.file:
        ingest_file(Path(args.file), args.collection)
    else:
        parser.error("Provide --all or --file")


if __name__ == "__main__":
    main()
