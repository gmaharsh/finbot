"""Orchestrates guards, semantic router, RBAC intersection, retrieval, LLM, output guards."""

from __future__ import annotations

import logging
from typing import Any

from finbot.access_matrix import collections_for_role
from finbot.guardrails import run_input_guards, run_output_guards
from finbot.llm import answer_from_context
from finbot.qdrant_store import search_filtered
from finbot.router_semantic import ROUTE_TO_COLLECTIONS, classify_route
from finbot.settings import get_settings

logger = logging.getLogger(__name__)

RBAC_REFUSAL = (
    "You do not have access to the document collections needed to answer this question. "
    "Please contact your administrator if you believe this is an error."
)


def _intersect_collections(route_name: str, role: str) -> tuple[list[str], list[str]]:
    """Returns (target_collections_after_intersection, user_allowed_collections)."""
    user_cols = collections_for_role(role)
    route_cols = ROUTE_TO_COLLECTIONS.get(route_name, ROUTE_TO_COLLECTIONS["cross_department_route"])
    targeted = [c for c in route_cols if c in user_cols]
    return targeted, user_cols


def process_chat(
    *,
    query: str,
    role: str,
    session_key: str,
) -> dict[str, Any]:
    s = get_settings()
    out: dict[str, Any] = {
        "answer": "",
        "sources": [],
        "route": None,
        "collections_accessible": collections_for_role(role),
        "target_collections": [],
        "blocked": False,
        "block_reason": None,
        "guardrail_flags": [],
        "guardrail_warnings": [],
    }

    g_in = run_input_guards(query, session_key, s.session_query_limit)
    if not g_in.allowed:
        out["blocked"] = True
        out["block_reason"] = g_in.message
        out["guardrail_flags"] = g_in.flags
        return out

    route = classify_route(query)
    targeted, user_cols = _intersect_collections(route, role)
    out["route"] = route
    out["target_collections"] = targeted

    logger.info(
        "chat_audit",
        extra={
            "user_role": role,
            "route_name": route,
            "allowed_collections_after_intersection": targeted,
            "session_key": session_key,
        },
    )

    if not targeted:
        out["blocked"] = True
        out["block_reason"] = RBAC_REFUSAL
        out["guardrail_flags"].append("rbac_route_denied")
        return out

    chunks = search_filtered(query, targeted, limit=8)
    if not chunks:
        out["answer"] = (
            "No matching internal documents were found for your question within your accessible collections."
        )
        out["sources"] = []
        return out

    answer = answer_from_context(query, chunks)
    contexts = [c.get("content", "") for c in chunks]
    g_out = run_output_guards(answer, contexts, user_cols)
    out["answer"] = answer
    out["sources"] = [
        {
            "source_document": c.get("source_document"),
            "page_number": c.get("page_number"),
            "collection": c.get("collection"),
            "score": c.get("score"),
        }
        for c in chunks
    ]
    out["guardrail_warnings"] = g_out.warnings
    out["guardrail_flags"].extend(g_out.flags)
    return out
