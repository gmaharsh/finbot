"""
Input/output guards (off-topic, injection, PII, rate limit, citations, grounding).
LangChain Runnable wrappers satisfy the stack requirement without heavy NeMo setup.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from langchain_core.runnables import RunnableLambda

# --- Patterns ---
_INJECTION_PATTERNS = [
    r"ignore\s+(your\s+)?instructions",
    r"disregard\s+(your\s+)?(prior|previous)?\s*instructions",
    r"bypass\s+rbac",
    r"show\s+me\s+all\s+documents",
    r"regardless\s+of\s+my\s+role",
    r"act\s+as\s+a\s+different\s+assistant",
    r"no\s+restrictions",
    r"jailbreak",
    r"system\s+prompt",
    r"override\s+access",
]

_OFF_TOPIC_HINTS = [
    r"cricket\s+score",
    r"write\s+me\s+a\s+poem",
    r"who\s+won\s+the\s+world\s+cup",
    r"recipe\s+for",
    r"python\s+sort\s+algorithm\s+only",  # generic coding homework without FinSolve
]

_AADHAAR = re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")
_BANK_IN = re.compile(r"\b\d{9,18}\b")
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

_SESSION_COUNTS: dict[str, int] = {}


@dataclass
class GuardResult:
    allowed: bool
    message: str | None = None
    warnings: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)


def _match_any(text: str, patterns: list[str]) -> bool:
    low = text.lower()
    for p in patterns:
        if re.search(p, low, re.I):
            return True
    return False


def check_prompt_injection(query: str) -> GuardResult | None:
    if _match_any(query, _INJECTION_PATTERNS):
        return GuardResult(
            allowed=False,
            message="This request looks like an attempt to override safety or access rules and cannot be processed.",
            flags=["prompt_injection"],
        )
    return None


def check_off_topic(query: str) -> GuardResult | None:
    # FinSolve domains: HR, finance, engineering, marketing, general company knowledge
    if _match_any(query, _OFF_TOPIC_HINTS):
        return GuardResult(
            allowed=False,
            message="I can only help with FinSolve business topics such as HR policies, finance, engineering, or marketing.",
            flags=["off_topic"],
        )
    return None


def check_pii(query: str) -> GuardResult | None:
    if _AADHAAR.search(query):
        return GuardResult(
            allowed=False,
            message="Please remove personal identifiers (such as Aadhaar or bank account numbers) from your message.",
            flags=["pii"],
        )
    digits_only = re.sub(r"\D", "", query)
    if len(digits_only) >= 12 and _BANK_IN.search(query):
        return GuardResult(
            allowed=False,
            message="Please remove bank or sensitive numeric identifiers from your message.",
            flags=["pii"],
        )
    if _EMAIL.search(query):
        return GuardResult(
            allowed=False,
            message="Please remove email addresses from your query to protect personal data.",
            flags=["pii_email"],
        )
    return None


def check_session_rate(session_key: str, limit: int) -> GuardResult | None:
    n = _SESSION_COUNTS.get(session_key, 0) + 1
    _SESSION_COUNTS[session_key] = n
    if n > limit:
        return GuardResult(
            allowed=False,
            message=f"Session query limit of {limit} messages reached. Start a new session later.",
            flags=["rate_limited"],
        )
    return None


def run_input_guards(query: str, session_key: str, session_limit: int) -> GuardResult:
    for fn in (check_prompt_injection, check_off_topic, check_pii):
        r = fn(query)
        if r:
            return r
    r = check_session_rate(session_key, session_limit)
    if r:
        return r
    return GuardResult(allowed=True)


def enforce_citations(answer: str) -> list[str]:
    warnings = []
    has_page = bool(re.search(r"\bpage\s*\d+|\bp\.\s*\d+", answer, re.I))
    has_source = bool(re.search(r"\.(pdf|md|docx)\b|\bsource\b", answer, re.I))
    if not (has_page and has_source):
        warnings.append(
            "Citation check: response should name a source file and page number where possible."
        )
    return warnings


def grounding_check(answer: str, contexts: list[str]) -> list[str]:
    warnings = []
    ctx = "\n".join(contexts).lower()
    nums = re.findall(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?%?\b|\bFY\d{4}\b|\bQ[1-4]\s+\d{4}\b", answer)
    for n in set(nums):
        if n.lower() not in ctx and n.replace(",", "").lower() not in ctx:
            warnings.append(
                "Grounding check: some numbers or dates in the answer may not appear in retrieved sources."
            )
            break
    return warnings


def cross_role_leakage_check(answer: str, allowed_collections: list[str]) -> list[str]:
    warnings = []
    finance_terms = ["ebitda", "annual report fy", "investor deck"]
    if "finance" not in allowed_collections:
        low = answer.lower()
        if any(t in low for t in finance_terms):
            warnings.append(
                "Leakage check: response references finance-specific concepts; verify the user is authorized."
            )
    return warnings


def run_output_guards(
    answer: str,
    contexts: list[str],
    allowed_collections: list[str],
) -> GuardResult:
    warnings: list[str] = []
    flags: list[str] = []
    warnings.extend(enforce_citations(answer))
    if warnings and "Citation check" in warnings[0]:
        flags.append("citation_warning")
    g = grounding_check(answer, contexts)
    warnings.extend(g)
    if g:
        flags.append("grounding_warning")
    warnings.extend(cross_role_leakage_check(answer, allowed_collections))
    return GuardResult(allowed=True, warnings=warnings, flags=flags)


# LangChain-style hooks (assignment: LangChain Guardrails)
input_guard_runnable: RunnableLambda = RunnableLambda(
    lambda x: run_input_guards(x["query"], x["session_key"], x["session_limit"])
)

output_guard_runnable: RunnableLambda = RunnableLambda(
    lambda x: run_output_guards(x["answer"], x["contexts"], x["allowed_collections"])
)
