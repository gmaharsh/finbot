"""semantic-router: classify query intent, map to collections (intersect with role elsewhere)."""

from __future__ import annotations

import logging

from semantic_router import Route as SemanticRoute
from semantic_router.encoders import OpenAIEncoder
from semantic_router.layer import RouteLayer

from finbot.settings import get_settings

logger = logging.getLogger(__name__)

_finance_utterances = [
    "What was our revenue last quarter?",
    "Show me the FY2024 annual report summary",
    "How much did we allocate to R&D in the budget?",
    "What are the key investor risk factors?",
    "Explain EBITDA trends for FinSolve",
    "Q3 earnings highlights and margins",
    "Cash flow from operations this year",
    "Debt to equity ratio in the financial statements",
    "Capital expenditure plan for next fiscal year",
    "Dividend policy for shareholders",
    "Operating expenses breakdown by department",
    "Audit findings from the finance team",
    "Projected revenue growth for FY2025",
]

_engineering_utterances = [
    "How do I page the payments API?",
    "What is the system architecture for the ledger service?",
    "Runbook for database failover",
    "Incident severity levels and escalation",
    "Onboarding steps for new engineers",
    "Where is the API rate limit configured?",
    "How does the auth service validate JWTs?",
    "SLO targets for core services",
    "Postmortem template for outages",
    "How to rotate production secrets",
    "CI pipeline stages for backend",
    "Kubernetes namespace layout",
    "Debugging high latency in the gateway",
]

_marketing_utterances = [
    "Campaign performance for Q2 digital ads",
    "Brand voice and tone guidelines",
    "Competitor positioning in wealth management",
    "Market share analysis for our segment",
    "Social media content pillars",
    "Email nurture sequence metrics",
    "Creative approval workflow",
    "Persona for CFO buyers",
    "Launch checklist for product marketing",
    "SEO keyword strategy",
    "Partner co-marketing rules",
    "Press release approval process",
    "Customer testimonial usage policy",
]

_hr_general_utterances = [
    "What is the leave policy?",
    "How many annual leave days do employees get?",
    "Code of conduct expectations",
    "Remote work eligibility rules",
    "Expense reimbursement process",
    "Whistleblower reporting channel",
    "Dress code at FinSolve offices",
    "Parental leave duration",
    "Performance review cycle timing",
    "Ethics training requirements",
    "Holiday calendar for India offices",
    "Grievance escalation path",
    "Probation period policy",
]

_cross_utterances = [
    "Give me an overview of FinSolve as a company",
    "What departments exist and what do they do?",
    "Summarize everything relevant about our Q3 goals",
    "High-level company priorities this year",
    "What should a new hire read first?",
    "Compare how finance and engineering collaborate",
    "List major initiatives across the organization",
    "What are company-wide OKRs?",
    "Explain FinSolve business model end to end",
    "What resources span multiple teams?",
    "Company history and mission statement",
    "Who should I contact for cross-functional projects?",
    "What are our core values?",
]

ROUTE_FINANCE = "finance_route"
ROUTE_ENGINEERING = "engineering_route"
ROUTE_MARKETING = "marketing_route"
ROUTE_HR_GENERAL = "hr_general_route"
ROUTE_CROSS = "cross_department_route"

ROUTE_TO_COLLECTIONS: dict[str, list[str]] = {
    ROUTE_FINANCE: ["finance"],
    ROUTE_ENGINEERING: ["engineering"],
    ROUTE_MARKETING: ["marketing"],
    ROUTE_HR_GENERAL: ["general"],
    ROUTE_CROSS: ["general", "finance", "engineering", "marketing"],
}


def _build_layer() -> RouteLayer:
    s = get_settings()
    kwargs: dict = {"openai_api_key": s.openai_api_key or None, "name": s.embedding_model}
    if s.openai_base_url:
        kwargs["openai_base_url"] = s.openai_base_url
    encoder = OpenAIEncoder(**kwargs)
    routes = [
        SemanticRoute(name=ROUTE_FINANCE, utterances=_finance_utterances),
        SemanticRoute(name=ROUTE_ENGINEERING, utterances=_engineering_utterances),
        SemanticRoute(name=ROUTE_MARKETING, utterances=_marketing_utterances),
        SemanticRoute(name=ROUTE_HR_GENERAL, utterances=_hr_general_utterances),
        SemanticRoute(name=ROUTE_CROSS, utterances=_cross_utterances),
    ]
    return RouteLayer(encoder=encoder, routes=routes)


_layer: RouteLayer | None = None


def get_route_layer() -> RouteLayer:
    global _layer
    if _layer is None:
        _layer = _build_layer()
    return _layer


def classify_route(query: str) -> str:
    try:
        layer = get_route_layer()
        out = layer(query)
        name = getattr(out, "name", None)
        if name:
            return str(name)
    except Exception as e:
        logger.warning("semantic router fallback to cross_department_route: %s", e)
    return ROUTE_CROSS
