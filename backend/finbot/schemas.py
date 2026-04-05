from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    role: str
    is_admin: bool
    collections_accessible: list[str]


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)


class SourceRef(BaseModel):
    source_document: str | None
    page_number: int | None
    collection: str | None
    score: float | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceRef]
    route: str | None
    role: str
    collections_accessible: list[str]
    target_collections: list[str]
    blocked: bool
    block_reason: str | None
    guardrail_flags: list[str]
    guardrail_warnings: list[str]


class UserCreate(BaseModel):
    username: str
    password: str
    role: str
    is_admin: bool = False


class UserOut(BaseModel):
    id: str
    username: str
    role: str
    is_admin: bool
    created_at: str | None = None


class DocumentOut(BaseModel):
    id: str
    path: str
    filename: str
    collection: str
    access_roles: list[str]
    active: bool
    created_at: str | None
    ingested_at: str | None
