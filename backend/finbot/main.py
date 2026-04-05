import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from finbot import ingest as ingest_mod
from finbot import qdrant_store
from finbot.access_matrix import access_roles_for_collection, collections_for_role
from finbot.auth_jwt import create_access_token, get_current_user, require_admin
from finbot.chat_service import process_chat
from finbot.db import (
    add_document_record,
    create_user,
    delete_user,
    init_db,
    list_documents,
    list_users,
    remove_document_record,
    touch_document_ingested_by_filename,
    verify_user,
)
from finbot.schemas import (
    ChatRequest,
    ChatResponse,
    DocumentOut,
    LoginRequest,
    LoginResponse,
    SourceRef,
    UserCreate,
    UserOut,
)
from finbot.settings import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("finbot.api")

app = FastAPI(title="FinBot API", version="0.1.0")


@app.on_event("startup")
def _startup() -> None:
    init_db()
    upload = Path(get_settings().upload_dir).resolve()
    upload.mkdir(parents=True, exist_ok=True)


def _cors_origins() -> list[str]:
    return [o.strip() for o in get_settings().cors_origins.split(",") if o.strip()]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/auth/login", response_model=LoginResponse)
def login(body: LoginRequest) -> LoginResponse:
    user = verify_user(body.username, body.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token(
        username=user["username"],
        role=user["role"],
        is_admin=user["is_admin"],
    )
    return LoginResponse(
        access_token=token,
        username=user["username"],
        role=user["role"],
        is_admin=user["is_admin"],
        collections_accessible=collections_for_role(user["role"]),
    )


@app.get("/api/me")
def me(user: dict = Depends(get_current_user)) -> dict:
    return user


@app.post("/api/chat", response_model=ChatResponse)
def chat(body: ChatRequest, user: dict = Depends(get_current_user)) -> ChatResponse:
    session_key = user["username"]
    result = process_chat(query=body.message, role=user["role"], session_key=session_key)
    return ChatResponse(
        answer=result["answer"],
        sources=[SourceRef(**s) for s in result["sources"]],
        route=result["route"],
        role=user["role"],
        collections_accessible=result["collections_accessible"],
        target_collections=result["target_collections"],
        blocked=result["blocked"],
        block_reason=result["block_reason"],
        guardrail_flags=result["guardrail_flags"],
        guardrail_warnings=result["guardrail_warnings"],
    )


# --- Admin ---


@app.get("/api/admin/users", response_model=list[UserOut])
def admin_users(_admin: dict = Depends(require_admin)) -> list[UserOut]:
    return [
        UserOut(
            id=r["id"],
            username=r["username"],
            role=r["role"],
            is_admin=bool(r["is_admin"]),
            created_at=r.get("created_at"),
        )
        for r in list_users()
    ]


@app.post("/api/admin/users", response_model=UserOut)
def admin_create_user(body: UserCreate, _admin: dict = Depends(require_admin)) -> UserOut:
    try:
        u = create_user(body.username, body.password, body.role, body.is_admin)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return UserOut(id=u["id"], username=u["username"], role=u["role"], is_admin=u["is_admin"])


@app.delete("/api/admin/users/{user_id}")
def admin_delete_user(user_id: str, _admin: dict = Depends(require_admin)) -> dict:
    delete_user(user_id)
    return {"ok": True}


@app.get("/api/admin/documents", response_model=list[DocumentOut])
def admin_docs(_admin: dict = Depends(require_admin)) -> list[DocumentOut]:
    return [DocumentOut(**d) for d in list_documents()]


@app.post("/api/admin/documents", response_model=DocumentOut)
async def admin_upload(
    _admin: dict = Depends(require_admin),
    file: UploadFile = File(...),
    collection: str = Form(...),
) -> DocumentOut:
    if collection not in {"general", "finance", "engineering", "marketing"}:
        raise HTTPException(status_code=400, detail="Invalid collection")
    upload_root = Path(get_settings().upload_dir).resolve()
    upload_root.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file.filename or "upload").name
    dest = upload_root / safe_name
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    roles = access_roles_for_collection(collection)
    rec = add_document_record(str(dest), safe_name, collection, roles)
    try:
        ingest_mod.ingest_file(dest, collection, roles)
        touch_document_ingested_by_filename(safe_name)
    except Exception as e:
        logger.exception("Ingest failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Ingest failed: {e}") from e
    rec["ingested_at"] = datetime.now(timezone.utc).isoformat()
    return DocumentOut(**rec)


@app.delete("/api/admin/documents/{doc_id}")
def admin_remove_doc(doc_id: str, _admin: dict = Depends(require_admin)) -> dict:
    docs = list_documents()
    target = next((d for d in docs if d["id"] == doc_id), None)
    if target:
        qdrant_store.delete_by_source_document(target["filename"])
        remove_document_record(doc_id)
    return {"ok": True}


@app.post("/api/admin/reindex")
def admin_reindex(_admin: dict = Depends(require_admin)) -> dict:
    try:
        ingest_mod.ingest_all_data_dir()
    except Exception as e:
        logger.exception("Reindex failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e
    return {"ok": True, "message": "Ingested all files under data_dir subfolders"}


# Expose app for uvicorn
__all__ = ["app"]
