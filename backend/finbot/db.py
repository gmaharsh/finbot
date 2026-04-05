import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from passlib.context import CryptContext

from finbot.settings import get_settings

_pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")


def _db_path() -> Path:
    s = get_settings()
    raw = s.database_url.replace("sqlite:///", "")
    if raw.startswith("./"):
        return Path(__file__).resolve().parent.parent / raw[2:]
    return Path(raw)


@contextmanager
def get_conn():
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL,
                is_admin INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                path TEXT NOT NULL,
                filename TEXT NOT NULL,
                collection TEXT NOT NULL,
                access_roles TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                ingested_at TEXT
            );
            """
        )
    seed_if_empty()


def seed_if_empty() -> None:
    with get_conn() as conn:
        row = conn.execute("SELECT COUNT(*) AS c FROM users").fetchone()
        if row and row["c"] > 0:
            return
        now = datetime.utcnow().isoformat() + "Z"
        demo = [
            ("employee", "employee"),
            ("finance", "finance"),
            ("engineering", "engineering"),
            ("marketing", "marketing"),
            ("c_level", "c_level"),
        ]
        for username, role in demo:
            conn.execute(
                "INSERT INTO users (id, username, password_hash, role, is_admin, created_at) VALUES (?,?,?,?,?,?)",
                (
                    str(uuid.uuid4()),
                    username,
                    _pwd.hash("demo123"),
                    role,
                    0,
                    now,
                ),
            )
        conn.execute(
            "INSERT INTO users (id, username, password_hash, role, is_admin, created_at) VALUES (?,?,?,?,?,?)",
            (
                str(uuid.uuid4()),
                "admin",
                _pwd.hash("admin123"),
                "c_level",
                1,
                now,
            ),
        )


def verify_user(username: str, password: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, username, password_hash, role, is_admin FROM users WHERE username = ?",
            (username,),
        ).fetchone()
    if not row or not _pwd.verify(password, row["password_hash"]):
        return None
    return {
        "id": row["id"],
        "username": row["username"],
        "role": row["role"],
        "is_admin": bool(row["is_admin"]),
    }


def list_users() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, username, role, is_admin, created_at FROM users ORDER BY username"
        ).fetchall()
    return [dict(r) for r in rows]


def create_user(username: str, password: str, role: str, is_admin: bool = False) -> dict:
    uid = str(uuid.uuid4())
    now = datetime.utcnow().isoformat() + "Z"
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO users (id, username, password_hash, role, is_admin, created_at) VALUES (?,?,?,?,?,?)",
            (uid, username, _pwd.hash(password), role, 1 if is_admin else 0, now),
        )
    return {"id": uid, "username": username, "role": role, "is_admin": is_admin}


def delete_user(user_id: str) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))


def list_documents() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["access_roles"] = json.loads(d["access_roles"])
        d["active"] = bool(d["active"])
        out.append(d)
    return out


def add_document_record(path: str, filename: str, collection: str, access_roles: list[str]) -> dict:
    did = str(uuid.uuid4())
    now = datetime.utcnow().isoformat() + "Z"
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO documents (id, path, filename, collection, access_roles, active, created_at)
               VALUES (?,?,?,?,?,?,?)""",
            (did, path, filename, collection, json.dumps(access_roles), 1, now),
        )
    return {
        "id": did,
        "path": path,
        "filename": filename,
        "collection": collection,
        "access_roles": access_roles,
        "active": True,
        "created_at": now,
        "ingested_at": None,
    }


def touch_document_ingested_by_filename(filename: str) -> None:
    now = datetime.utcnow().isoformat() + "Z"
    with get_conn() as conn:
        conn.execute("UPDATE documents SET ingested_at = ? WHERE filename = ?", (now, filename))


def remove_document_record(doc_id: str) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))


def set_document_ingested(doc_id: str) -> None:
    now = datetime.utcnow().isoformat() + "Z"
    with get_conn() as conn:
        conn.execute("UPDATE documents SET ingested_at = ? WHERE id = ?", (now, doc_id))
