from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from finbot.access_matrix import collections_for_role
from finbot.db import verify_user
from finbot.settings import get_settings

_bearer = HTTPBearer(auto_error=False)


def create_access_token(*, username: str, role: str, is_admin: bool) -> str:
    s = get_settings()
    now = datetime.now(timezone.utc)
    payload = {
        "sub": username,
        "role": role,
        "adm": bool(is_admin),
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=s.jwt_expire_minutes)).timestamp()),
    }
    return jwt.encode(payload, s.jwt_secret, algorithm=s.jwt_algorithm)


def decode_token(token: str) -> dict:
    s = get_settings()
    try:
        return jwt.decode(token, s.jwt_secret, algorithms=[s.jwt_algorithm])
    except jwt.PyJWTError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token") from e


def get_current_user(creds: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> dict:
    if creds is None or creds.scheme.lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")
    data = decode_token(creds.credentials)
    return {
        "username": data["sub"],
        "role": data["role"],
        "is_admin": bool(data.get("adm", False)),
        "collections_accessible": collections_for_role(data["role"]),
    }


def require_admin(user: dict = Depends(get_current_user)) -> dict:
    if not user.get("is_admin"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin only")
    return user
