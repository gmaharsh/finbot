"""Role to collection access. Enforced at Qdrant filter + router intersection."""

from typing import Literal

Role = Literal["employee", "finance", "engineering", "marketing", "c_level"]
Collection = Literal["general", "finance", "engineering", "marketing"]

ROLE_COLLECTIONS: dict[str, list[str]] = {
    "employee": ["general"],
    "finance": ["general", "finance"],
    "engineering": ["general", "engineering"],
    "marketing": ["general", "marketing"],
    "c_level": ["general", "finance", "engineering", "marketing"],
}

# Default access_roles stored per chunk at ingest (who may see this document)
COLLECTION_ACCESS_ROLES: dict[str, list[str]] = {
    "general": ["employee", "finance", "engineering", "marketing", "c_level"],
    "finance": ["finance", "c_level"],
    "engineering": ["engineering", "c_level"],
    "marketing": ["marketing", "c_level"],
}


def collections_for_role(role: str) -> list[str]:
    if role not in ROLE_COLLECTIONS:
        return []
    return list(ROLE_COLLECTIONS[role])


def access_roles_for_collection(collection: str) -> list[str]:
    return list(COLLECTION_ACCESS_ROLES.get(collection, []))
