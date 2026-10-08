"""Permission checks before tool execution."""

from __future__ import annotations

ROLE_PERMISSIONS = {
    "user": {"order:read", "ticket:read", "ticket:write", "kb:read", "data:read"},
    "agent": {"order:read", "ticket:read", "ticket:write", "ticket:update", "kb:read", "data:read"},
    "admin": {
        "order:read",
        "ticket:read",
        "ticket:write",
        "ticket:update",
        "kb:read",
        "kb:write",
        "data:read",
        "eval:run",
    },
}


def check_permission(role: str, permission: str) -> bool:
    perms = ROLE_PERMISSIONS.get(role, set())
    return permission in perms


def require_permission(role: str, permission: str) -> None:
    if not check_permission(role, permission):
        raise PermissionError(f"role '{role}' missing permission '{permission}'")
