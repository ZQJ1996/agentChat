"""Prompt injection detection and sensitive data masking."""

from __future__ import annotations

import re
from dataclasses import dataclass


INJECTION_PATTERNS = [
    r"忽略.{0,8}(指令|提示|规则)",
    r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions|prompts)",
    r"jailbreak",
    r"you\s+are\s+now\s+dan",
    r"system\s+prompt",
    r"揭示.{0,6}(系统)?提示词",
]


PHONE_RE = re.compile(r"(?<!\d)(1[3-9]\d{9})(?!\d)")
ID_RE = re.compile(r"(?<!\d)(\d{6})(\d{8})(\d{4})(?!\d)")
EMAIL_RE = re.compile(r"([a-zA-Z0-9_.+-]+)@([a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)")


@dataclass
class SecurityCheck:
    safe: bool
    reasons: list[str]
    sanitized_text: str


def detect_injection(text: str) -> list[str]:
    reasons: list[str] = []
    for pat in INJECTION_PATTERNS:
        if re.search(pat, text, flags=re.IGNORECASE):
            reasons.append(f"matched injection pattern: {pat}")
    return reasons


def mask_sensitive(text: str) -> str:
    text = PHONE_RE.sub(lambda m: m.group(1)[:3] + "****" + m.group(1)[-4:], text)
    text = ID_RE.sub(lambda m: m.group(1) + "********" + m.group(3), text)
    text = EMAIL_RE.sub(lambda m: m.group(1)[:2] + "***@" + m.group(2), text)
    return text


def check_user_input(text: str) -> SecurityCheck:
    reasons = detect_injection(text)
    sanitized = mask_sensitive(text)
    return SecurityCheck(safe=not reasons, reasons=reasons, sanitized_text=sanitized)
