"""Document loading and chunking."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DocumentChunk:
    chunk_id: str
    text: str
    source: str
    title: str
    section: str
    metadata: dict[str, str]


def load_markdown_files(directory: Path) -> list[tuple[str, str, str]]:
    """Return list of (source, title, content)."""
    docs: list[tuple[str, str, str]] = []
    if not directory.exists():
        return docs
    for path in sorted(directory.glob("**/*")):
        if path.suffix.lower() not in {".md", ".txt", ".markdown"}:
            continue
        text = path.read_text(encoding="utf-8")
        title = path.stem
        # first heading as title if present
        m = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
        if m:
            title = m.group(1).strip()
        docs.append((str(path.name), title, text))
    return docs


def chunk_text(
    text: str,
    *,
    source: str,
    title: str,
    chunk_size: int = 400,
    overlap: int = 60,
) -> list[DocumentChunk]:
    sections = _split_sections(text)
    chunks: list[DocumentChunk] = []
    counter = 0
    for section_title, body in sections:
        pieces = _window_split(body, chunk_size=chunk_size, overlap=overlap)
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            counter += 1
            chunk_id = f"{source}::{counter}"
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    text=piece,
                    source=source,
                    title=title,
                    section=section_title,
                    metadata={
                        "source": source,
                        "title": title,
                        "section": section_title,
                        "chunk_id": chunk_id,
                    },
                )
            )
    return chunks


def _split_sections(text: str) -> list[tuple[str, str]]:
    parts = re.split(r"(?m)^(##?\s.+)$", text)
    if len(parts) == 1:
        return [("正文", text)]
    sections: list[tuple[str, str]] = []
    intro = parts[0].strip()
    if intro:
        sections.append(("引言", intro))
    i = 1
    while i < len(parts):
        heading = re.sub(r"^#+\s*", "", parts[i]).strip()
        body = parts[i + 1] if i + 1 < len(parts) else ""
        sections.append((heading or "章节", body))
        i += 2
    return sections


def _window_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    # approximate tokens by characters for CJK-friendly chunking
    if len(text) <= chunk_size:
        return [text]
    step = max(1, chunk_size - overlap)
    out: list[str] = []
    for start in range(0, len(text), step):
        out.append(text[start : start + chunk_size])
        if start + chunk_size >= len(text):
            break
    return out
