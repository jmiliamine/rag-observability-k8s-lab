"""Corpus loading: every .md/.txt file under a folder, split into overlapping chunks."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

TEXT_SUFFIXES = {".md", ".txt"}


def load_folder(folder: Path) -> list[Document]:
    docs = []
    for f in sorted(folder.rglob("*")):
        if f.suffix.lower() in TEXT_SUFFIXES and f.is_file():
            text = f.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                docs.append(Document(page_content=text, metadata={"source": f.relative_to(folder).as_posix()}))
    return docs


def index_id(chunks: Sequence[Document], embed_model: str) -> str:
    """Short identity of an index: the embedding model and the exact chunks it holds. Two indexes
    with the same id answer a search the same way; a changed note, chunk size or model gives
    another id. Written on the retrieval span, so a trace says which index answered."""
    h = hashlib.sha256(embed_model.encode())
    for c in chunks:
        h.update(b"\x00" + str(c.metadata.get("source", "")).encode() + b"\x00" + c.page_content.encode())
    return h.hexdigest()[:12]


def split(docs: Iterable[Document], chunk_size: int = 800, overlap: int = 120) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = splitter.split_documents(list(docs))
    for i, c in enumerate(chunks):
        c.metadata["chunk"] = i
    return chunks
