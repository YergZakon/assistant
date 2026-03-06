#!/usr/bin/env python3
"""Hybrid protocol backend adapter for BMAD workflow and API usage."""

from __future__ import annotations

import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import quote_plus

from pdf_hybrid_index import Pipeline


TOKEN_RE = re.compile(r"[0-9a-zA-Zа-яА-ЯёЁ]+(?:\.[0-9a-zA-Zа-яА-ЯёЁ]+)?")


def _normalize_key(value: str) -> str:
    value = (value or "").strip().casefold()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[^0-9a-zа-яё _-]+", "", value)
    return value.strip()


class HybridProtocolAssistant:
    """Adapter exposing legacy assistant interface on top of PDF hybrid index."""

    def __init__(
        self,
        project_root: Path,
        corpus_dir: Optional[Path] = None,
        index_dir: Optional[Path] = None,
        embedding_model: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
    ) -> None:
        self.project_root = project_root.resolve()
        self.corpus_dir = (
            corpus_dir.resolve()
            if corpus_dir
            else self.project_root / "clinical_protocols_2026-03-06_041600"
        )
        self.index_dir = (
            index_dir.resolve()
            if index_dir
            else self.project_root
            / "_bmad-output"
            / "implementation-artifacts"
            / "pdf_vector_index"
        )
        self.embedding_model = embedding_model or os.environ.get("EMBEDDING_MODEL")
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.openai_base_url = openai_base_url or os.environ.get("OPENAI_BASE_URL")

        self.pipeline = Pipeline(corpus_dir=self.corpus_dir, index_dir=self.index_dir)
        self.searcher = None
        self._doc_registry: Dict[str, Dict[str, Any]] = {}
        self._title_url_map = self._load_title_url_map()

    def _load_title_url_map(self) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        index_path = self.project_root / "index.json"
        if not index_path.exists():
            return mapping
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            return mapping
        protocols = payload.get("protocols", []) if isinstance(payload, dict) else []
        if not isinstance(protocols, list):
            return mapping
        for item in protocols:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip()
            url = str(item.get("url", "")).strip()
            file_name = str(item.get("file", "")).strip()
            if url:
                if title:
                    mapping[_normalize_key(title)] = url
                if file_name:
                    mapping[_normalize_key(Path(file_name).stem)] = url
        return mapping

    def _resolve_url(self, title: str, canonical_file: str) -> str:
        by_file = self._title_url_map.get(_normalize_key(Path(canonical_file).stem))
        if by_file:
            return by_file
        by_title = self._title_url_map.get(_normalize_key(title))
        if by_title:
            return by_title
        return f"https://nrchd.kz/clinical-protocols/catalog?q={quote_plus(title)}"

    def _index_ready(self) -> bool:
        return (
            self.pipeline.lexical_db.exists()
            and self.pipeline.metadata_json.exists()
            and self.pipeline.manifest_jsonl.exists()
        )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.pipeline.lexical_db))
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _clip_text(text: str, limit: int) -> str:
        text = " ".join((text or "").split()).strip()
        if len(text) <= limit:
            return text
        return text[:limit].rstrip() + "..."

    def _load_doc_registry(self) -> None:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT doc_id, title, year, mkb_codes_json, patient_group, canonical_file,
                       aliases_json, section, text, page_start, page_end
                FROM chunks
                ORDER BY doc_id, page_start, page_end
                """
            ).fetchall()

        registry: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            doc_id = str(row["doc_id"])
            item = registry.get(doc_id)
            if item is None:
                title = str(row["title"])
                canonical_file = str(row["canonical_file"])
                mkb_codes = json.loads(row["mkb_codes_json"] or "[]")
                aliases = json.loads(row["aliases_json"] or "[]")
                summary = self._clip_text(str(row["text"] or ""), 900)
                item = {
                    "id": doc_id,
                    "title": title,
                    "year": str(row["year"] or ""),
                    "mkb_codes": mkb_codes if isinstance(mkb_codes, list) else [],
                    "patient_group": str(row["patient_group"] or "unknown"),
                    "canonical_file": canonical_file,
                    "aliases": aliases if isinstance(aliases, list) else [],
                    "summary": summary,
                    "sections": set(),
                    "url": self._resolve_url(title=title, canonical_file=canonical_file),
                }
                registry[doc_id] = item
            section = str(row["section"] or "").strip()
            if section:
                item["sections"].add(section)

        for item in registry.values():
            item["sections"] = sorted(item["sections"])
        self._doc_registry = registry

    def ensure_index(self, force_rebuild: bool = False) -> None:
        if force_rebuild or not self._index_ready():
            self.pipeline.build(
                with_embeddings=False,
                embedding_model=self.embedding_model or "text-embedding-3-small",
                openai_api_key=self.openai_api_key,
                openai_base_url=self.openai_base_url,
            )

        self.searcher = self.pipeline.load_searcher(
            embedding_model=self.embedding_model,
            openai_api_key=self.openai_api_key,
            openai_base_url=self.openai_base_url,
        )
        self._load_doc_registry()

    @staticmethod
    def _doc_relevance(score: float) -> float:
        score = max(0.0, float(score))
        return round(min(1.0, score * 2.5), 4)

    def _aggregate_doc_results(
        self,
        chunk_results: Sequence[Dict[str, Any]],
        limit: int,
    ) -> List[Dict[str, Any]]:
        by_doc: Dict[str, Dict[str, Any]] = {}
        for chunk in chunk_results:
            doc_id = str(chunk.get("doc_id", "")).strip()
            if not doc_id:
                continue
            score = float(chunk.get("final_score", 0.0))
            current = by_doc.get(doc_id)
            if current is None:
                meta = self._doc_registry.get(doc_id, {})
                title = str(chunk.get("title") or meta.get("title") or doc_id)
                canonical_file = str(
                    chunk.get("canonical_file") or meta.get("canonical_file") or ""
                )
                mkb_codes = chunk.get("mkb_codes") or meta.get("mkb_codes") or []
                if not isinstance(mkb_codes, list):
                    mkb_codes = []
                section = str(chunk.get("section") or "")
                snippet = self._clip_text(str(chunk.get("text") or ""), 500)
                summary = str(meta.get("summary") or snippet)
                sections = set(meta.get("sections") or [])
                if section:
                    sections.add(section)
                by_doc[doc_id] = {
                    "id": doc_id,
                    "title": title,
                    "mkb_codes": mkb_codes,
                    "url": str(meta.get("url") or self._resolve_url(title, canonical_file)),
                    "file": canonical_file,
                    "summary": summary,
                    "sections": sections,
                    "snippet": snippet,
                    "relevance": self._doc_relevance(score),
                    "bm25_score": float(chunk.get("lexical_score", 0.0)),
                    "_best_score": score,
                }
                continue

            current_section = str(chunk.get("section") or "")
            if current_section:
                current["sections"].add(current_section)
            if score > current["_best_score"]:
                current["_best_score"] = score
                current["relevance"] = self._doc_relevance(score)
                current["snippet"] = self._clip_text(str(chunk.get("text") or ""), 500)
                current["bm25_score"] = float(chunk.get("lexical_score", 0.0))

        rows = list(by_doc.values())
        rows.sort(
            key=lambda x: (
                float(x.get("_best_score", 0.0)),
                float(x.get("relevance", 0.0)),
            ),
            reverse=True,
        )
        out: List[Dict[str, Any]] = []
        for row in rows[:limit]:
            row["sections"] = sorted(row.get("sections") or [])
            row.pop("_best_score", None)
            out.append(row)
        return out

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        query = (query or "").strip()
        if not query:
            return []
        if self.searcher is None:
            self.ensure_index()

        assert self.searcher is not None
        limit = max(1, min(int(limit), 20))
        candidate_k = max(30, limit * 8)
        fetch_top = max(limit * 6, 18)
        payload = self.searcher.search(
            query=query,
            top_k=fetch_top,
            candidate_k=candidate_k,
        )
        chunk_results = payload.get("results") or []
        if not isinstance(chunk_results, list):
            return []
        return self._aggregate_doc_results(chunk_results, limit=limit)

    def get_protocol(self, protocol_id: str, include_full_text: bool = False) -> Dict[str, Any]:
        protocol_id = (protocol_id or "").strip()
        if not protocol_id:
            raise ValueError("protocol_id is required")
        if self.searcher is None:
            self.ensure_index()

        meta = self._doc_registry.get(protocol_id)
        if not meta:
            raise KeyError(f"Protocol not found: {protocol_id}")

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT section, text, page_start, page_end
                FROM chunks
                WHERE doc_id = ?
                ORDER BY page_start ASC, page_end ASC
                """,
                (protocol_id,),
            ).fetchall()

        content: Dict[str, str] = {}
        full_parts: List[str] = []
        for row in rows:
            section = str(row["section"] or "other").strip() or "other"
            part = self._clip_text(str(row["text"] or ""), 2600)
            if not part:
                continue
            full_parts.append(part)
            prev = content.get(section)
            if prev:
                merged = f"{prev}\n\n{part}"
                content[section] = self._clip_text(merged, 8000)
            else:
                content[section] = part

        response = {
            "id": meta["id"],
            "title": meta["title"],
            "mkb_codes": meta.get("mkb_codes") or [],
            "url": meta.get("url") or "",
            "file": meta.get("canonical_file") or "",
            "summary": meta.get("summary") or "",
            "sections": meta.get("sections") or sorted(content.keys()),
            "content": content,
            "version": meta.get("year") or "",
        }
        if include_full_text:
            response["full_text"] = self._clip_text("\n\n".join(full_parts), 30000)
        return response

    def protocol_count(self) -> int:
        if not self._doc_registry:
            if self.searcher is None:
                self.ensure_index()
            else:
                self._load_doc_registry()
        return len(self._doc_registry)

