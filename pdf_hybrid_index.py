#!/usr/bin/env python3
"""Hybrid indexing and search pipeline for clinical protocol PDFs.

Implements:
- Corpus canonicalization (sha256 doc_id + aliases)
- PDF text extraction with section detection
- Section-aware chunking
- Lexical index (SQLite FTS5)
- Semantic index (Qdrant local + OpenAI embeddings)
- Hybrid retrieval (RRF + heuristic rerank)
- Incremental embedding updates
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import os
import re
import sqlite3
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from urllib.parse import urlparse

import fitz  # type: ignore
import pandas as pd
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm


LOGGER = logging.getLogger("protocols.vector")
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"

TOKEN_RE = re.compile(r"\S+")
MKB_RE = re.compile(r"\b[A-Z]\d{1,2}(?:\.\d{1,2})?\b")
MKB_RANGE_RE = re.compile(
    r"\b([A-Z]\d{1,2}(?:\.\d{1,2})?)\s*-\s*([A-Z]?\d{1,2}(?:\.\d{1,2})?)\b"
)
YEAR_RE = re.compile(r"\b(20[0-2]\d)\b")

SECTION_PATTERNS: List[Tuple[str, re.Pattern[str]]] = [
    ("introduction", re.compile(r"\b(вводная часть|общая информация|краткое описание)\b", re.I)),
    ("classification", re.compile(r"\bклассификац", re.I)),
    ("diagnostics", re.compile(r"\bдиагностик", re.I)),
    ("treatment", re.compile(r"\b(лечение|тактика лечения|терапия)\b", re.I)),
    ("hospitalization", re.compile(r"\bгоспитализац", re.I)),
    ("rehabilitation", re.compile(r"\bреабилитац", re.I)),
    ("prevention", re.compile(r"\bпрофилактик", re.I)),
    ("monitoring", re.compile(r"\bмониторинг", re.I)),
    ("appendix", re.compile(r"\bприложени", re.I)),
    ("references", re.compile(r"\b(литератур|references)\b", re.I)),
    ("organization", re.compile(r"\bорганизационн", re.I)),
]

QUERY_INTENT_RULES: List[Tuple[str, re.Pattern[str], List[str]]] = [
    ("diagnostics", re.compile(r"\b(диагностик|обслед|скрининг|анамнез)\b", re.I), ["diagnostics", "classification"]),
    ("treatment", re.compile(r"\b(лечение|терап|препарат|доз|назнач)\b", re.I), ["treatment", "hospitalization"]),
    ("hospitalization", re.compile(r"\b(госпитализац|стационар)\b", re.I), ["hospitalization", "treatment"]),
    ("rehabilitation", re.compile(r"\b(реабилитац)\b", re.I), ["rehabilitation", "treatment"]),
    ("prevention", re.compile(r"\b(профилактик|вакцин)\b", re.I), ["prevention", "monitoring"]),
    ("monitoring", re.compile(r"\b(наблюдени|мониторинг|контрол)\b", re.I), ["monitoring", "diagnostics"]),
]

PEDIATRIC_MARKERS = {"дет", "ребен", "ребён", "новорож", "подрост"}
ADULT_MARKERS = {"взросл"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def normalize_openai_base_url(raw_value: Optional[str]) -> Optional[str]:
    value = (raw_value or "").strip()
    if not value:
        return None
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"}:
        LOGGER.warning(
            "Ignoring invalid OPENAI_BASE_URL without http/https scheme: %s",
            value,
        )
        return None
    return value


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_text_basic(text: str) -> str:
    text = text.replace("\x00", " ").replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_title_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    stem = re.sub(r"\s+", " ", stem).strip()
    return stem


def detect_section(line: str) -> Optional[str]:
    stripped = line.strip()
    if len(stripped) < 3:
        return None
    for section, pattern in SECTION_PATTERNS:
        if pattern.search(stripped):
            return section
    # All-caps medium lines are frequently section headers.
    if (
        len(stripped) <= 80
        and re.search(r"[А-ЯA-Z]", stripped)
        and stripped.upper() == stripped
        and sum(ch.isalpha() for ch in stripped) >= 5
    ):
        return "header"
    return None


def extract_mkb_codes(text: str) -> List[str]:
    up = text.upper()
    codes = set(MKB_RE.findall(up))
    for left, right in MKB_RANGE_RE.findall(up):
        l = left.replace(" ", "")
        r = right.replace(" ", "")
        if re.match(r"^[A-Z]", r):
            rr = r
        else:
            rr = l[0] + r
        codes.add(f"{l}-{rr}")
    return sorted(codes)


def classify_patient_group(text: str, title: str) -> str:
    hay = f"{title} {text}".lower()
    is_child = any(marker in hay for marker in PEDIATRIC_MARKERS)
    is_adult = any(marker in hay for marker in ADULT_MARKERS)
    if is_child and is_adult:
        return "mixed"
    if is_child:
        return "pediatric"
    if is_adult:
        return "adult"
    return "unknown"


def extract_year(title: str, text: str) -> Optional[str]:
    m = YEAR_RE.search(title)
    if m:
        return m.group(1)
    m2 = YEAR_RE.search(text[:8000])
    return m2.group(1) if m2 else None


def safe_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


@dataclass
class CanonicalDocument:
    doc_id: str
    sha256: str
    canonical_file: str
    display_name: str
    canonical_path: str
    aliases: List[str]
    file_size: int


@dataclass
class PageUnit:
    page: int
    section: str
    text: str


@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    page_start: int
    page_end: int
    year: Optional[str]
    mkb_codes: List[str]
    patient_group: str
    text: str
    canonical_file: str
    aliases: List[str]


class CorpusCanonicalizer:
    def __init__(self, corpus_dir: Path) -> None:
        self.corpus_dir = corpus_dir

    def scan(self, max_docs: Optional[int] = None) -> List[CanonicalDocument]:
        pdf_files = sorted(self.corpus_dir.glob("*.pdf"))
        if max_docs is not None:
            pdf_files = pdf_files[: max(1, max_docs)]
        LOGGER.info("Scanning corpus: %d pdf files", len(pdf_files))

        buckets: Dict[str, List[Path]] = defaultdict(list)
        for pdf in pdf_files:
            file_sha = sha256_file(pdf)
            buckets[file_sha].append(pdf)

        docs: List[CanonicalDocument] = []
        for sha, group in buckets.items():
            group_sorted = sorted(group, key=lambda p: (len(p.name), p.name.lower()))
            canonical = group_sorted[0]
            aliases = [p.name for p in group_sorted]
            docs.append(
                CanonicalDocument(
                    doc_id=sha,
                    sha256=sha,
                    canonical_file=canonical.name,
                    display_name=normalize_title_from_filename(canonical.name),
                    canonical_path=str(canonical),
                    aliases=aliases,
                    file_size=canonical.stat().st_size,
                )
            )
        docs.sort(key=lambda d: d.canonical_file.lower())
        LOGGER.info(
            "Canonicalization complete: %d canonical docs, %d duplicates collapsed",
            len(docs),
            len(pdf_files) - len(docs),
        )
        return docs


class PdfSectionExtractor:
    def __init__(self, repeat_threshold_ratio: float = 0.22) -> None:
        self.repeat_threshold_ratio = repeat_threshold_ratio

    def _remove_repeating_headers(self, pages_lines: List[List[str]]) -> List[List[str]]:
        if not pages_lines:
            return pages_lines
        line_counter: Counter[str] = Counter()
        candidates: List[List[str]] = []
        for lines in pages_lines:
            picked = []
            if lines:
                picked.extend(lines[:2])
                picked.extend(lines[-2:])
            norm = [
                re.sub(r"\s+", " ", line).strip()
                for line in picked
                if 3 <= len(line.strip()) <= 120
            ]
            candidates.append(norm)
            line_counter.update(set(norm))

        threshold = max(8, int(len(pages_lines) * self.repeat_threshold_ratio))
        repeated = {line for line, cnt in line_counter.items() if cnt >= threshold}

        cleaned: List[List[str]] = []
        for lines in pages_lines:
            keep = []
            for line in lines:
                norm = re.sub(r"\s+", " ", line).strip()
                if norm in repeated:
                    continue
                keep.append(line)
            cleaned.append(keep)
        return cleaned

    def extract_units(self, pdf_path: Path) -> Tuple[List[PageUnit], int]:
        doc = fitz.open(str(pdf_path))
        pages_lines: List[List[str]] = []
        for page in doc:
            raw = page.get_text("text") or ""
            raw = clean_text_basic(raw)
            lines = [ln.strip() for ln in raw.split("\n") if ln.strip()]
            pages_lines.append(lines)
        page_count = len(pages_lines)
        doc.close()

        pages_lines = self._remove_repeating_headers(pages_lines)
        units: List[PageUnit] = []
        current_section = "introduction"
        for page_idx, lines in enumerate(pages_lines, start=1):
            if not lines:
                continue
            segment_section = current_section
            segment_lines: List[str] = []
            for line in lines:
                detected = detect_section(line)
                if detected and detected != segment_section and segment_lines:
                    units.append(
                        PageUnit(
                            page=page_idx,
                            section=segment_section,
                            text=clean_text_basic(" ".join(segment_lines)),
                        )
                    )
                    segment_section = detected
                    segment_lines = []
                    # Keep heading line in new segment for context.
                    segment_lines.append(line)
                    current_section = detected
                else:
                    if detected:
                        segment_section = detected
                        current_section = detected
                    segment_lines.append(line)
            if segment_lines:
                units.append(
                    PageUnit(
                        page=page_idx,
                        section=segment_section,
                        text=clean_text_basic(" ".join(segment_lines)),
                    )
                )
        return units, page_count


class SectionAwareChunker:
    def __init__(self, min_tokens: int = 700, max_tokens: int = 1100, overlap: int = 120) -> None:
        if min_tokens <= 0 or max_tokens <= min_tokens:
            raise ValueError("Invalid token limits for chunker")
        if overlap < 0 or overlap >= max_tokens:
            raise ValueError("Invalid overlap")
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.overlap = overlap

    @staticmethod
    def _tokens(text: str) -> List[str]:
        return TOKEN_RE.findall(text)

    def _chunk_stream(
        self,
        doc_id: str,
        title: str,
        section: str,
        segment_index: int,
        year: Optional[str],
        mkb_codes: List[str],
        patient_group: str,
        canonical_file: str,
        aliases: List[str],
        tokens: List[str],
        token_pages: List[int],
    ) -> List[ChunkRecord]:
        chunks: List[ChunkRecord] = []
        if not tokens:
            return chunks
        i = 0
        while i < len(tokens):
            j = min(i + self.max_tokens, len(tokens))
            # If trailing chunk too tiny and we already have chunks, merge into previous window.
            if len(tokens) - j < self.min_tokens // 3 and j < len(tokens):
                j = len(tokens)
            sub_tokens = tokens[i:j]
            pages = token_pages[i:j]
            text = " ".join(sub_tokens).strip()
            if text:
                page_start = min(pages) if pages else 1
                page_end = max(pages) if pages else page_start
                base = (
                    f"{doc_id}|{section}|{segment_index}|{page_start}|{page_end}|{i}|{j}|"
                    f"{hashlib.sha1(text.encode('utf-8')).hexdigest()}"
                )
                chunk_id = hashlib.sha256(base.encode("utf-8")).hexdigest()
                chunks.append(
                    ChunkRecord(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        title=title,
                        section=section,
                        page_start=page_start,
                        page_end=page_end,
                        year=year,
                        mkb_codes=mkb_codes,
                        patient_group=patient_group,
                        text=text,
                        canonical_file=canonical_file,
                        aliases=aliases,
                    )
                )
            if j == len(tokens):
                break
            i = max(0, j - self.overlap)
        return chunks

    def build_chunks(
        self,
        doc: CanonicalDocument,
        units: List[PageUnit],
    ) -> List[ChunkRecord]:
        title = normalize_title_from_filename(doc.canonical_file)
        all_text = " ".join(unit.text for unit in units if unit.text)
        year = extract_year(title, all_text)
        mkb_codes = extract_mkb_codes(all_text)
        patient_group = classify_patient_group(all_text[:12000], title)

        grouped: List[Tuple[str, List[PageUnit]]] = []
        for unit in units:
            if not unit.text:
                continue
            if not grouped or grouped[-1][0] != unit.section:
                grouped.append((unit.section, [unit]))
            else:
                grouped[-1][1].append(unit)

        all_chunks: List[ChunkRecord] = []
        for segment_index, (section, section_units) in enumerate(grouped):
            tokens: List[str] = []
            pages: List[int] = []
            for unit in section_units:
                unit_tokens = self._tokens(unit.text)
                if not unit_tokens:
                    continue
                tokens.extend(unit_tokens)
                pages.extend([unit.page] * len(unit_tokens))
            all_chunks.extend(
                self._chunk_stream(
                    doc_id=doc.doc_id,
                    title=title,
                    section=section,
                    segment_index=segment_index,
                    year=year,
                    mkb_codes=mkb_codes,
                    patient_group=patient_group,
                    canonical_file=doc.canonical_file,
                    aliases=doc.aliases,
                    tokens=tokens,
                    token_pages=pages,
                )
            )
        return all_chunks


class LexicalIndex:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def rebuild(self, chunks: Sequence[ChunkRecord]) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                DROP TABLE IF EXISTS chunks;
                DROP TABLE IF EXISTS chunks_fts;
                CREATE TABLE chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    section TEXT NOT NULL,
                    page_start INTEGER NOT NULL,
                    page_end INTEGER NOT NULL,
                    year TEXT,
                    mkb_codes_json TEXT NOT NULL,
                    patient_group TEXT NOT NULL,
                    canonical_file TEXT NOT NULL,
                    aliases_json TEXT NOT NULL,
                    text TEXT NOT NULL
                );
                CREATE VIRTUAL TABLE chunks_fts USING fts5(
                    chunk_id UNINDEXED,
                    title,
                    section,
                    mkb_codes,
                    text,
                    tokenize='unicode61 remove_diacritics 2'
                );
                """
            )
            for chunk in chunks:
                conn.execute(
                    """
                    INSERT INTO chunks (
                        chunk_id, doc_id, title, section, page_start, page_end,
                        year, mkb_codes_json, patient_group, canonical_file,
                        aliases_json, text
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.chunk_id,
                        chunk.doc_id,
                        chunk.title,
                        chunk.section,
                        chunk.page_start,
                        chunk.page_end,
                        chunk.year,
                        safe_json_dumps(chunk.mkb_codes),
                        chunk.patient_group,
                        chunk.canonical_file,
                        safe_json_dumps(chunk.aliases),
                        chunk.text,
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO chunks_fts (chunk_id, title, section, mkb_codes, text)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.chunk_id,
                        chunk.title,
                        chunk.section,
                        " ".join(chunk.mkb_codes),
                        chunk.text,
                    ),
                )
            conn.commit()
        LOGGER.info("Lexical index rebuilt: %d chunks", len(chunks))

    @staticmethod
    def _build_match_query(query: str) -> str:
        raw_tokens = TOKEN_RE.findall(query)
        terms: List[str] = []
        for token in raw_tokens[:16]:
            t = token.strip().replace('"', "")
            if not t:
                continue
            up = t.upper()
            if MKB_RE.fullmatch(up):
                terms.append(f'"{up}"')
            elif not re.fullmatch(r"[0-9A-Za-zА-Яа-яЁё]+", t):
                terms.append(f'"{t}"')
            elif len(t) <= 2:
                terms.append(f'"{t}"')
            else:
                terms.append(f"{t}*")
        return " OR ".join(terms) if terms else '""'

    def search(
        self,
        query: str,
        limit: int = 30,
        patient_group_filter: Optional[str] = None,
        section_filters: Optional[Sequence[str]] = None,
        year_filters: Optional[Sequence[str]] = None,
        mkb_filters: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        match_expr = self._build_match_query(query)
        sql = [
            """
            SELECT
                c.chunk_id, c.doc_id, c.title, c.section, c.page_start, c.page_end,
                c.year, c.mkb_codes_json, c.patient_group, c.canonical_file,
                c.aliases_json, c.text,
                bm25(chunks_fts, 9.0, 3.0, 4.0, 1.0) AS bm25_score
            FROM chunks_fts
            JOIN chunks c ON c.chunk_id = chunks_fts.chunk_id
            WHERE chunks_fts MATCH ?
            """
        ]
        params: List[Any] = [match_expr]

        if patient_group_filter and patient_group_filter != "unknown":
            sql.append("AND c.patient_group IN (?, ?, ?)")
            params.extend([patient_group_filter, "mixed", "unknown"])

        if section_filters:
            sections = sorted(set(str(s).strip() for s in section_filters if str(s).strip()))
            if sections:
                placeholders = ", ".join("?" for _ in sections)
                sql.append(f"AND c.section IN ({placeholders})")
                params.extend(sections)

        if year_filters:
            years = sorted(set(str(y).strip() for y in year_filters if str(y).strip()))
            if years:
                placeholders = ", ".join("?" for _ in years)
                sql.append(f"AND c.year IN ({placeholders})")
                params.extend(years)

        if mkb_filters:
            mkb = sorted(set(str(code).upper().strip() for code in mkb_filters if str(code).strip()))
            if mkb:
                mkb_sql: List[str] = []
                for code in mkb:
                    mkb_sql.append(
                        """
                        EXISTS (
                            SELECT 1
                            FROM json_each(c.mkb_codes_json) je
                            WHERE UPPER(CAST(je.value AS TEXT)) = ?
                               OR UPPER(CAST(je.value AS TEXT)) LIKE ?
                               OR ? LIKE UPPER(CAST(je.value AS TEXT)) || '%'
                        )
                        """
                    )
                    params.extend([code, f"{code}%", code])
                sql.append("AND (" + " OR ".join(mkb_sql) + ")")

        sql.append("ORDER BY bm25_score ASC")
        sql.append("LIMIT ?")
        params.append(max(1, limit * 3))

        with self._connect() as conn:
            rows = conn.execute("\n".join(sql), params).fetchall()
        out: List[Dict[str, Any]] = []
        for row in rows:
            rec = {
                "chunk_id": row["chunk_id"],
                "doc_id": row["doc_id"],
                "title": row["title"],
                "section": row["section"],
                "page_start": int(row["page_start"]),
                "page_end": int(row["page_end"]),
                "year": row["year"],
                "mkb_codes": json.loads(row["mkb_codes_json"]),
                "patient_group": row["patient_group"],
                "canonical_file": row["canonical_file"],
                "aliases": json.loads(row["aliases_json"]),
                "text": row["text"],
                "lexical_score": float(row["bm25_score"]),
            }
            out.append(rec)
            if len(out) >= limit:
                break
        return out


class SemanticIndex:
    def __init__(
        self,
        qdrant_path: Path,
        collection_name: str,
        embedding_model: str,
        openai_api_key: Optional[str],
        openai_base_url: Optional[str] = None,
    ) -> None:
        self.qdrant_path = qdrant_path
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.openai_api_key = openai_api_key
        self.openai_base_url = openai_base_url
        self.qdrant_path.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(self.qdrant_path))

    def _openai_client(self) -> OpenAI:
        if not self.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for semantic indexing/search")
        explicit_base_url = normalize_openai_base_url(self.openai_base_url)
        kwargs: Dict[str, Any] = {
            "api_key": self.openai_api_key,
            "base_url": explicit_base_url or DEFAULT_OPENAI_BASE_URL,
        }
        return OpenAI(**kwargs)

    @staticmethod
    def _point_id(chunk_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

    def _ensure_collection(self, vector_size: int) -> None:
        exists = self.client.collection_exists(self.collection_name)
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qm.VectorParams(size=vector_size, distance=qm.Distance.COSINE),
            )
            LOGGER.info("Qdrant collection created: %s (dim=%d)", self.collection_name, vector_size)

    def _existing_map(self) -> Dict[str, Optional[str]]:
        points_map: Dict[str, Optional[str]] = {}
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=2048,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for p in points:
                payload = p.payload or {}
                points_map[str(p.id)] = str(payload.get("chunk_id")) if payload.get("chunk_id") else None
            if offset is None:
                break
        return points_map

    def _delete_points(self, point_ids: Sequence[str], batch_size: int = 1024) -> int:
        deleted = 0
        for start in range(0, len(point_ids), batch_size):
            batch = list(point_ids[start : start + batch_size])
            if not batch:
                continue
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=qm.PointIdsList(points=batch),
                wait=True,
            )
            deleted += len(batch)
        return deleted

    def build_embeddings(
        self,
        chunks: Sequence[ChunkRecord],
        batch_size: int = 64,
        prune_stale: bool = True,
    ) -> Dict[str, Any]:
        if not chunks:
            return {"embedded": 0, "skipped_existing": 0, "pruned_stale": 0}

        oai = self._openai_client()
        first_vec = oai.embeddings.create(
            model=self.embedding_model,
            input=[chunks[0].text],
        ).data[0].embedding
        self._ensure_collection(len(first_vec))

        existing = self._existing_map()
        existing_ids: Set[str] = set(existing.keys())
        points_to_embed: List[ChunkRecord] = []
        current_ids: Set[str] = set()
        for chunk in chunks:
            pid = self._point_id(chunk.chunk_id)
            current_ids.add(pid)
            if pid in existing_ids:
                continue
            points_to_embed.append(chunk)

        LOGGER.info(
            "Semantic indexing: total=%d, already_indexed=%d, to_embed=%d",
            len(chunks),
            len(chunks) - len(points_to_embed),
            len(points_to_embed),
        )

        embedded = 0
        for start in range(0, len(points_to_embed), batch_size):
            batch = points_to_embed[start : start + batch_size]
            texts = [c.text for c in batch]
            resp = oai.embeddings.create(model=self.embedding_model, input=texts)
            vectors = [d.embedding for d in resp.data]
            points: List[qm.PointStruct] = []
            for chunk, vec in zip(batch, vectors):
                payload = {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "title": chunk.title,
                    "section": chunk.section,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "year": chunk.year,
                    "mkb_codes": chunk.mkb_codes,
                    "patient_group": chunk.patient_group,
                    "canonical_file": chunk.canonical_file,
                    "aliases": chunk.aliases,
                    "text": chunk.text[:4000],
                    "embedding_model": self.embedding_model,
                }
                points.append(
                    qm.PointStruct(
                        id=self._point_id(chunk.chunk_id),
                        vector=vec,
                        payload=payload,
                    )
                )
            self.client.upsert(collection_name=self.collection_name, points=points, wait=True)
            embedded += len(points)
            LOGGER.info("Embedded %d/%d chunks", embedded, len(points_to_embed))

        pruned = 0
        if prune_stale:
            stale_ids = sorted(existing_ids - current_ids)
            if stale_ids:
                pruned = self._delete_points(stale_ids)
                LOGGER.info("Pruned stale vectors: %d", pruned)

        return {
            "embedded": embedded,
            "skipped_existing": len(chunks) - len(points_to_embed),
            "pruned_stale": pruned,
        }

    def search(
        self,
        query: str,
        limit: int = 30,
        patient_group_filter: Optional[str] = None,
        section_filters: Optional[Sequence[str]] = None,
        year_filters: Optional[Sequence[str]] = None,
        mkb_filters: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        if not self.client.collection_exists(self.collection_name):
            return []
        oai = self._openai_client()
        qvec = oai.embeddings.create(model=self.embedding_model, input=[query]).data[0].embedding
        must: List[qm.FieldCondition] = []
        if patient_group_filter and patient_group_filter != "unknown":
            must.append(
                qm.FieldCondition(
                    key="patient_group",
                    match=qm.MatchAny(any=[patient_group_filter, "mixed", "unknown"]),
                )
            )
        if section_filters:
            sections = sorted(set(str(s).strip() for s in section_filters if str(s).strip()))
            if sections:
                must.append(
                    qm.FieldCondition(
                        key="section",
                        match=qm.MatchAny(any=sections),
                    )
                )
        if year_filters:
            years = sorted(set(str(y).strip() for y in year_filters if str(y).strip()))
            if years:
                must.append(
                    qm.FieldCondition(
                        key="year",
                        match=qm.MatchAny(any=years),
                    )
                )
        if mkb_filters:
            mkb = sorted(set(str(code).upper().strip() for code in mkb_filters if str(code).strip()))
            if mkb:
                must.append(
                    qm.FieldCondition(
                        key="mkb_codes",
                        match=qm.MatchAny(any=mkb),
                    )
                )
        qfilter = qm.Filter(must=must) if must else None
        if hasattr(self.client, "search"):
            hits = self.client.search(
                collection_name=self.collection_name,
                query_vector=qvec,
                limit=limit,
                query_filter=qfilter,
                with_payload=True,
                with_vectors=False,
            )
        else:
            query_resp = self.client.query_points(
                collection_name=self.collection_name,
                query=qvec,
                limit=limit,
                query_filter=qfilter,
                with_payload=True,
                with_vectors=False,
            )
            hits = query_resp.points
        out: List[Dict[str, Any]] = []
        for hit in hits:
            payload = hit.payload or {}
            out.append(
                {
                    "chunk_id": payload.get("chunk_id"),
                    "doc_id": payload.get("doc_id"),
                    "title": payload.get("title"),
                    "section": payload.get("section"),
                    "page_start": payload.get("page_start"),
                    "page_end": payload.get("page_end"),
                    "year": payload.get("year"),
                    "mkb_codes": payload.get("mkb_codes") or [],
                    "patient_group": payload.get("patient_group"),
                    "canonical_file": payload.get("canonical_file"),
                    "aliases": payload.get("aliases") or [],
                    "text": payload.get("text") or "",
                    "semantic_score": float(hit.score),
                }
            )
        return out


class HybridSearcher:
    def __init__(
        self,
        lexical_index: LexicalIndex,
        semantic_index: Optional[SemanticIndex] = None,
    ) -> None:
        self.lexical = lexical_index
        self.semantic = semantic_index

    @staticmethod
    def classify_query(query: str) -> Dict[str, Any]:
        q = query.strip()
        q_lower = q.lower()
        mkb_codes = sorted(set(code.upper() for code in MKB_RE.findall(q.upper())))
        years = sorted(set(YEAR_RE.findall(q)))
        patient_group = "unknown"
        if any(marker in q_lower for marker in PEDIATRIC_MARKERS):
            patient_group = "pediatric"
        elif any(marker in q_lower for marker in ADULT_MARKERS):
            patient_group = "adult"

        intent = "general"
        preferred_sections = ["introduction", "diagnostics", "treatment"]
        for name, pattern, sections in QUERY_INTENT_RULES:
            if pattern.search(q):
                intent = name
                preferred_sections = sections
                break

        use_section_filter = intent != "general"
        return {
            "intent": intent,
            "patient_group": patient_group,
            "mkb_codes": mkb_codes,
            "year_filters": years,
            "preferred_sections": preferred_sections,
            "section_filters": preferred_sections if use_section_filter else [],
            "use_metadata_filters": bool(
                use_section_filter or mkb_codes or years or patient_group != "unknown"
            ),
        }

    @staticmethod
    def _rrf_merge(
        lexical: List[Dict[str, Any]],
        semantic: List[Dict[str, Any]],
        k: int = 60,
    ) -> Dict[str, Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = {}
        for rank, row in enumerate(lexical, start=1):
            cid = row["chunk_id"]
            rec = merged.setdefault(cid, dict(row))
            rec["rrf"] = rec.get("rrf", 0.0) + 1.0 / (k + rank)
            rec["lex_rank"] = rank
        for rank, row in enumerate(semantic, start=1):
            cid = row["chunk_id"]
            rec = merged.setdefault(cid, dict(row))
            # Merge fields from semantic payload if lexical row was sparse.
            for key, value in row.items():
                rec.setdefault(key, value)
            rec["rrf"] = rec.get("rrf", 0.0) + 1.0 / (k + rank)
            rec["sem_rank"] = rank
        return merged

    @staticmethod
    def _mkb_overlap_boost(query_codes: List[str], candidate_codes: List[str]) -> float:
        if not query_codes:
            return 0.0
        can = [str(c).upper() for c in candidate_codes]
        hits = 0
        for qc in query_codes:
            if any(qc == cc or cc.startswith(qc) for cc in can):
                hits += 1
        return min(0.4, 0.2 * hits)

    def search(self, query: str, top_k: int = 8, candidate_k: int = 30) -> Dict[str, Any]:
        qmeta = self.classify_query(query)
        section_filters = qmeta.get("section_filters") or None
        year_filters = qmeta.get("year_filters") or None
        mkb_filters = qmeta.get("mkb_codes") or None

        lexical = self.lexical.search(
            query=query,
            limit=candidate_k,
            patient_group_filter=qmeta["patient_group"],
            section_filters=section_filters,
            year_filters=year_filters,
            mkb_filters=mkb_filters,
        )
        if not lexical and qmeta.get("use_metadata_filters"):
            lexical = self.lexical.search(
                query=query,
                limit=candidate_k,
                patient_group_filter=qmeta["patient_group"],
            )

        semantic: List[Dict[str, Any]] = []
        if self.semantic:
            try:
                semantic = self.semantic.search(
                    query=query,
                    limit=candidate_k,
                    patient_group_filter=qmeta["patient_group"],
                    section_filters=section_filters,
                    year_filters=year_filters,
                    mkb_filters=mkb_filters,
                )
                if not semantic and qmeta.get("use_metadata_filters"):
                    semantic = self.semantic.search(
                        query=query,
                        limit=candidate_k,
                        patient_group_filter=qmeta["patient_group"],
                    )
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("Semantic search failed, fallback to lexical only: %s", exc)
                semantic = []

        merged = self._rrf_merge(lexical, semantic)
        rescored: List[Dict[str, Any]] = []
        for row in merged.values():
            section = str(row.get("section") or "")
            section_boost = 0.15 if section in qmeta["preferred_sections"] else 0.0
            mkb_boost = self._mkb_overlap_boost(qmeta["mkb_codes"], row.get("mkb_codes") or [])
            sem_signal = float(row.get("semantic_score") or 0.0)
            lex_signal = 1.0 / (1.0 + abs(float(row.get("lexical_score") or 1.0)))
            final = (
                0.55 * float(row.get("rrf", 0.0))
                + 0.20 * sem_signal
                + 0.20 * lex_signal
                + section_boost
                + mkb_boost
            )
            row["final_score"] = round(final, 6)
            rescored.append(row)

        rescored.sort(key=lambda x: x["final_score"], reverse=True)
        top = rescored[: max(1, top_k)]

        citations = [
            {
                "title": r.get("title"),
                "canonical_file": r.get("canonical_file"),
                "section": r.get("section"),
                "page_start": r.get("page_start"),
                "page_end": r.get("page_end"),
                "chunk_id": r.get("chunk_id"),
            }
            for r in top
        ]
        return {
            "query": query,
            "query_meta": qmeta,
            "results": top,
            "citations": citations,
            "counts": {
                "lexical_candidates": len(lexical),
                "semantic_candidates": len(semantic),
                "merged_candidates": len(merged),
            },
            "applied_filters": {
                "patient_group": qmeta["patient_group"],
                "section_filters": section_filters or [],
                "year_filters": year_filters or [],
                "mkb_filters": mkb_filters or [],
            },
        }


class Pipeline:
    def __init__(
        self,
        corpus_dir: Path,
        index_dir: Path,
        min_tokens: int = 700,
        max_tokens: int = 1100,
        overlap: int = 120,
    ) -> None:
        self.corpus_dir = corpus_dir
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_jsonl = self.index_dir / "manifest.jsonl"
        self.manifest_parquet = self.index_dir / "manifest.parquet"
        self.chunks_jsonl = self.index_dir / "chunks.jsonl"
        self.chunks_parquet = self.index_dir / "chunks.parquet"
        self.metadata_json = self.index_dir / "index_metadata.json"
        self.lexical_db = self.index_dir / "lexical_chunks.db"
        self.qdrant_dir = self.index_dir / "qdrant_local"

        self.canonicalizer = CorpusCanonicalizer(corpus_dir=self.corpus_dir)
        self.extractor = PdfSectionExtractor()
        self.chunker = SectionAwareChunker(
            min_tokens=min_tokens, max_tokens=max_tokens, overlap=overlap
        )

    def _write_jsonl(self, path: Path, rows: Iterable[Dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _write_parquet(self, path: Path, rows: List[Dict[str, Any]]) -> None:
        df = pd.DataFrame(rows)
        try:
            df.to_parquet(path, index=False)
        except ImportError as exc:
            LOGGER.warning(
                "Parquet engine is unavailable; skipping optional file write path=%s error=%s",
                path,
                exc,
            )

    def _load_previous_doc_ids(self) -> Set[str]:
        if not self.manifest_jsonl.exists():
            return set()
        doc_ids: Set[str] = set()
        with self.manifest_jsonl.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                doc_id = str(row.get("doc_id", "")).strip()
                if doc_id:
                    doc_ids.add(doc_id)
        return doc_ids

    def build(
        self,
        max_docs: Optional[int] = None,
        with_embeddings: bool = False,
        embedding_model: str = "text-embedding-3-small",
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
        embedding_batch_size: int = 64,
        prune_stale_vectors: bool = True,
    ) -> Dict[str, Any]:
        t0 = time.time()
        previous_doc_ids = self._load_previous_doc_ids()
        docs = self.canonicalizer.scan(max_docs=max_docs)
        current_doc_ids = {doc.doc_id for doc in docs}
        ingest_stats = {
            "new_docs": len(current_doc_ids - previous_doc_ids),
            "unchanged_docs": len(current_doc_ids & previous_doc_ids),
            "removed_docs": len(previous_doc_ids - current_doc_ids),
        }
        manifest_rows = [dataclasses.asdict(d) for d in docs]
        self._write_jsonl(self.manifest_jsonl, manifest_rows)
        self._write_parquet(self.manifest_parquet, manifest_rows)
        LOGGER.info("Manifest written: %s (%d rows)", self.manifest_jsonl, len(manifest_rows))

        all_chunks: List[ChunkRecord] = []
        for idx, doc in enumerate(docs, start=1):
            units, page_count = self.extractor.extract_units(Path(doc.canonical_path))
            chunks = self.chunker.build_chunks(doc, units)
            all_chunks.extend(chunks)
            if idx % 25 == 0 or idx == len(docs):
                LOGGER.info(
                    "Processed docs %d/%d | pages=%d | cumulative_chunks=%d",
                    idx,
                    len(docs),
                    page_count,
                    len(all_chunks),
                )

        chunk_rows = [dataclasses.asdict(c) for c in all_chunks]
        self._write_jsonl(self.chunks_jsonl, chunk_rows)
        self._write_parquet(self.chunks_parquet, chunk_rows)
        LOGGER.info("Chunks written: %s (%d rows)", self.chunks_jsonl, len(chunk_rows))

        lexical = LexicalIndex(self.lexical_db)
        lexical.rebuild(all_chunks)

        semantic_info: Dict[str, Any] = {
            "embedded": 0,
            "skipped_existing": 0,
            "pruned_stale": 0,
            "enabled": False,
        }
        collection_name = f"protocol_chunks_{embedding_model.replace('-', '_').replace('.', '_')}"
        if with_embeddings:
            semantic = SemanticIndex(
                qdrant_path=self.qdrant_dir,
                collection_name=collection_name,
                embedding_model=embedding_model,
                openai_api_key=openai_api_key,
                openai_base_url=openai_base_url,
            )
            semantic_info = semantic.build_embeddings(
                chunks=all_chunks,
                batch_size=embedding_batch_size,
                prune_stale=prune_stale_vectors,
            )
            semantic_info["enabled"] = True

        metadata = {
            "index_version": "v2",
            "ingest_date": utc_now_iso(),
            "corpus_dir": str(self.corpus_dir),
            "index_dir": str(self.index_dir),
            "doc_count": len(docs),
            "chunk_count": len(all_chunks),
            "ingest_stats": ingest_stats,
            "with_embeddings": with_embeddings,
            "embedding_model": embedding_model,
            "qdrant_collection": collection_name,
            "semantic_info": semantic_info,
            "duration_seconds": round(time.time() - t0, 2),
        }
        self.metadata_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        LOGGER.info("Build complete in %.2fs", time.time() - t0)
        return metadata

    def load_searcher(
        self,
        embedding_model: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
        enable_semantic: bool = True,
    ) -> HybridSearcher:
        lexical = LexicalIndex(self.lexical_db)
        if not enable_semantic:
            return HybridSearcher(lexical_index=lexical, semantic_index=None)
        if not openai_api_key:
            return HybridSearcher(lexical_index=lexical, semantic_index=None)

        semantic: Optional[SemanticIndex] = None
        model = embedding_model
        collection_name = None
        if self.metadata_json.exists():
            meta = json.loads(self.metadata_json.read_text(encoding="utf-8"))
            model = model or meta.get("embedding_model")
            collection_name = meta.get("qdrant_collection")
        if model and collection_name:
            try:
                semantic = SemanticIndex(
                    qdrant_path=self.qdrant_dir,
                    collection_name=collection_name,
                    embedding_model=model,
                    openai_api_key=openai_api_key,
                    openai_base_url=openai_base_url,
                )
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("Semantic backend unavailable, using lexical only: %s", exc)
                semantic = None
        return HybridSearcher(lexical_index=lexical, semantic_index=semantic)


def run_build(args: argparse.Namespace) -> None:
    openai_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
    openai_base = normalize_openai_base_url(
        args.openai_base_url or os.environ.get("OPENAI_BASE_URL")
    )
    if args.with_embeddings and not openai_key:
        raise RuntimeError("OPENAI_API_KEY is required when --with-embeddings is set")

    pipeline = Pipeline(
        corpus_dir=Path(args.corpus_dir).resolve(),
        index_dir=Path(args.index_dir).resolve(),
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        overlap=args.overlap,
    )
    metadata = pipeline.build(
        max_docs=args.max_docs,
        with_embeddings=args.with_embeddings,
        embedding_model=args.embedding_model,
        openai_api_key=openai_key,
        openai_base_url=openai_base,
        embedding_batch_size=args.embedding_batch_size,
        prune_stale_vectors=not args.no_prune_stale_vectors,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


def run_search(args: argparse.Namespace) -> None:
    openai_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
    openai_base = normalize_openai_base_url(
        args.openai_base_url or os.environ.get("OPENAI_BASE_URL")
    )
    pipeline = Pipeline(
        corpus_dir=Path(args.corpus_dir).resolve(),
        index_dir=Path(args.index_dir).resolve(),
    )
    searcher = pipeline.load_searcher(
        embedding_model=args.embedding_model,
        openai_api_key=openai_key,
        openai_base_url=openai_base,
    )
    result = searcher.search(query=args.query, top_k=args.top_k, candidate_k=args.candidate_k)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _normalize_expected_doc_ids(row: Dict[str, Any]) -> List[str]:
    candidates = (
        row.get("expected_doc_ids")
        or row.get("relevant_doc_ids")
        or row.get("doc_ids")
        or []
    )
    if isinstance(candidates, str):
        candidates = [candidates]
    out: List[str] = []
    if isinstance(candidates, list):
        for item in candidates:
            value = str(item).strip()
            if value:
                out.append(value)
    return sorted(set(out))


def _load_gold_queries(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Gold file not found: {path}")

    rows: List[Dict[str, Any]] = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                raw = line.strip()
                if not raw:
                    continue
                row = json.loads(raw)
                if not isinstance(row, dict):
                    continue
                query = str(row.get("query", "")).strip()
                if not query:
                    continue
                tags = row.get("tags") or []
                if isinstance(tags, str):
                    tags = [tags]
                rows.append(
                    {
                        "query": query,
                        "expected_doc_ids": _normalize_expected_doc_ids(row),
                        "tags": [str(t).strip().lower() for t in tags if str(t).strip()],
                    }
                )
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("queries", [])
        if not isinstance(payload, list):
            raise ValueError("Gold file must be JSON array or JSONL")
        for row in payload:
            if not isinstance(row, dict):
                continue
            query = str(row.get("query", "")).strip()
            if not query:
                continue
            tags = row.get("tags") or []
            if isinstance(tags, str):
                tags = [tags]
            rows.append(
                {
                    "query": query,
                    "expected_doc_ids": _normalize_expected_doc_ids(row),
                    "tags": [str(t).strip().lower() for t in tags if str(t).strip()],
                }
            )
    return rows


def _ranked_doc_ids(results: Sequence[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for row in results:
        doc_id = str(row.get("doc_id", "")).strip()
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        out.append(doc_id)
    return out


def _init_metric_bucket() -> Dict[str, Any]:
    return {"n": 0, "recall": 0.0, "mrr": 0.0, "top1": 0.0}


def _add_metric(bucket: Dict[str, Any], recall: float, rr: float, top1: float) -> None:
    bucket["n"] += 1
    bucket["recall"] += recall
    bucket["mrr"] += rr
    bucket["top1"] += top1


def _finalize_metric(bucket: Dict[str, Any]) -> Dict[str, Any]:
    n = int(bucket["n"])
    if n <= 0:
        return {"n": 0, "recall": 0.0, "mrr": 0.0, "top1_accuracy": 0.0}
    return {
        "n": n,
        "recall": round(float(bucket["recall"]) / n, 4),
        "mrr": round(float(bucket["mrr"]) / n, 4),
        "top1_accuracy": round(float(bucket["top1"]) / n, 4),
    }


def run_evaluate(args: argparse.Namespace) -> None:
    openai_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
    openai_base = normalize_openai_base_url(
        args.openai_base_url or os.environ.get("OPENAI_BASE_URL")
    )
    pipeline = Pipeline(
        corpus_dir=Path(args.corpus_dir).resolve(),
        index_dir=Path(args.index_dir).resolve(),
    )
    searcher = pipeline.load_searcher(
        embedding_model=args.embedding_model,
        openai_api_key=openai_key,
        openai_base_url=openai_base,
    )

    gold_rows = _load_gold_queries(Path(args.gold_file).resolve())
    total = _init_metric_bucket()
    mkb_slice = _init_metric_bucket()
    pediatric_slice = _init_metric_bucket()
    per_query: List[Dict[str, Any]] = []

    eval_fetch_k = max(args.candidate_k, args.top_k * 4, 30)
    for row in gold_rows:
        query = row["query"]
        expected = set(row["expected_doc_ids"])
        if not expected:
            continue

        result = searcher.search(query=query, top_k=eval_fetch_k, candidate_k=args.candidate_k)
        ranked = _ranked_doc_ids(result.get("results") or [])
        ranked_top = ranked[: args.top_k]

        recall = 1.0 if any(doc in expected for doc in ranked_top) else 0.0
        rr = 0.0
        for rank, doc_id in enumerate(ranked_top, start=1):
            if doc_id in expected:
                rr = 1.0 / rank
                break
        top1 = 1.0 if ranked_top and ranked_top[0] in expected else 0.0

        _add_metric(total, recall, rr, top1)

        tags = set(row.get("tags") or [])
        is_mkb = bool(MKB_RE.search(query.upper())) or "mkb" in tags or "мкб" in tags
        if is_mkb:
            _add_metric(mkb_slice, recall, rr, top1)

        q_lower = query.lower()
        is_pediatric = (
            any(marker in q_lower for marker in PEDIATRIC_MARKERS)
            or "pediatric" in tags
            or "дети" in tags
        )
        if is_pediatric:
            _add_metric(pediatric_slice, recall, rr, top1)

        if args.include_per_query:
            per_query.append(
                {
                    "query": query,
                    "expected_doc_ids": sorted(expected),
                    "predicted_doc_ids": ranked_top,
                    "recall": recall,
                    "rr": round(rr, 4),
                    "top1": top1,
                }
            )

    report = {
        "generated_at": utc_now_iso(),
        "gold_file": str(Path(args.gold_file).resolve()),
        "top_k": args.top_k,
        "candidate_k": args.candidate_k,
        "metrics": _finalize_metric(total),
        "slices": {
            "mkb": _finalize_metric(mkb_slice),
            "pediatric": _finalize_metric(pediatric_slice),
        },
    }
    if args.include_per_query:
        report["per_query"] = per_query

    if args.save_report:
        out_path = Path(args.save_report).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clinical protocol hybrid vector pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    default_corpus = (
        "/Users/ergalimabiev/Desktop/protocols/clinical_protocols_2026-03-06_041600"
    )
    default_index_dir = (
        "/Users/ergalimabiev/Desktop/protocols/_bmad-output/implementation-artifacts/pdf_vector_index"
    )

    p_build = sub.add_parser("build", help="Build manifest/chunks/lexical/vector indexes")
    p_build.add_argument("--corpus-dir", default=default_corpus)
    p_build.add_argument("--index-dir", default=default_index_dir)
    p_build.add_argument("--min-tokens", type=int, default=700)
    p_build.add_argument("--max-tokens", type=int, default=1100)
    p_build.add_argument("--overlap", type=int, default=120)
    p_build.add_argument("--max-docs", type=int, default=None)
    p_build.add_argument("--with-embeddings", action="store_true")
    p_build.add_argument("--embedding-model", default="text-embedding-3-small")
    p_build.add_argument("--embedding-batch-size", type=int, default=64)
    p_build.add_argument("--openai-api-key", default=None)
    p_build.add_argument("--openai-base-url", default=None)
    p_build.add_argument(
        "--no-prune-stale-vectors",
        action="store_true",
        help="Не удалять устаревшие points из Qdrant при rebuild.",
    )
    p_build.add_argument("--log-level", default="INFO")
    p_build.set_defaults(func=run_build)

    p_search = sub.add_parser("search", help="Run hybrid search against built index")
    p_search.add_argument("--query", required=True)
    p_search.add_argument("--corpus-dir", default=default_corpus)
    p_search.add_argument("--index-dir", default=default_index_dir)
    p_search.add_argument("--top-k", type=int, default=8)
    p_search.add_argument("--candidate-k", type=int, default=30)
    p_search.add_argument("--embedding-model", default=None)
    p_search.add_argument("--openai-api-key", default=None)
    p_search.add_argument("--openai-base-url", default=None)
    p_search.add_argument("--log-level", default="INFO")
    p_search.set_defaults(func=run_search)

    p_eval = sub.add_parser("evaluate", help="Evaluate retrieval quality on a gold dataset")
    p_eval.add_argument("--gold-file", required=True, help="JSON/JSONL with query + expected_doc_ids")
    p_eval.add_argument("--corpus-dir", default=default_corpus)
    p_eval.add_argument("--index-dir", default=default_index_dir)
    p_eval.add_argument("--top-k", type=int, default=10)
    p_eval.add_argument("--candidate-k", type=int, default=30)
    p_eval.add_argument("--embedding-model", default=None)
    p_eval.add_argument("--openai-api-key", default=None)
    p_eval.add_argument("--openai-base-url", default=None)
    p_eval.add_argument("--save-report", default=None)
    p_eval.add_argument("--include-per-query", action="store_true")
    p_eval.add_argument("--log-level", default="INFO")
    p_eval.set_defaults(func=run_evaluate)

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    setup_logging(level=args.log_level)
    args.func(args)


if __name__ == "__main__":
    main()
