#!/usr/bin/env python3
"""Protocol assistant built around local treatment protocols.

Features:
- Builds SQLite FTS5 index from index.json + json/*.json files
- Finds best-matching protocol by free-text description
- Exposes HTTP API for website integration
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import parse_qs, unquote, urlparse

from hybrid_protocol_backend import HybridProtocolAssistant


TOKEN_RE = re.compile(r"[0-9a-zA-Zа-яА-ЯёЁ]+(?:\.[0-9a-zA-Zа-яА-ЯёЁ]+)?")
MKB_RE = re.compile(r"^[A-Za-z]\d{1,2}(?:\.\d{1,2})?$")
STOPWORDS = {
    "и",
    "в",
    "на",
    "по",
    "с",
    "со",
    "к",
    "из",
    "за",
    "у",
    "о",
    "от",
    "до",
    "при",
    "для",
    "или",
    "как",
    "что",
    "это",
    "не",
    "без",
    "после",
    "под",
    "а",
    "the",
    "and",
    "or",
    "with",
    "without",
}
DEFAULT_MIN_QUERY_WORDS = 10
QUERY_INPUT_EXAMPLE = (
    "Пациент 34 года, 3 дня кашель и боль в горле, температура 38.2, "
    "насморк, слабость, одышки нет, МКБ J06.9."
)


def count_query_words(text: str) -> int:
    return len(TOKEN_RE.findall(text or ""))


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and (
            (value.startswith('"') and value.endswith('"'))
            or (value.startswith("'") and value.endswith("'"))
        ):
            value = value[1:-1]
        if value == "":
            continue
        os.environ.setdefault(key, value)


class ProtocolAssistant:
    """Loads protocol data, builds index, and performs search."""

    def __init__(self, project_root: Path, db_path: Optional[Path] = None) -> None:
        self.project_root = project_root.resolve()
        self.index_file = self.project_root / "index.json"
        self.protocol_dir = self.project_root / "json"
        self.db_path = db_path or (
            self.project_root
            / "_bmad-output"
            / "implementation-artifacts"
            / "protocol_assistant.db"
        )
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def ensure_index(self, force_rebuild: bool = False) -> None:
        if force_rebuild or not self._is_index_current():
            self._rebuild_index()

    def _is_index_current(self) -> bool:
        if not self.db_path.exists():
            return False
        if not self.index_file.exists():
            raise FileNotFoundError(f"Missing catalog file: {self.index_file}")

        with self._connect() as conn:
            try:
                row = conn.execute(
                    "SELECT value FROM metadata WHERE key = 'catalog_mtime'"
                ).fetchone()
                row2 = conn.execute(
                    "SELECT value FROM metadata WHERE key = 'protocol_count'"
                ).fetchone()
            except sqlite3.OperationalError:
                return False

        if not row or not row2:
            return False

        catalog_mtime = str(int(self.index_file.stat().st_mtime))
        if row["value"] != catalog_mtime:
            return False

        with self.index_file.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        protocol_count = str(len(payload.get("protocols", [])))
        return row2["value"] == protocol_count

    def _rebuild_index(self) -> None:
        if not self.index_file.exists():
            raise FileNotFoundError(f"Missing catalog file: {self.index_file}")

        with self.index_file.open("r", encoding="utf-8") as fh:
            catalog = json.load(fh)
        protocols = catalog.get("protocols", [])

        with self._connect() as conn:
            conn.executescript(
                """
                DROP TABLE IF EXISTS protocols;
                DROP TABLE IF EXISTS protocols_fts;
                DROP TABLE IF EXISTS metadata;

                CREATE TABLE protocols (
                    rowid INTEGER PRIMARY KEY,
                    protocol_id TEXT NOT NULL UNIQUE,
                    title TEXT NOT NULL,
                    mkb_codes_json TEXT NOT NULL,
                    section TEXT,
                    version TEXT,
                    url TEXT NOT NULL,
                    source_file TEXT NOT NULL,
                    sections_json TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    full_text TEXT NOT NULL
                );

                CREATE TABLE metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE VIRTUAL TABLE protocols_fts USING fts5(
                    protocol_id UNINDEXED,
                    title,
                    mkb_codes,
                    summary,
                    full_text,
                    tokenize='unicode61 remove_diacritics 2'
                );
                """
            )

            inserted = 0
            for item in protocols:
                protocol_id = str(item.get("id", "")).strip()
                if not protocol_id:
                    continue

                source_file = str(item.get("file", "")).strip()
                protocol_path = self._resolve_protocol_path(source_file, protocol_id)
                record = self._load_protocol_record(protocol_path)

                title = str(record.get("title") or item.get("title") or "").strip()
                mkb_codes = record.get("mkb_codes") or item.get("mkb_codes") or []
                if not isinstance(mkb_codes, list):
                    mkb_codes = [str(mkb_codes)]
                mkb_codes = [str(x).strip() for x in mkb_codes if str(x).strip()]

                full_text = str(record.get("full_text") or "").strip()
                if not full_text:
                    content = record.get("content")
                    if isinstance(content, dict):
                        full_text = "\n".join(str(v) for v in content.values() if v)
                full_text = full_text.strip()

                content = record.get("content")
                section_keys = (
                    list(content.keys()) if isinstance(content, dict) else []
                )
                summary = self._extract_summary(record, full_text)

                cur = conn.execute(
                    """
                    INSERT INTO protocols (
                        protocol_id,
                        title,
                        mkb_codes_json,
                        section,
                        version,
                        url,
                        source_file,
                        sections_json,
                        summary,
                        full_text
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        protocol_id,
                        title or protocol_id,
                        json.dumps(mkb_codes, ensure_ascii=False),
                        str(item.get("section") or record.get("section") or ""),
                        str(item.get("version") or record.get("version") or ""),
                        str(item.get("url") or record.get("url") or ""),
                        source_file or protocol_path.name,
                        json.dumps(section_keys, ensure_ascii=False),
                        summary,
                        full_text,
                    ),
                )
                rowid = cur.lastrowid
                indexed_text = self._clip_text(full_text, 20000)
                conn.execute(
                    """
                    INSERT INTO protocols_fts (
                        rowid,
                        protocol_id,
                        title,
                        mkb_codes,
                        summary,
                        full_text
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        rowid,
                        protocol_id,
                        title or protocol_id,
                        " ".join(mkb_codes),
                        summary,
                        indexed_text,
                    ),
                )
                inserted += 1

            conn.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
                ("catalog_mtime", str(int(self.index_file.stat().st_mtime))),
            )
            conn.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
                ("protocol_count", str(inserted)),
            )
            conn.commit()

    def _resolve_protocol_path(self, source_file: str, protocol_id: str) -> Path:
        if source_file:
            candidate = self.protocol_dir / source_file
            if candidate.exists():
                return candidate

        matches = sorted(self.protocol_dir.glob(f"{protocol_id}_*.json"))
        if matches:
            return matches[0]
        raise FileNotFoundError(
            f"Protocol file not found for id={protocol_id} file={source_file}"
        )

    @staticmethod
    def _load_protocol_record(path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    @staticmethod
    def _extract_summary(record: Dict[str, Any], full_text: str) -> str:
        content = record.get("content")
        if isinstance(content, dict):
            for preferred_key in ("Краткое описание", "Общая информация"):
                value = content.get(preferred_key)
                if isinstance(value, str) and value.strip():
                    return ProtocolAssistant._clip_text(value.strip(), 700)
            for value in content.values():
                if isinstance(value, str) and value.strip():
                    return ProtocolAssistant._clip_text(value.strip(), 700)
        return ProtocolAssistant._clip_text(full_text, 700)

    @staticmethod
    def _clip_text(value: str, limit: int) -> str:
        if len(value) <= limit:
            return value
        return value[:limit].rstrip() + "..."

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [token.casefold() for token in TOKEN_RE.findall(text)]

    def _build_fts_query(self, query: str) -> str:
        tokens = self._prepare_query_tokens(query)[:12]
        if not tokens:
            return '""'

        prepared_terms: List[str] = []
        for token in tokens:
            safe = token.replace('"', "")
            if not safe:
                continue
            if MKB_RE.fullmatch(safe.upper()):
                prepared_terms.append(f'"{safe.upper()}"')
            elif not re.fullmatch(r"[0-9A-Za-zА-Яа-яЁё]+", safe):
                prepared_terms.append(f'"{safe}"')
            elif len(safe) <= 2:
                prepared_terms.append(f'"{safe}"')
            else:
                prepared_terms.append(f"{safe}*")

        if not prepared_terms:
            return '""'
        return " OR ".join(prepared_terms)

    def _prepare_query_tokens(self, query: str) -> List[str]:
        tokens = [
            token
            for token in self._tokenize(query)
            if len(token) > 2 and token not in STOPWORDS
        ]
        if not tokens:
            tokens = self._tokenize(query)[:8]
        return tokens

    @staticmethod
    def _token_match(query_token: str, candidate_token: str) -> bool:
        if query_token == candidate_token:
            return True
        min_len = min(len(query_token), len(candidate_token))
        if min_len >= 5 and query_token[:5] == candidate_token[:5]:
            return True
        if min_len >= 6 and (query_token in candidate_token or candidate_token in query_token):
            return True
        return False

    def _overlap_ratio(self, query_tokens: List[str], candidate_text: str) -> float:
        if not query_tokens:
            return 0.0
        candidate_tokens = self._tokenize(candidate_text)
        if not candidate_tokens:
            return 0.0

        matched = 0
        for token in query_tokens:
            if any(self._token_match(token, c_token) for c_token in candidate_tokens):
                matched += 1
        return matched / len(query_tokens)

    @staticmethod
    def _phrase_bonus(query: str, candidate_text: str) -> float:
        q_tokens = [token.casefold() for token in TOKEN_RE.findall(query)]
        if len(q_tokens) < 2:
            return 0.0
        candidate = candidate_text.casefold()
        phrases = [f"{q_tokens[i]} {q_tokens[i + 1]}" for i in range(len(q_tokens) - 1)]
        hit_count = sum(1 for phrase in phrases if phrase in candidate)
        return min(hit_count * 0.15, 0.45)

    def _rerank_results(
        self,
        query: str,
        query_tokens: List[str],
        results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        for result in results:
            title_score = self._overlap_ratio(query_tokens, result["title"])
            summary_score = self._overlap_ratio(query_tokens, result["summary"])
            snippet_score = self._overlap_ratio(query_tokens, result["snippet"])
            mkb_score = self._mkb_overlap_score(query_tokens, result["mkb_codes"])
            phrase_score = self._phrase_bonus(
                query, f'{result["title"]} {result["summary"]} {result["snippet"]}'
            )
            bm25_score = float(result["bm25_score"])
            bm25_signal = 1.0 / (1.0 + abs(bm25_score))

            final_score = (
                0.30 * title_score
                + 0.30 * summary_score
                + 0.20 * mkb_score
                + 0.10 * snippet_score
                + 0.10 * phrase_score
                + 0.10 * bm25_signal
            )

            result["relevance"] = round(min(final_score, 1.0), 4)
            result["_sort_score"] = final_score

        results.sort(
            key=lambda item: (
                item.get("_sort_score", 0.0),
                -float(item.get("bm25_score", 0.0)),
            ),
            reverse=True,
        )
        for item in results:
            item.pop("_sort_score", None)
        return results

    @staticmethod
    def _mkb_overlap_score(query_tokens: List[str], mkb_codes: List[str]) -> float:
        if not query_tokens:
            return 0.0
        query_codes = [token.upper() for token in query_tokens if MKB_RE.fullmatch(token.upper())]
        if not query_codes:
            return 0.0
        if not mkb_codes:
            return 0.0
        mkb_upper = [str(code).upper() for code in mkb_codes]
        matched = 0
        for q_code in query_codes:
            if any(q_code == code or code.startswith(q_code) for code in mkb_upper):
                matched += 1
        return matched / len(query_codes)

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        query = (query or "").strip()
        if not query:
            return []

        self.ensure_index()
        fts_query = self._build_fts_query(query)
        limit = max(1, min(int(limit), 20))
        candidate_limit = max(limit * 12, 40)
        query_tokens = self._prepare_query_tokens(query)

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    p.protocol_id,
                    p.title,
                    p.mkb_codes_json,
                    p.url,
                    p.source_file,
                    p.summary,
                    p.sections_json,
                    snippet(protocols_fts, 3, '[', ']', ' ... ', 28) AS snippet,
                    bm25(protocols_fts, 0.0, 20.0, 6.0, 12.0, 1.0) AS bm25_score
                FROM protocols_fts
                JOIN protocols p ON p.rowid = protocols_fts.rowid
                WHERE protocols_fts MATCH ?
                ORDER BY bm25_score ASC
                LIMIT ?
                """,
                (fts_query, candidate_limit),
            ).fetchall()

            if not rows:
                fallback_like = f"%{query.casefold()}%"
                rows = conn.execute(
                    """
                    SELECT
                        protocol_id,
                        title,
                        mkb_codes_json,
                        url,
                        source_file,
                        summary,
                        sections_json,
                        '' AS snippet,
                        999.0 AS bm25_score
                    FROM protocols
                    WHERE lower(title) LIKE ?
                    ORDER BY length(title) ASC
                    LIMIT ?
                    """,
                    (fallback_like, candidate_limit),
                ).fetchall()

        results = [self._row_to_result(row) for row in rows]
        results = self._rerank_results(query, query_tokens, results)
        return results[:limit]

    @staticmethod
    def _row_to_result(row: sqlite3.Row) -> Dict[str, Any]:
        score = float(row["bm25_score"])
        if score < 0:
            relevance = 1.0
        else:
            relevance = 1.0 / (1.0 + score)
        return {
            "id": row["protocol_id"],
            "title": row["title"],
            "mkb_codes": json.loads(row["mkb_codes_json"]),
            "url": row["url"],
            "file": row["source_file"],
            "summary": row["summary"],
            "sections": json.loads(row["sections_json"]),
            "snippet": row["snippet"] or "",
            "relevance": round(relevance, 4),
            "bm25_score": round(score, 4),
        }

    def get_protocol(self, protocol_id: str, include_full_text: bool = False) -> Dict[str, Any]:
        protocol_id = (protocol_id or "").strip()
        if not protocol_id:
            raise ValueError("protocol_id is required")

        self.ensure_index()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT protocol_id, title, mkb_codes_json, url, source_file,
                       sections_json, summary
                FROM protocols
                WHERE protocol_id = ?
                """,
                (protocol_id,),
            ).fetchone()
        if not row:
            raise KeyError(f"Protocol not found: {protocol_id}")

        file_path = self._resolve_protocol_path(row["source_file"], protocol_id)
        payload = self._load_protocol_record(file_path)

        response = {
            "id": row["protocol_id"],
            "title": row["title"],
            "mkb_codes": json.loads(row["mkb_codes_json"]),
            "url": row["url"],
            "file": row["source_file"],
            "summary": row["summary"],
            "sections": json.loads(row["sections_json"]),
            "content": payload.get("content", {}),
            "version": payload.get("version", ""),
        }
        if include_full_text:
            response["full_text"] = payload.get("full_text", "")
        return response

    def protocol_count(self) -> int:
        self.ensure_index()
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS cnt FROM protocols").fetchone()
            return int(row["cnt"]) if row else 0


class ProtocolAPIHandler(BaseHTTPRequestHandler):
    assistant: ProtocolAssistant
    workflow_engine: Any = None
    min_query_words: int = DEFAULT_MIN_QUERY_WORDS

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/health":
            self._send_json(
                200,
                {
                    "status": "ok",
                    "indexed_protocols": self.assistant.protocol_count(),
                },
            )
            return

        if path.startswith("/protocol/"):
            protocol_id = unquote(path.split("/protocol/", 1)[1]).strip()
            include_full = parse_qs(parsed.query).get("full", ["false"])[0].lower() in {
                "1",
                "true",
                "yes",
            }
            try:
                protocol = self.assistant.get_protocol(protocol_id, include_full_text=include_full)
            except KeyError:
                self._send_json(404, {"error": "Protocol not found"})
                return
            self._send_json(200, protocol)
            return

        self._send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/assist":
            self._send_json(404, {"error": "Not found"})
            return

        try:
            payload = self._read_json()
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return

        query = str(payload.get("query") or payload.get("description") or "").strip()
        if not query:
            self._send_json(400, {"error": "Field 'query' is required"})
            return
        query_word_count = count_query_words(query)
        if query_word_count < self.min_query_words:
            self._send_json(
                400,
                {
                    "error": (
                        f"Описание слишком короткое: {query_word_count} слов(а). "
                        f"Нужно минимум {self.min_query_words} слов."
                    ),
                    "min_query_words": self.min_query_words,
                    "query_word_count": query_word_count,
                    "input_example": QUERY_INPUT_EXAMPLE,
                },
            )
            return

        top_k = payload.get("top_k", 3)
        try:
            top_k_int = int(top_k)
        except (TypeError, ValueError):
            top_k_int = 3

        mode = str(payload.get("mode") or "agentic").strip().lower()
        include_trace = str(payload.get("include_trace", "true")).lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        raw_answers = payload.get("clarification_answers")
        clarification_answers: Dict[str, Any] = {}
        if isinstance(raw_answers, dict):
            clarification_answers = {str(k): v for k, v in raw_answers.items()}

        if mode != "search-only" and self.workflow_engine is not None:
            result = self.workflow_engine.run(
                query=query,
                top_k=top_k_int,
                include_trace=include_trace,
                clarification_answers=clarification_answers,
            )
            self._send_json(200, result)
            return

        results = self.assistant.search(query, limit=top_k_int)
        self._send_json(
            200,
            {
                "query": query,
                "top_match": results[0] if results else None,
                "results": results,
                "total_results": len(results),
            },
        )

    def _read_json(self) -> Dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(content_length) if content_length > 0 else b""
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON: {exc}") from exc

    def _send_json(self, status_code: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Iterable[Any]) -> None:
        # Keep default logs minimal for production terminal usage.
        return


def run_server(
    assistant: ProtocolAssistant,
    host: str,
    port: int,
    workflow_config: Optional[Path] = None,
) -> None:
    from assistant_system.agentic_workflow import AgentWorkflowEngine

    class _Handler(ProtocolAPIHandler):
        pass

    _Handler.assistant = assistant
    raw_min_words = str(os.environ.get("MIN_QUERY_WORDS", DEFAULT_MIN_QUERY_WORDS)).strip()
    try:
        _Handler.min_query_words = max(1, int(raw_min_words))
    except ValueError:
        _Handler.min_query_words = DEFAULT_MIN_QUERY_WORDS
    _Handler.workflow_engine = AgentWorkflowEngine(
        search_backend=assistant,
        workflow_path=workflow_config,
    )
    server = ThreadingHTTPServer((host, port), _Handler)
    print(f"Protocol Assistant API started on http://{host}:{port}")
    print(
        "Endpoints: POST /assist (agentic by default), GET /protocol/<id>, GET /health"
    )
    print(
        f"Input policy: minimum {_Handler.min_query_words} words in 'query' (set MIN_QUERY_WORDS to change)."
    )
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Protocol assistant for treatment protocol lookup"
    )
    parser.add_argument(
        "--backend",
        default="hybrid",
        choices=["hybrid", "legacy"],
        help="Search backend: hybrid (PDF vector+BM25) or legacy (index.json/json)",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Project root path (default: current directory)",
    )
    parser.add_argument(
        "--db",
        default="",
        help="SQLite DB path (default: _bmad-output/implementation-artifacts/protocol_assistant.db)",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Force rebuilding search index",
    )
    parser.add_argument(
        "--query",
        default="",
        help="Run one search query and print JSON result",
    )
    parser.add_argument(
        "--agentic",
        action="store_true",
        help="Run --query via agentic workflow (BMAD-style multi-agent pipeline)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of results for --query or /assist (default: 3)",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Run HTTP API server",
    )
    parser.add_argument(
        "--workflow-config",
        default="assistant_system/workflows/protocol-assistant.workflow.json",
        help="Path to agentic workflow config JSON",
    )
    parser.add_argument(
        "--corpus-dir",
        default="",
        help="PDF corpus directory for hybrid backend",
    )
    parser.add_argument(
        "--index-dir",
        default="",
        help="Hybrid index directory (manifest/chunks/fts/qdrant)",
    )
    parser.add_argument(
        "--embedding-model",
        default="",
        help="Embedding model name for hybrid semantic search (if available)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    load_env_file(root / ".env")
    backend = str(args.backend).strip().lower()

    if backend == "legacy":
        db_path = Path(args.db).resolve() if args.db else None
        assistant: Any = ProtocolAssistant(project_root=root, db_path=db_path)
    else:
        corpus_dir = Path(args.corpus_dir).resolve() if args.corpus_dir else None
        index_dir = Path(args.index_dir).resolve() if args.index_dir else None
        embedding_model = args.embedding_model.strip() or None
        assistant = HybridProtocolAssistant(
            project_root=root,
            corpus_dir=corpus_dir,
            index_dir=index_dir,
            embedding_model=embedding_model,
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            openai_base_url=os.environ.get("OPENAI_BASE_URL"),
        )

    assistant.ensure_index(force_rebuild=args.rebuild_index)

    workflow_config = Path(args.workflow_config).resolve()

    if args.query and args.agentic:
        from assistant_system.agentic_workflow import AgentWorkflowEngine

        engine = AgentWorkflowEngine(search_backend=assistant, workflow_path=workflow_config)
        result = engine.run(args.query, top_k=args.top_k, include_trace=True)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.query:
        result = {
            "query": args.query,
            "results": assistant.search(args.query, limit=args.top_k),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.serve:
        run_server(assistant, args.host, args.port, workflow_config=workflow_config)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
