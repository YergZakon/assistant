# Assistant API

API сервис ассистента клинических протоколов (hybrid retrieval + agentic workflow).

## Endpoints
- `GET /health`
- `POST /assist`
- `GET /protocol/<id>`

## Local run
```bash
python protocol_assistant.py --root . --backend hybrid --serve --host 0.0.0.0 --port 8080
```

## Build index
```bash
python pdf_hybrid_index.py build \
  --corpus-dir ./clinical_protocols_2026-03-06_041600 \
  --index-dir ./_bmad-output/implementation-artifacts/pdf_vector_index
```

С эмбеддингами:
```bash
python pdf_hybrid_index.py build \
  --with-embeddings \
  --embedding-model text-embedding-3-small \
  --corpus-dir ./clinical_protocols_2026-03-06_041600 \
  --index-dir ./_bmad-output/implementation-artifacts/pdf_vector_index
```

## Railway
1. Создайте отдельный сервис `assistant-api`.
2. Добавьте Volume с mount path `/data`.
3. Переменные: `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `EMBEDDING_MODEL`, `MIN_QUERY_WORDS`.
4. Один раз соберите индекс в `/data/pdf_vector_index` через Railway Shell.
