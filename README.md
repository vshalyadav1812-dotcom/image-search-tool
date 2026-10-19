# Image Search Tool

A self-hosted image search application using semantic text queries, OCR, and reverse image search.

## Security setup (required)

The API is protected by a shared API key. The server fails closed: API requests return `503` until `IMAGE_SEARCH_API_KEY` is configured, and `401` for missing or incorrect keys. Never commit the key to source control.

### Linux/macOS

```bash
export IMAGE_SEARCH_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
export IMAGE_SEARCH_ALLOWED_ROOTS="$HOME/Pictures"
uvicorn main:app --host 127.0.0.1 --port 8000
```

### Windows PowerShell

```powershell
$env:IMAGE_SEARCH_API_KEY = python -c "import secrets; print(secrets.token_urlsafe(32))"
$env:IMAGE_SEARCH_ALLOWED_ROOTS = "$HOME\Pictures"
uvicorn main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`, enter the key in the UI, and click **Connect**. The browser stores it in `sessionStorage` for the current tab session. API clients must send `X-API-Key`.

`IMAGE_SEARCH_ALLOWED_ROOTS` is a list of permitted roots separated by the platform path separator (`:` on Linux/macOS, `; ` on Windows). If omitted, the user's home directory is allowed. Indexing and image retrieval are constrained to these roots. Use HTTPS and network controls before exposing the app beyond a trusted local machine. This shared key is for a trusted, single-user self-hosted tool, not multi-user identity management.

Optional: `MAX_UPLOAD_BYTES` sets the maximum reverse-search upload size (default 10 MiB). Search results are limited to 100.

## Features

- Semantic text-to-image search using Sentence Transformers / CLIP.
- Reverse image search from an uploaded image.
- Optional OCR text extraction.
- Background indexing of local image directories.
- Browser UI served by FastAPI.

## Installation

```bash
pip install -r requirements.txt
```

Install Tesseract OCR separately if OCR is needed.

## Run

Configure the environment variables above, then run:

```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```

Avoid `--host 0.0.0.0` unless remote access is intentional and protected by HTTPS, firewall rules, and a trusted gateway.

## API endpoints

Every API endpoint requires `X-API-Key: <configured-key>`.

- `POST /index`: JSON body `{"directory": "/path/to/images", "skip_ocr": false}`; starts indexing.
- `GET /status`: indexing status and indexed image count.
- `GET /search?query=...`: semantic search; `top_k` is limited to 1–100.
- `POST /search_image`: reverse image search using multipart form data; upload size is limited.
- `GET /image?path=...`: serves supported images only when inside an allowed root and present in the index.

## State and storage

ChromaDB persists its index in `./chroma_db` by default. Treat it as private because it contains image paths and derived metadata/embeddings. Back it up if needed and do not commit it.

## Tests

```bash
pip install pytest httpx
pytest -q
```
