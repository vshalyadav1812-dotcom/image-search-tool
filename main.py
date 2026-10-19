import hmac
import io
import logging
import os
import threading
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from PIL import Image

from search_engine import ImageSearchEngine

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)
app = FastAPI(title="Image Search Tool")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
MAX_RESULTS = 100
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp", ".heic"}

_engine: Optional[ImageSearchEngine] = None
_engine_lock = threading.Lock()
_index_lock = threading.Lock()
is_indexing = False


def get_engine() -> ImageSearchEngine:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = ImageSearchEngine()
    return _engine


async def require_api_key(api_key: Optional[str] = Depends(api_key_header)) -> None:
    expected = os.getenv("IMAGE_SEARCH_API_KEY", "")
    if not expected:
        raise HTTPException(status_code=503, detail="Server authentication is not configured. Set IMAGE_SEARCH_API_KEY.")
    if not api_key or not hmac.compare_digest(api_key, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def allowed_roots():
    configured = os.getenv("IMAGE_SEARCH_ALLOWED_ROOTS", "").strip()
    raw_roots = configured.split(os.pathsep) if configured else [str(Path.home())]
    return [Path(root).expanduser().resolve() for root in raw_roots if root.strip()]


def is_within_allowed_root(path: Path) -> bool:
    return any(path == root or root in path.parents for root in allowed_roots())


def resolve_allowed_directory(raw_path: str) -> Path:
    try:
        candidate = Path(raw_path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError):
        raise HTTPException(status_code=400, detail="Directory not found")
    if not candidate.is_dir():
        raise HTTPException(status_code=400, detail="Directory not found")
    if not is_within_allowed_root(candidate):
        raise HTTPException(status_code=403, detail="Directory is outside configured image roots")
    return candidate


class IndexRequest(BaseModel):
    directory: str
    skip_ocr: bool = False


def background_index_task(directory: str, skip_ocr: bool):
    global is_indexing
    try:
        get_engine().index_directory(directory, skip_ocr)
    except Exception:
        logger.exception("Image indexing failed")
    finally:
        with _index_lock:
            is_indexing = False


@app.get("/", include_in_schema=False)
async def serve_index():
    index_path = Path(__file__).resolve().parent / "static" / "index.html"
    if not index_path.is_file():
        raise HTTPException(status_code=404, detail="Frontend not found")
    html = index_path.read_text(encoding="utf-8")
    html = html.replace("</body>", '<script src="/auth.js"></script></body>')
    return HTMLResponse(content=html)


@app.post("/index", dependencies=[Depends(require_api_key)])
async def index_images(request: IndexRequest, background_tasks: BackgroundTasks):
    global is_indexing
    directory = resolve_allowed_directory(request.directory)
    with _index_lock:
        if is_indexing:
            raise HTTPException(status_code=409, detail="Indexing is already in progress")
        is_indexing = True
    background_tasks.add_task(background_index_task, str(directory), request.skip_ocr)
    return {"message": "Indexing started in background."}


@app.get("/status", dependencies=[Depends(require_api_key)])
async def get_status():
    count = 0
    if _engine is not None:
        try:
            count = _engine.indexing_count if is_indexing else _engine.collection.count()
        except Exception:
            logger.exception("Unable to read index status")
    return {"is_indexing": is_indexing, "image_count": count}


@app.get("/search", dependencies=[Depends(require_api_key)])
async def search(
    query: str = Query(..., min_length=1, max_length=500),
    top_k: int = Query(5, ge=1, le=MAX_RESULTS),
    year: str = Query("", max_length=4),
):
    try:
        return {"results": get_engine().search(query, top_k=top_k, year_filter=year)}
    except Exception:
        logger.exception("Text image search failed")
        raise HTTPException(status_code=500, detail="Image search failed")


@app.post("/search_image", dependencies=[Depends(require_api_key)])
async def search_image(
    file: UploadFile = File(...),
    top_k: int = Query(5, ge=1, le=MAX_RESULTS),
    year: str = Query("", max_length=4),
):
    try:
        contents = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Image upload exceeds the size limit")
        try:
            with Image.open(io.BytesIO(contents)) as source:
                source.load()
                image = source.convert("RGB")
        except (OSError, ValueError, Image.DecompressionBombError):
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid supported image")
        return {"results": get_engine().search_by_image(image, top_k=top_k, year_filter=year)}
    except HTTPException:
        raise
    except Exception:
        logger.exception("Reverse image search failed")
        raise HTTPException(status_code=500, detail="Image search failed")
    finally:
        await file.close()


@app.get("/image", dependencies=[Depends(require_api_key)])
async def get_image(path: str = Query(..., min_length=1, max_length=4096)):
    try:
        image_path = Path(path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError):
        raise HTTPException(status_code=404, detail="Image not found")
    if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise HTTPException(status_code=404, detail="Image not found")
    if not is_within_allowed_root(image_path):
        raise HTTPException(status_code=403, detail="Image is outside configured image roots")
    try:
        indexed = get_engine().collection.get(where={"path": str(image_path)}, include=["metadatas"])
        indexed_paths = {
            Path(item["path"]).resolve()
            for item in (indexed.get("metadatas") or [])
            if item and item.get("path")
        }
    except Exception:
        logger.exception("Unable to verify image index membership")
        raise HTTPException(status_code=500, detail="Unable to retrieve image")
    if image_path not in indexed_paths:
        raise HTTPException(status_code=404, detail="Image not found in index")

    if image_path.suffix.lower() in {".tif", ".tiff", ".bmp"}:
        try:
            with Image.open(image_path) as source:
                img = source.convert("RGB")
                img_io = io.BytesIO()
                img.save(img_io, "JPEG", quality=85)
                img_io.seek(0)
            return StreamingResponse(img_io, media_type="image/jpeg")
        except Exception:
            logger.exception("Image conversion failed")
            raise HTTPException(status_code=415, detail="Image format could not be converted")
    return FileResponse(image_path)


static_dir = Path(__file__).resolve().parent / "static"
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
