from fastapi import FastAPI, HTTPException, Query, UploadFile, File, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from search_engine import ImageSearchEngine
from PIL import Image
import io
import os

app = FastAPI()

# Initialize Search Engine
# We initialize it lazily or empty first
engine = ImageSearchEngine()

class IndexRequest(BaseModel):
    directory: str
    skip_ocr: bool = False

is_indexing = False

def background_index_task(directory: str, skip_ocr: bool):
    global is_indexing
    try:
        engine.index_directory(directory, skip_ocr)
    except Exception as e:
        print(f"Indexing failed: {e}")
    finally:
        is_indexing = False

@app.post("/index")
async def index_images(request: IndexRequest, background_tasks: BackgroundTasks):
    global is_indexing
    if is_indexing:
        return {"message": "Indexing is already in progress."}
    
    if not os.path.isdir(request.directory):
        raise HTTPException(status_code=400, detail="Directory not found")
    
    is_indexing = True
    background_tasks.add_task(background_index_task, request.directory, request.skip_ocr)
    return {"message": "Indexing started in background."}

@app.get("/status")
async def get_status():
    try:
        count = engine.indexing_count if is_indexing else engine.collection.count()
    except Exception:
        count = 0
    return {
        "is_indexing": is_indexing,
        "image_count": count
    }

@app.get("/search")
async def search(query: str = Query(..., min_length=1), top_k: int = 5, year: str = ""):
    try:
        results = engine.search(query, top_k=top_k, year_filter=year)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search_image")
async def search_image(file: UploadFile = File(...), top_k: int = 5, year: str = ""):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        results = engine.search_by_image(image, top_k=top_k, year_filter=year)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/image")
async def get_image(path: str):
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image not found")
        
    ext = os.path.splitext(path)[1].lower()
    # Web browsers cannot natively render TIFF formats (and struggle with some BMPs).
    if ext in {'.tif', '.tiff', '.bmp'}:
        try:
            img = Image.open(path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_io = io.BytesIO()
            img.save(img_io, 'JPEG', quality=85)
            img_io.seek(0)
            return StreamingResponse(img_io, media_type="image/jpeg")
        except Exception:
            pass # Fallback to returning the raw file if conversion somehow fails
            
    return FileResponse(path)

# Mount static files for frontend
# Ensure the static directory exists
os.makedirs("static", exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
