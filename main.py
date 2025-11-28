from fastapi import FastAPI, HTTPException, Query, UploadFile, File, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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

is_indexing = False

def background_index_task(directory: str):
    global is_indexing
    is_indexing = True
    try:
        engine.index_directory(directory)
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
    
    background_tasks.add_task(background_index_task, request.directory)
    return {"message": "Indexing started in background."}

@app.get("/status")
async def get_status():
    return {
        "is_indexing": is_indexing,
        "image_count": len(engine.image_paths)
    }

@app.get("/search")
async def search(query: str = Query(..., min_length=1)):
    try:
        results = engine.search(query)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search_image")
async def search_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        results = engine.search_by_image(image)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/image")
async def get_image(path: str):
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path)

# Mount static files for frontend
# Ensure the static directory exists
os.makedirs("static", exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
