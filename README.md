# Image Search Tool

An intelligent, self-hosted web application that allows you to easily index and search through your local image collections using semantic text queries (powered by CLIP), optical character recognition (OCR), and reverse image search.

## Features

- **Semantic Text-to-Image Search**: Find images based on abstract concepts, descriptions, or objects within the images. Uses `sentence-transformers` (`clip-ViT-B-32`).
- **Reverse Image Search (Image-to-Image)**: Upload an image to find visually similar images within your indexed directory.
- **OCR Search Boost**: Automatically extracts text from images during indexing using Tesseract OCR. Text queries matching extracted text receive a relevance boost.
- **Background Indexing**: Asynchronously index large directories of images without blocking the main application.
- **Interactive Web UI**: A clean, single-page interface served by FastAPI for managing indexing and performing searches.

## Architecture

- **Backend**: FastAPI (Python) handles API requests, background task scheduling, and serves the static frontend.
- **Search Engine**: `SentenceTransformer` (`clip-ViT-B-32`) is used to generate image and text embeddings. Native PyTorch tensor operations compute cosine similarities.
- **OCR**: `pytesseract` wraps the Tesseract OCR engine to extract textual content from images.
- **Frontend**: Vanilla HTML/JS/CSS served directly from the `static/` directory.

## Prerequisites

1. **Python 3.8+**
2. **Tesseract OCR**: You must have Tesseract installed on your system.
   - **Windows**: Download the installer from [UB-Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki). You may need to add the installation directory to your system's `PATH`, or configure the path in `ocr_service.py` if it is not detected automatically:
     ```python
     pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
     ```
   - **macOS**: `brew install tesseract`
   - **Linux (Ubuntu/Debian)**: `sudo apt-install tesseract-ocr`

## Installation

1. Navigate to the project directory:
   ```bash
   cd path/to/image_search_tool
   ```
2. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the FastAPI server:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```
2. Open your web browser and navigate to `http://localhost:8000`.
3. **Indexing**: 
   - Before you can search, you need to index a directory of images.
   - Enter the absolute path to a folder containing images on your local machine and click the "Index" button. 
   - Depending on the number of images, this may take a few moments. The server runs the vision transformer and OCR on each image in the background.
4. **Search by Text**: Enter a descriptive phrase (e.g., "a sunny beach", "invoice document") into the search bar.
5. **Search by Image**: Use the file upload option in the UI to perform a reverse image search.

## API Endpoints

- `POST /index`: Triggers the background indexing of a given directory. `{"directory": "path/to/folder"}`
- `GET /status`: Returns the current indexing status (`is_indexing`) and number of indexed images.
- `GET /search?query="..."`: Returns a list of the most semantically relevant images and their matching scores.
- `POST /search_image`: Accepts an image file upload via `multipart/form-data` and returns visually similar images.
- `GET /image?path="..."`: Serves the raw image file for the frontend to render.

## State Management

When you index a directory, the tool automatically saves the embeddings and indexing metadata to `image_index.pt` and `image_index.pt.json` in the project root. Upon restarting the application, it will gracefully attempt to load this cached index so you don't have to re-index your images.
