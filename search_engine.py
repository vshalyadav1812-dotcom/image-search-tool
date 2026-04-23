from sentence_transformers import SentenceTransformer
from PIL import Image, ExifTags
import os
import chromadb
import numpy as np
from typing import List, Dict, Any
from ocr_service import OCRService
from pillow_heif import register_heif_opener

# Register HEIF opener to support Apple .heic images natively
register_heif_opener()

class ImageSearchEngine:
    def __init__(self, model_name='clip-ViT-B-32', db_path='./chroma_db'):
        self.model = SentenceTransformer(model_name)
        
        # Initialize robust vector database
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name="images_collection",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.ocr_service = OCRService()
        self.indexing_count = 0

    def _extract_exif_year(self, img: Image.Image) -> str:
        """Attempts to extract the year the photo was taken from EXIF data."""
        try:
            exif = img.getexif()
            if exif is not None:
                for tag_id, value in exif.items():
                    tag = ExifTags.TAGS.get(tag_id, tag_id)
                    if tag == 'DateTime' or tag == 'DateTimeOriginal':
                        # EXIF format: 'YYYY:MM:DD HH:MM:SS'
                        if isinstance(value, str) and len(value) >= 4:
                            return value[:4]
        except Exception:
            pass
        return ""

    def index_directory(self, directory_path: str, skip_ocr: bool = False):
        """
        Scans a directory for images, generates embeddings, extracts EXIF, and saves to ChromaDB.
        Uses upsert so re-indexing the same files just updates them.
        """
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff', '.webp', '.wemp', '.heic'}
        raw_paths = []
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in extensions:
                    raw_paths.append(os.path.join(root, file))
        
        print(f"Found {len(raw_paths)} potential images. Indexing to DB...")
        
        self.indexing_count = 0
        batch_size = 32
        
        # Process in batches to prevent memory overflow
        for i in range(0, len(raw_paths), batch_size):
            batch_paths = raw_paths[i:i+batch_size]
            
            valid_images = []
            valid_paths = []
            valid_metadatas = []
            
            for img_path in batch_paths:
                try:
                    img = Image.open(img_path)
                    img = img.convert('RGB')
                    img.load()
                    
                    year = self._extract_exif_year(img)
                    
                    text = ""
                    if not skip_ocr:
                        text = self.ocr_service.extract_text(img_path)
                    
                    valid_images.append(img)
                    valid_paths.append(img_path)
                    valid_metadatas.append({
                        "path": img_path,
                        "ocr_text": text,
                        "year": year
                    })
                    self.indexing_count += 1
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
            
            if valid_images:
                embeddings = self.model.encode(valid_images)
                
                # Upsert into ChromaDB
                self.collection.upsert(
                    ids=valid_paths,
                    embeddings=embeddings.tolist(),
                    metadatas=valid_metadatas
                )
        
        print(f"Indexing complete. Processed {self.indexing_count} files.")
        self.indexing_count = 0

    def search(self, query: str, top_k: int = 5, year_filter: str = "") -> List[Dict[str, Any]]:
        """
        Searches for images matching the query using ChromaDB.
        """
        if self.collection.count() == 0:
            return []

        query_embedding = self.model.encode(query)
        
        where_clause = None
        if year_filter:
            where_clause = {"year": year_filter}
            
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where_clause
        )
        
        output = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                # ChromaDB cosine returns distance (1 - cosine similarity). Back-convert to score.
                score = 1.0 - results['distances'][0][i]
                meta = results['metadatas'][0][i]
                
                # Apply OCR Boost
                ocr_text = meta.get('ocr_text', '')
                if query.lower() in ocr_text.lower():
                    score += 0.05
                    
                output.append({
                    "path": meta['path'],
                    "score": float(score),
                    "ocr_text": ocr_text,
                    "year": meta.get('year', '')
                })
                
        # Sort by boosted score
        output.sort(key=lambda x: x['score'], reverse=True)
        return output

    def search_by_image(self, image: Image.Image, top_k: int = 5, year_filter: str = "") -> List[Dict[str, Any]]:
        """
        Searches for visually similar images using ChromaDB.
        """
        if self.collection.count() == 0:
            return []

        query_embedding = self.model.encode(image)
        
        where_clause = None
        if year_filter:
            where_clause = {"year": year_filter}
            
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where_clause
        )
        
        output = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                score = 1.0 - results['distances'][0][i]
                meta = results['metadatas'][0][i]
                
                output.append({
                    "path": meta['path'],
                    "score": float(score),
                    "ocr_text": meta.get('ocr_text', ''),
                    "year": meta.get('year', '')
                })
                
        return output
