from sentence_transformers import SentenceTransformer, util
from PIL import Image
import os
import glob
import torch
import numpy as np
import json
from typing import List, Dict, Any
from ocr_service import OCRService

class ImageSearchEngine:
    def __init__(self, model_name='clip-ViT-B-32', index_path='image_index.pt'):
        self.model = SentenceTransformer(model_name)
        self.index_path = index_path
        self.image_paths = []
        self.embeddings = None
        self.ocr_service = OCRService()
        self.ocr_texts = [] # Store OCR text for text-based fallback or hybrid search
        self.load_index()

    def save_index(self):
        print(f"Saving index to {self.index_path}...")
        data = {
            "image_paths": self.image_paths,
            "ocr_texts": self.ocr_texts
        }
        torch.save(self.embeddings, self.index_path)
        with open(self.index_path + ".json", 'w') as f:
            json.dump(data, f)
        print("Index saved.")

    def load_index(self):
        if os.path.exists(self.index_path) and os.path.exists(self.index_path + ".json"):
            print(f"Loading index from {self.index_path}...")
            try:
                self.embeddings = torch.load(self.index_path)
                with open(self.index_path + ".json", 'r') as f:
                    data = json.load(f)
                    self.image_paths = data["image_paths"]
                    self.ocr_texts = data["ocr_texts"]
                print(f"Index loaded. {len(self.image_paths)} images.")
                return True
            except Exception as e:
                print(f"Failed to load index: {e}")
        return False

    def index_directory(self, directory_path: str):
        """
        Scans a directory for images, generates embeddings, and extracts text.
        """
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.tif', '*.tiff', '*.webp', '*.wemp']
        raw_paths = []
        for ext in extensions:
            raw_paths.extend(glob.glob(os.path.join(directory_path, ext)))
        
        print(f"Found {len(raw_paths)} potential images. Indexing...")
        
        images = []
        self.image_paths = [] # Reset and only store valid paths
        self.ocr_texts = []
        
        for img_path in raw_paths:
            try:
                img = Image.open(img_path)
                # Force load to ensure it's valid
                img.load()
                
                # Extract text for potential text search
                text = self.ocr_service.extract_text(img_path)
                
                # Only append if everything succeeded
                images.append(img)
                self.image_paths.append(img_path)
                self.ocr_texts.append(text)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")

        if images:
            self.embeddings = self.model.encode(images, convert_to_tensor=True)
            print(f"Indexing complete. {len(self.image_paths)} valid images indexed.")
            self.save_index()
        else:
            print("No valid images found to index.")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Searches for images matching the query (semantic + OCR).
        """
        if self.embeddings is None or len(self.image_paths) == 0:
            return []

        # 1. Semantic Search (Text-to-Image)
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, self.embeddings)[0]
        
        # 2. OCR Search (Simple keyword matching boost)
        # We'll add a small boost to the score if the query word appears in the OCR text
        ocr_boost = torch.zeros_like(cos_scores)
        query_lower = query.lower()
        
        for idx, text in enumerate(self.ocr_texts):
            # Simple substring match can be noisy. 
            # TODO: Implement whole word matching or regex for better precision.
            if query_lower in text.lower():
                ocr_boost[idx] = 0.05 # Reduced from 0.5 to avoid overpowering semantic search
        
        final_scores = cos_scores + ocr_boost

        # Get top_k results
        top_results = torch.topk(final_scores, k=min(top_k, len(self.image_paths)))
        
        results = []
        print(f"Search Query: '{query}'")
        for score, idx in zip(top_results.values, top_results.indices):
            path = self.image_paths[idx]
            ocr_text_preview = self.ocr_texts[idx][:50] + "..." if self.ocr_texts[idx] else "No text"
            print(f" - {path}: Score {score:.4f} (Semantic: {cos_scores[idx]:.4f}, OCR Boost: {ocr_boost[idx]:.4f}) Text: {ocr_text_preview}")
            
            results.append({
                "path": path,
                "score": float(score),
                "ocr_text": self.ocr_texts[idx]
            })
            
        return results

    def search_by_image(self, image: Image.Image, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Searches for images visually similar to the input image.
        """
        if self.embeddings is None or len(self.image_paths) == 0:
            return []

        # Generate embedding for the query image
        query_embedding = self.model.encode(image, convert_to_tensor=True)
        
        # Compute cosine similarity
        cos_scores = util.cos_sim(query_embedding, self.embeddings)[0]
        
        # Get top_k results
        top_results = torch.topk(cos_scores, k=min(top_k, len(self.image_paths)))
        
        results = []
        for score, idx in zip(top_results.values, top_results.indices):
            results.append({
                "path": self.image_paths[idx],
                "score": float(score),
                "ocr_text": self.ocr_texts[idx]
            })
            
        return results
