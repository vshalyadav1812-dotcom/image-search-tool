import pytesseract
from PIL import Image
import os

class OCRService:
    def __init__(self):
        # Assuming tesseract is in the PATH. If not, user might need to configure it.
        # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        pass

    def extract_text(self, image_path: str) -> str:
        """
        Extracts text from an image file using Tesseract OCR.
        """
        try:
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image)
            return text.strip()
        except Exception as e:
            print(f"Error extracting text from {image_path}: {e}")
            return ""
