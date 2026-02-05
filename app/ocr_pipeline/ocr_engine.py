import pytesseract
from PIL import Image

TESSERACT_CONFIG = "--oem 1 --psm 6"

def extract_text_with_boxes(image_path: str):
    """
    Lightweight OCR using Tesseract.
    Returns list of text lines.
    """
    try:
        image = Image.open(image_path)
        raw_text = pytesseract.image_to_string(
            image,
            lang="eng",
            config=TESSERACT_CONFIG
        )

        lines = [
            line.strip().lower()
            for line in raw_text.splitlines()
            if line.strip()
        ]
        return lines

    except Exception as e:
        print(f"OCR failed: {e}")
        return []
