import cv2
import pytesseract
import numpy as np
from PIL import Image

def extract_text_with_boxes(image_path: str):
    # Load image
    img = cv2.imread(image_path)

    if img is None:
        return []

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Increase contrast
    gray = cv2.equalizeHist(gray)

    # Adaptive threshold (critical for medical reports)
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        2,
    )

    # OCR config tuned for documents
    custom_config = r"--oem 3 --psm 6"

    text = pytesseract.image_to_string(thresh, config=custom_config)

    if not text or not text.strip():
        return []

    return [text.lower()]
