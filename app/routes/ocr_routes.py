from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import shutil
import uuid

from app.ocr_pipeline.ocr_engine import extract_text_with_boxes
from app.ocr_pipeline.field_mapper import map_fields

# ✅ PREFIX FIX (IMPORTANT)
router = APIRouter(prefix="/ocr", tags=["OCR"])

# ✅ SAFE ABSOLUTE UPLOAD PATH
BASE_DIRF = Path(__file__).resolve().parents[2]
UPLOAD_DIR = BASEF / "uploads" / "reports"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/upload")
async def ocr_upload(file: UploadFile = File(...)):

    # ✅ BASIC VALIDATION
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files allowed")

    # ✅ SAFE UNIQUE FILENAME
    suffix = Path(file.filename).suffix
    filename = f"{uuid.uuid4().hex}{suffix}"
    path = UPLOAD_DIR / filename

    # ✅ SAVE IMAGE
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # ✅ OCR PIPELINE
    texts_with_boxes = extract_text_with_boxes(str(path))
    mapped_fields = map_fields(texts_with_boxes)

    return {
        "extracted_fields": mapped_fields
    }
