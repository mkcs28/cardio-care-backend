from fastapi import APIRouter, UploadFile, File, HTTPException
import shutil
from pathlib import Path
import uuid

from app.ocr_pipeline.ocr_engine import extract_text_with_boxes
from app.ocr_pipeline.field_mapper import map_fields

router = APIRouter()

# ✅ ALWAYS resolve upload directory relative to project root
BASE_DIR = Path(__file__).resolve().parents[2]  # cardio-care-backend/
UPLOAD_DIR = BASE_DIR / "uploads" / "reports"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/ocr/upload")
async def ocr_upload(file: UploadFile = File(...)):
    try:
        # ✅ Safe unique filename
        suffix = Path(file.filename).suffix
        filename = f"{uuid.uuid4()}{suffix}"
        path = UPLOAD_DIR / filename

        # ✅ Save file
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # ✅ OCR
        texts = extract_text_with_boxes(str(path))
        if not texts:
            raise HTTPException(status_code=400, detail="No text detected")

        # ✅ Field mapping
        mapped = map_fields(texts)

        return {
            "extracted_fields": mapped
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"OCR processing failed: {str(e)}"
        )
