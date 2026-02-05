from fastapi import APIRouter, UploadFile, File, HTTPException
import shutil
from pathlib import Path
import uuid

router = APIRouter(prefix="/ocr", tags=["OCR"])

BASE_DIR = Path(__file__).resolve().parents[2]
UPLOAD_DIR = BASE_DIR / "uploads" / "reports"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/upload")
async def ocr_upload(file: UploadFile = File(...)):
    try:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Only image files allowed")

        suffix = Path(file.filename).suffix
        filename = f"{uuid.uuid4()}{suffix}"
        path = UPLOAD_DIR / filename

        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # ✅ Lazy imports (critical for memory)
        from app.ocr_pipeline.ocr_engine import extract_text_with_boxes
        from app.ocr_pipeline.field_mapper import map_fields

        texts = extract_text_with_boxes(str(path))
        if not texts:
            raise HTTPException(status_code=400, detail="No text detected")

        mapped = map_fields(texts)

        if not mapped:
            return {
                "extracted_fields": {},
                "warning": "OCR text detected but no fields matched"
            }

    return {"extracted_fields": mapped}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
