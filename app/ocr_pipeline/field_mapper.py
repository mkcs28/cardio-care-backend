import re
from .schema import EXPECTED_FIELDS
from .preprocess import normalize

FIELD_PATTERNS = {
    "age": r"age\s*[:\-]?\s*(\d{1,3})",
    "bmi": r"bmi\s*[:\-]?\s*([\d.]+)",
    "whr": r"whr\s*[:\-]?\s*([\d.]+)",
    "fbs": r"fbs\s*[:\-]?\s*([\d.]+)",
    "hba1c": r"hba1c\s*[:\-]?\s*([\d.]+)",
    "hdl": r"hdl\s*[:\-]?\s*([\d.]+)",
    "ldl": r"ldl\s*[:\-]?\s*([\d.]+)",
    "vldl": r"vldl\s*[:\-]?\s*([\d.]+)",
    "tgl": r"(tgl|triglycerides)\s*[:\-]?\s*([\d.]+)",
    "tc": r"(tc|total cholesterol)\s*[:\-]?\s*([\d.]+)",
    "creatinine": r"creatinine\s*[:\-]?\s*([\d.]+)",

    # ✅ FIXED BP (no slash required)
    "systolic": r"systolic\s*bp\s*[:\-]?\s*(\d{2,3})",
    "diastolic": r"diastolic\s*bp\s*[:\-]?\s*(\d{2,3})",
}

def map_fields(texts: list) -> dict:
    """
    HARD FIELD MAPPER:
    BMI → BMI only
    HDL → HDL only
    """
    joined = normalize(" ".join(texts))
    output = {}

    for field, pattern in FIELD_PATTERNS.items():
        match = re.search(pattern, joined)
        if match:
            value = match.groups()[-1]
            try:
                output[field] = float(value)
            except ValueError:
                pass

    return output
