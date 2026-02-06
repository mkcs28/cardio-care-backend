import re
from .preprocess import normalize

FIELD_PATTERNS = {
    "age": r"\bage\b\s*(\d{1,3})",
    "bmi": r"\bbmi\b\s*([\d.]+)",
    "whr": r"\bwhr\b\s*([\d.]+)",
    "fbs": r"\bfbs\b\s*([\d.]+)",
    "hba1c": r"\bhba1c\b\s*([\d.]+)",
    "hdl": r"\bhdl\b\s*([\d.]+)",
    "ldl": r"\bldl\b\s*([\d.]+)",
    "vldl": r"\bvldl\b\s*([\d.]+)",
    "tgl": r"\b(tgl|triglycerides)\b\s*([\d.]+)",
    "tc": r"\b(tc|total cholesterol)\b\s*([\d.]+)",
    "creatinine": r"\bcreatinine\b\s*([\d.]+)",
    "systolic": r"\bsystolic\b\s*bp\s*(\d{2,3})",
    "diastolic": r"\bdiastolic\b\s*bp\s*(\d{2,3})",
}

def map_fields(texts: list) -> dict:
    joined = normalize(" ".join(texts))
    output = {}

    for field, pattern in FIELD_PATTERNS.items():
        match = re.search(pattern, joined)
        if match:
            value = match.groups()[-1]
            try:
                output[field] = float(value)
            except:
                pass

    return output
