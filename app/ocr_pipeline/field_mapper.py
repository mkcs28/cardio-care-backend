import re
from .preprocess import normalize

FIELD_PATTERNS = {
    "age": r"\bage\b.*?(\d{1,3})",
    "bmi": r"\bbmi\b.*?([\d.]+)",
    "whr": r"\bwhr\b.*?([\d.]+)",
    "fbs": r"\b(fbs|fasting blood sugar)\b.*?([\d.]+)",
    "hba1c": r"\b(hba1c|hb a1c)\b.*?([\d.]+)",
    "hdl": r"\bhdl\b.*?([\d.]+)",
    "ldl": r"\bldl\b.*?([\d.]+)",
    "vldl": r"\bvldl\b.*?([\d.]+)",
    "tgl": r"\b(tgl|triglycerides)\b.*?([\d.]+)",
    "tc": r"\b(tc|total cholesterol)\b.*?([\d.]+)",
    "creatinine": r"\bcreatinine\b.*?([\d.]+)",
    "systolic": r"\bsystolic\b.*?(\d{2,3})",
    "diastolic": r"\bdiastolic\b.*?(\d{2,3})",
}

def map_fields(texts: list) -> dict:
    joined = normalize(" ".join(texts))
    output = {}

    for field, pattern in FIELD_PATTERNS.items():
        match = re.search(pattern, joined, re.IGNORECASE)
        if match:
            value = match.groups()[-1]
            try:
                output[field] = float(value)
            except ValueError:
                pass

    return output
