import re
from .preprocess import normalize

# ======================================================
# FIELD REGEX PATTERNS (ROBUST)
# ======================================================

FIELD_PATTERNS = {
    "age": r"age\s*(\d+)",
    "bmi": r"bmi\s*([\d.]+)",
    "whr": r"whr\s*([\d.]+)",
    "hba1c": r"hba1c\s*([\d.]+)",
    "fbs": r"fbs\s*(\d+)",
    "hdl": r"hdl\s*(\d+)",
    "ldl": r"ldl\s*(\d+)",
    "vldl": r"vldl\s*(\d+)",
    "tc": r"(total cholesterol|tc)\s*(\d+)",
    "tgl": r"(triglycerides|tgl)\s*(\d+)",
    "creatinine": r"creatinine\s*([\d.]+)",
    "systolic": r"systolic\s*bp\s*(\d+)",
    "diastolic": r"diastolic\s*bp\s*(\d+)",
}

# ======================================================
# MAIN MAPPER
# ======================================================

def map_fields(texts: list) -> dict:
    """
    Robust mapper:
    - Works for table reports
    - Works for inline text
    - Ignores units (mg/dL, %, mmHg)
    """

    if not texts:
        return {}

    text = normalize(" ".join(texts))

    # Remove units explicitly
    text = re.sub(r"(mg\/dl|mmhg|%)", "", text)

    output = {}

    for field, pattern in FIELD_PATTERNS.items():
        match = re.search(pattern, text)
        if match:
            # last group is always the numeric value
            value = match.groups()[-1]
            try:
                output[field] = float(value)
            except ValueError:
                pass

    return output
