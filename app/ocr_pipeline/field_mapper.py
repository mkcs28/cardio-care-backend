# field_mapper.py
import re
from .preprocess import normalize

FIELD_ALIASES = {
    "age": ["age"],
    "bmi": ["bmi"],
    "whr": ["whr"],
    "fbs": ["fbs", "fasting blood sugar"],
    "hba1c": ["hba1c", "hbale", "hbaic"],
    "hdl": ["hdl"],
    "ldl": ["ldl"],
    "vldl": ["vldl"],
    "tgl": ["tgl", "triglycerides"],
    "tc": ["tc", "total cholesterol"],
    "creatinine": ["creatinine"],
    "systolic": ["systolic"],
    "diastolic": ["diastolic"],
}

NUMBER_PATTERN = re.compile(r"\d+(\.\d+)?")

def map_fields(texts: list) -> dict:
    text = normalize(" ".join(texts))
    output = {}

    tokens = text.split()

    for field, aliases in FIELD_ALIASES.items():
        for alias in aliases:
            if alias in tokens:
                idx = tokens.index(alias)
                for j in range(idx + 1, min(idx + 6, len(tokens))):
                    if NUMBER_PATTERN.fullmatch(tokens[j]):
                        value = float(tokens[j])

                        # ✅ CRITICAL FIX
                        if field in ["age", "systolic", "diastolic"]:
                            output[field] = int(round(value))
                        else:
                            output[field] = value
                        break

    return output
