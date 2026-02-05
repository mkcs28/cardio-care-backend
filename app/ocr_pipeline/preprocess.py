import re

def normalize(text):
    return re.sub(r"[^a-z0-9./]", " ", text.lower())
