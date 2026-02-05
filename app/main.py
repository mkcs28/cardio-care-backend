import numpy as np
import pickle
import torch
import torch.nn as nn
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path

# ✅ ADDITION (REQUIRED FOR OCR ROUTES)
from app.routes.ocr_routes import router as ocr_router

# =========================================================
# PATH SETUP
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# =========================================================
# LOAD PREPROCESSING
# =========================================================
with open(MODEL_DIR / "bnn_19f_GPU2.pkl", "rb") as f:
    preproc = pickle.load(f)

scaler = preproc["scaler"]
le = preproc["label_encoder"]

# =========================================================
# MODEL DEFINITIONS
# =========================================================
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return torch.matmul(x, self.weight_mu.t()) + self.bias_mu


class CardioBNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = BayesianLinear(19, 64)
        self.fc2 = BayesianLinear(64, 32)
        self.fc3 = BayesianLinear(32, 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


# =========================================================
# LOAD MODEL
# =========================================================
model = CardioBNN()
model.load_state_dict(
    torch.load(MODEL_DIR / "bnn_19f_GPU2.pth", map_location="cpu"),
    strict=False
)
model.eval()

# =========================================================
# MAPPING FUNCTIONS (SAME AS TRAINING)
# =========================================================
def map_gender(g): 
    return 1 if str(g).lower() == "male" else 0

def map_age(a):
    if 20 <= a <= 39: return 10
    elif 40 < a <= 59: return 20
    elif 60 < a <= 80: return 30
    return 0

def map_bmi(b):
    if 18.5 <= b <= 22.9: return 5
    elif 23 <= b <= 24.9: return 10
    elif 25 <= b <= 29.9: return 15
    elif b >= 30: return 20
    return 0

def map_whr(w): 
    return 1 if w < 0.9 else 10

def map_fbs(f): 
    return 1 if f < 110 else 5 if f < 126 else 10

def map_hba1c(h): 
    return 1 if h < 5.6 else 10 if h < 6.5 else 30 if h < 8 else 50

def map_hdl(h, g):
    if g == 1:
        return 1 if h > 55 else 10 if h >= 35 else 20
    return 1 if h > 65 else 10 if h >= 45 else 20

def map_ldl(l): 
    return 1 if l < 100 else 20 if l <= 160 else 40

def map_vldl(v): 
    return 1 if v <= 30 else 2 if v <= 40 else 3

def map_tgl(t): 
    return 1 if t < 150 else 20 if t <= 250 else 30

def map_tc(tc): 
    return 1 if tc < 200 else 10 if tc <= 300 else 20

def map_creatinine(c): 
    return 1 if c <= 1.2 else 3 if c <= 2 else 6

def map_tyg(t):
    return 1 if t < 8 else 10 if t <= 8.7 else 20

def map_hdl_ldl(r):
    return 1 if r < 2 else 5 if r < 3.5 else 10

def map_bp_numbers(s, d):
    if s < 120 and d < 80: return 1
    elif s < 140 or d < 90: return 5
    elif s < 160 or d < 100: return 10
    return 20

def map_alcohol(a): 
    return 5 if a == "Yes" else 1

def map_alcohol_freq(f): 
    return {
        "Daily": 20,
        "Several times a week": 15,
        "Once a week": 10,
        "Occasionally": 5,
        "Past": 3
    }.get(f, 1)

def map_smoking(s): 
    return 10 if s == "Yes" else 1

def map_smoking_freq(f): 
    return {
        "Daily": 40,
        "Several times a week": 30,
        "Once a week": 20,
        "Occasionally": 10,
        "Past": 6
    }.get(f, 2)

# =========================================================
# FASTAPI APP
# =========================================================
app = FastAPI(title="Cardio Care API")

# ✅ ADDITION (REGISTER OCR ROUTES)
app.include_router(ocr_router)

@app.get("/")
def health():
    return {"status":"ok"}
# =========================================================
# INPUT SCHEMA
# =========================================================
class PredictInput(BaseModel):
    gender: str
    age: int
    bmi: float
    whr: float
    fbs: float
    hba1c: float
    hdl: float
    ldl: float
    vldl: float
    tgl: float
    tc: float
    creatinine: float
    systolic: int
    diastolic: int
    alcohol: str
    alcohol_freq: str
    smoking: str
    smoking_freq: str

# =========================================================
# PREDICT ENDPOINT
# =========================================================
@app.post("/predict")
def predict(data: PredictInput):

    tyg = np.log((data.tgl * data.fbs) / 2)
    ldl_hdl = data.ldl / data.hdl if data.hdl != 0 else 0
    gender_code = map_gender(data.gender)

    ml_features = [
        gender_code,
        map_age(data.age),
        map_bmi(data.bmi),
        map_whr(data.whr),
        map_fbs(data.fbs),
        map_hba1c(data.hba1c),
        map_hdl(data.hdl, gender_code),
        map_ldl(data.ldl),
        map_vldl(data.vldl),
        map_tgl(data.tgl),
        map_tc(data.tc),
        map_creatinine(data.creatinine),
        map_tyg(tyg),
        map_hdl_ldl(ldl_hdl),
        map_bp_numbers(data.systolic, data.diastolic),
        map_alcohol(data.alcohol),
        map_alcohol_freq(data.alcohol_freq),
        map_smoking(data.smoking),
        map_smoking_freq(data.smoking_freq),
    ]

    assert len(ml_features) == 19, "Feature mismatch with training"

    X = scaler.transform([ml_features])
    X = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        probs = torch.softmax(model(X), dim=1).numpy()[0]

    pred_idx = int(np.argmax(probs))
    pred_label = le.inverse_transform([pred_idx])[0]

    return {
        "prediction": pred_label,
        "confidence": float(probs[pred_idx]),
        "probabilities": {
            le.classes_[i]: float(probs[i]) for i in range(len(probs))
        }
    }
