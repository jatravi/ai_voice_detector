from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from audio_utils import decode_audio
from model import predict

API_KEY = "buildathon2026"

app = FastAPI(title="AI Generated Voice Detection API")

class AudioRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

@app.get("/")
def home():
    return {"message": "AI Voice Detector API is running"}

@app.post("/detect")
def detect_voice(
    request: AudioRequest,
    x_api_key: str = Header(None)
):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    waveform = decode_audio(request.audioBase64)
    label, confidence = predict(waveform)

    return {
        "classification": label,
        "confidence": round(confidence, 4),
        "language": request.language
    }