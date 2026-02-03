import torch
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification
)

MODEL_NAME = "MattyB95/AST-ASVspoof2019-Synthetic-Voice-Detection"

feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME)
model.eval()

LABELS = model.config.id2label


def predict(waveform: torch.Tensor):
    with torch.no_grad():
        inputs = feature_extractor(
            waveform.squeeze().numpy(),
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )

        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]

        confidence, idx = torch.max(probs, dim=0)
        label = LABELS[idx.item()]

    return label.upper(), float(confidence)
