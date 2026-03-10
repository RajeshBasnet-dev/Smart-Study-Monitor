"""Model loading and inference services."""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from django.conf import settings

from .preprocessing import clean_text

try:
    from transformers import pipeline
except Exception:  # pragma: no cover
    pipeline = None


@dataclass
class PredictionResult:
    label: str
    confidence: float
    model_used: str
    highlights: list[dict[str, Any]]


class FakeNewsPredictor:
    """Hybrid inference using DistilBERT (if available) + Logistic Regression explanation."""

    def __init__(self) -> None:
        artifact_dir = Path(settings.MODEL_ARTIFACT_DIR)
        self.vectorizer = joblib.load(artifact_dir / 'tfidf_vectorizer.joblib')
        self.log_reg = joblib.load(artifact_dir / 'logistic_regression.joblib')
        self.bert_classifier = None
        distilbert_path = artifact_dir / 'distilbert_model'
        if pipeline and distilbert_path.exists():
            self.bert_classifier = pipeline('text-classification', model=str(distilbert_path), tokenizer=str(distilbert_path))

    def predict(self, text: str) -> PredictionResult:
        cleaned = clean_text(text)
        if not cleaned.strip():
            raise ValueError('Input text is empty after cleaning. Provide more meaningful content.')

        lr_probs = self.log_reg.predict_proba(self.vectorizer.transform([cleaned]))[0]
        lr_label = 'Real' if int(np.argmax(lr_probs)) == 1 else 'Fake'
        lr_conf = float(np.max(lr_probs))

        label = lr_label
        confidence = lr_conf
        model_used = 'logistic_regression'

        if self.bert_classifier:
            bert_result = self.bert_classifier(text[:4000])[0]
            bert_label = 'Real' if bert_result['label'].upper() in {'LABEL_1', 'REAL'} else 'Fake'
            label = bert_label
            confidence = float(bert_result['score'])
            model_used = 'distilbert'

        highlights = self._explain_with_lr(cleaned, label)
        return PredictionResult(label=label, confidence=confidence, model_used=model_used, highlights=highlights)

    def predict_from_file(self, uploaded_file) -> PredictionResult:
        content = uploaded_file.read()
        if isinstance(content, bytes):
            text = content.decode('utf-8', errors='ignore')
        else:
            text = io.TextIOWrapper(uploaded_file, encoding='utf-8').read()
        return self.predict(text)

    def _explain_with_lr(self, cleaned_text: str, predicted_label: str) -> list[dict[str, Any]]:
        tokens = cleaned_text.split()
        token_set = sorted(set(tokens))
        if not token_set:
            return []

        coefficients = self.log_reg.coef_[0]
        features = self.vectorizer.vocabulary_
        scores = []
        for token in token_set:
            idx = features.get(token)
            if idx is None:
                continue
            score = float(coefficients[idx])
            if predicted_label == 'Fake':
                score *= -1
            scores.append({'token': token, 'score': abs(score), 'direction': 'support' if score > 0 else 'oppose'})

        scores.sort(key=lambda item: item['score'], reverse=True)
        return scores[:12]


_predictor: FakeNewsPredictor | None = None


def get_predictor() -> FakeNewsPredictor:
    global _predictor
    if _predictor is None:
        _predictor = FakeNewsPredictor()
    return _predictor
