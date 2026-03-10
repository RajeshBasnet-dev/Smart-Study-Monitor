# Smart Study Monitor - Fake News Detection (Django + NLP)

A full-stack Django web application for detecting whether a news article is **Fake** or **Real** using the Kaggle Fake and Real News dataset (`Fake.csv`, `True.csv`).

## Features

- Full-stack Django app with modular architecture.
- Baseline NLP model: **TF-IDF + Logistic Regression**.
- Advanced NLP model: **DistilBERT** fine-tuning (optional training, used automatically if artifacts are available).
- REST APIs (Django REST Framework):
  - `POST /api/predict/text/`
  - `POST /api/predict/file/`
- Frontend UI for text and file input (Bootstrap-based, responsive).
- Explainable AI output via influential token highlights.
- Prediction logging in database + simple dashboard counters.
- Deployment-ready settings for Render/Heroku/PythonAnywhere.

## Project Structure

```text
.
├── detector/
│   ├── api/
│   ├── services/
│   ├── templates/detector/
│   ├── static/css/
│   └── models.py
├── ml/
│   ├── train_models.py
│   └── artifacts/
├── news_detector/
├── Fake.csv
├── True.csv
├── manage.py
└── requirements.txt
```

## AI Pipeline

1. Load `Fake.csv` and `True.csv`, assign labels (`0` fake, `1` real).
2. Clean text (lowercase, remove punctuation/special chars/digits/stopwords).
3. Train/test split with stratification.
4. Train baseline Logistic Regression model on TF-IDF features.
5. Optionally fine-tune DistilBERT for stronger performance.
6. Save artifacts under `ml/artifacts/`.
7. During inference:
   - Use DistilBERT if present.
   - Fallback to Logistic Regression.
   - Use logistic coefficients to highlight influential tokens.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python manage.py migrate
```

## Train Models

Baseline only:

```bash
python ml/train_models.py
```

Baseline + DistilBERT:

```bash
python ml/train_models.py --train-distilbert --epochs 1
```

## Run Application

```bash
python manage.py runserver
```

Open `http://127.0.0.1:8000/`.

## API Examples

### Text Prediction

```bash
curl -X POST http://127.0.0.1:8000/api/predict/text/ \
  -H "Content-Type: application/json" \
  -d '{"text": "Your article content here"}'
```

### File Prediction

```bash
curl -X POST http://127.0.0.1:8000/api/predict/file/ \
  -F "file=@sample.txt"
```

## Evaluation Metrics

Training script prints:
- Accuracy
- F1-score
- Precision
- Recall

## Deployment Notes

- Set production environment variables from `.env.example`.
- Use `gunicorn news_detector.wsgi:application` (Procfile included).
- Run migrations on deploy.
- Collect static:

```bash
python manage.py collectstatic --noinput
```

## Dataset Reference

Kaggle: Fake and Real News Dataset  
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
