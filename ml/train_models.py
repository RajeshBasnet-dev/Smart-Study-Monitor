"""Train baseline and DistilBERT fake-news models."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from detector.services.preprocessing import clean_text


def load_dataset(fake_path: Path, true_path: Path) -> pd.DataFrame:
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)
    fake_df['label'] = 0
    true_df['label'] = 1
    df = pd.concat([fake_df, true_df], ignore_index=True)
    df = df[['title', 'text', 'label']].fillna('')
    df['full_text'] = (df['title'] + ' ' + df['text']).apply(clean_text)
    return df[df['full_text'].str.len() > 0]


def evaluate(y_true, y_pred, model_name: str) -> dict:
    return {
        'model': model_name,
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
    }


def train_baseline(df: pd.DataFrame, artifact_dir: Path) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        df['full_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )

    vectorizer = TfidfVectorizer(max_features=30000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    classifier = LogisticRegression(max_iter=300, n_jobs=-1)
    classifier.fit(X_train_vec, y_train)
    y_pred = classifier.predict(X_test_vec)

    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, artifact_dir / 'tfidf_vectorizer.joblib')
    joblib.dump(classifier, artifact_dir / 'logistic_regression.joblib')

    return evaluate(y_test, y_pred, 'logistic_regression')


def train_distilbert(df: pd.DataFrame, artifact_dir: Path, epochs: int = 1) -> dict:
    """Fine-tune DistilBERT for stronger accuracy (optional due to runtime requirements)."""
    from datasets import Dataset
    from transformers import (
        DistilBertForSequenceClassification,
        DistilBertTokenizerFast,
        Trainer,
        TrainingArguments,
    )

    train_df, test_df = train_test_split(df[['full_text', 'label']], test_size=0.2, random_state=42, stratify=df['label'])
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    def tokenize(batch):
        return tokenizer(batch['full_text'], truncation=True, padding='max_length', max_length=256)

    train_ds = Dataset.from_pandas(train_df).map(tokenize, batched=True)
    test_ds = Dataset.from_pandas(test_df).map(tokenize, batched=True)
    train_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    test_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

    args = TrainingArguments(
        output_dir=str(artifact_dir / 'distilbert_checkpoints'),
        eval_strategy='epoch',
        save_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_steps=50,
        load_best_model_at_end=True,
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=test_ds)
    trainer.train()

    predictions = trainer.predict(test_ds)
    y_pred = predictions.predictions.argmax(axis=-1)
    y_true = test_df['label'].to_numpy()

    export_path = artifact_dir / 'distilbert_model'
    model.save_pretrained(export_path)
    tokenizer.save_pretrained(export_path)

    return evaluate(y_true, y_pred, 'distilbert')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--fake-csv', default='Fake.csv')
    parser.add_argument('--true-csv', default='True.csv')
    parser.add_argument('--artifact-dir', default='ml/artifacts')
    parser.add_argument('--train-distilbert', action='store_true')
    parser.add_argument('--epochs', type=int, default=1)
    args = parser.parse_args()

    df = load_dataset(Path(args.fake_csv), Path(args.true_csv))
    artifact_dir = Path(args.artifact_dir)

    baseline_metrics = train_baseline(df, artifact_dir)
    print('Baseline Metrics:', baseline_metrics)

    if args.train_distilbert:
        bert_metrics = train_distilbert(df, artifact_dir, epochs=args.epochs)
        print('DistilBERT Metrics:', bert_metrics)


if __name__ == '__main__':
    main()
