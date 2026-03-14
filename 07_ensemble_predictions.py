#!/usr/bin/env python3
"""
Ensemble Predictions

This script generates ensemble predictions by combining:
1. Neural model (T5) predictions
2. Retrieval-based (TF-IDF) predictions

Usage:
    python 07_ensemble_predictions.py
"""

import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import Counter


def main():
    # Device setup
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    test_df = pd.read_csv("test.csv")
    train_df = pd.read_csv("train_augmented.csv")

    print(f"Test size: {len(test_df)}")
    print(f"Train size: {len(train_df)}")

    # Load neural model
    print("\nLoading neural model...")
    try:
        tokenizer = T5Tokenizer.from_pretrained("./ensemble_models/t5_small")
        model = T5ForConditionalGeneration.from_pretrained("./ensemble_models/t5_small")
        print("Loaded ensemble T5 model.")
    except:
        print("Ensemble model not found, using base T5-small...")
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        model = T5ForConditionalGeneration.from_pretrained("t5-small")

    model = model.to(DEVICE)
    print(f"Model loaded on: {next(model.parameters()).device}")

    # Generate neural predictions
    print("\nGenerating neural predictions...")
    test_texts = [
        "translate Akkadian to English: " + str(t)
        for t in test_df["transliteration"].tolist()
    ]

    test_inputs = tokenizer(
        test_texts, max_length=128, padding=True, truncation=True, return_tensors="pt"
    )

    test_inputs = {k: v.to(DEVICE) for k, v in test_inputs.items()}

    outputs = model.generate(
        input_ids=test_inputs["input_ids"],
        attention_mask=test_inputs["attention_mask"],
        max_length=128,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3,
    )

    neural_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    print("Neural predictions:")
    for i, pred in enumerate(neural_preds):
        print(f"  {i}: {pred[:80]}...")

    # Generate retrieval predictions
    print("\nGenerating retrieval predictions...")

    train_df = train_df.dropna(subset=["transliteration", "translation"])
    train_texts = train_df["transliteration"].tolist()
    train_labels = train_df["translation"].tolist()
    test_texts_raw = test_df["transliteration"].tolist()

    # TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=15000, ngram_range=(1, 3), analyzer="char_wb", sublinear_tf=True
    )

    train_vectors = vectorizer.fit_transform(train_texts)
    test_vectors = vectorizer.transform(test_texts_raw)

    similarities = cosine_similarity(test_vectors, train_vectors)

    # Get retrieval predictions
    retrieval_preds = []
    for i in range(len(test_texts_raw)):
        best_idx = np.argmax(similarities[i])
        retrieval_preds.append(train_labels[best_idx])

    print("Retrieval predictions:")
    for i, pred in enumerate(retrieval_preds):
        best_idx = np.argmax(similarities[i])
        print(f"  {i}: {pred[:80]}...")
        print(f"     Similarity: {similarities[i][best_idx]:.3f}")

    # Create ensemble predictions
    print("\n--- Creating Ensemble ---")

    ensemble_preds = []
    for i in range(len(test_texts_raw)):
        max_sim = similarities[i][np.argmax(similarities[i])]

        # If retrieval similarity is high, prefer retrieval
        if max_sim > 0.6:
            ensemble_preds.append(retrieval_preds[i])
        else:
            # Otherwise use neural prediction
            ensemble_preds.append(neural_preds[i])

    print("Ensemble predictions:")
    for i, pred in enumerate(ensemble_preds):
        max_sim = similarities[i][np.argmax(similarities[i])]
        source = "retrieval" if max_sim > 0.6 else "neural"
        print(f"  {i} [{source}]: {pred[:80]}...")

    # Save submission
    submission = pd.DataFrame({"id": test_df["id"], "translation": ensemble_preds})

    submission.to_csv("submission_ensemble.csv", index=False)
    print("\nEnsemble submission saved to submission_ensemble.csv")

    print("\n--- Final Submission ---")
    print(submission)

    # Save all predictions for analysis
    all_preds = pd.DataFrame(
        {
            "id": test_df["id"],
            "transliteration": test_df["transliteration"],
            "neural_pred": neural_preds,
            "retrieval_pred": retrieval_preds,
            "ensemble_pred": ensemble_preds,
            "retrieval_similarity": [
                similarities[i][np.argmax(similarities[i])] for i in range(len(test_df))
            ],
        }
    )

    all_preds.to_csv("all_predictions.csv", index=False)
    print("\nAll predictions saved to all_predictions.csv")

    # Analysis
    print("\n--- Ensemble Analysis ---")

    sources = []
    for i in range(len(test_texts_raw)):
        max_sim = similarities[i][np.argmax(similarities[i])]
        sources.append("retrieval" if max_sim > 0.6 else "neural")

    source_counts = Counter(sources)
    print(f"Predictions from neural model: {source_counts.get('neural', 0)}")
    print(f"Predictions from retrieval: {source_counts.get('retrieval', 0)}")

    print("\n--- Prediction Comparison ---")
    for i in range(len(test_df)):
        print(f"\nTest {i}:")
        print(f"  Neural:     {neural_preds[i][:60]}...")
        print(f"  Retrieval:  {retrieval_preds[i][:60]}...")
        print(f"  Ensemble:   {ensemble_preds[i][:60]}...")


if __name__ == "__main__":
    main()
