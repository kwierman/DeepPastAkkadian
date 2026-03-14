#!/usr/bin/env python3
"""
Baseline Model: Retrieval-Based Translation

This script implements a retrieval-based baseline that finds the most similar
training example for each test sentence using TF-IDF similarity.

Usage:
    python 04_baseline_predictions.py
"""

import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt


def main():
    # Device setup (for consistency, though not used in this baseline)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\nLoading data...")
    train_df = pd.read_csv("train_augmented.csv")
    test_df = pd.read_csv("test.csv")

    train_df = train_df.dropna(subset=["transliteration", "translation"])
    train_texts = train_df["transliteration"].tolist()
    train_labels = train_df["translation"].tolist()
    test_texts = test_df["transliteration"].tolist()

    print(f"Train size: {len(train_texts)}, Test size: {len(test_texts)}")

    # Build TF-IDF vectorizer
    print("\nCreating TF-IDF vectors...")
    vectorizer = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 3),
        analyzer="char_wb",  # Character n-grams at word boundaries
        sublinear_tf=True,
    )

    # Fit on training data and transform both train and test
    train_vectors = vectorizer.fit_transform(train_texts)
    test_vectors = vectorizer.transform(test_texts)

    print(f"Train vectors shape: {train_vectors.shape}")
    print(f"Test vectors shape: {test_vectors.shape}")

    # Compute similarities
    print("\nComputing similarities...")
    similarities = cosine_similarity(test_vectors, train_vectors)

    # Find most similar example for each test sentence
    print("\n--- Similarity Analysis ---")
    for i in range(len(test_texts)):
        top_indices = np.argsort(similarities[i])[-5:][::-1]
        print(
            f"\nTest {i}: Best match similarity = {similarities[i][top_indices[0]]:.3f}"
        )
        print(f"  Test text: {test_texts[i][:80]}...")
        print(f"  Best match: {train_labels[top_indices[0]][:80]}...")

    # Generate predictions
    print("\n--- Generating Predictions ---")
    predictions = []
    for i in range(len(test_texts)):
        best_idx = np.argmax(similarities[i])
        predictions.append(train_labels[best_idx])
        print(f"Test {i}: Similarity={similarities[i][best_idx]:.3f}")
        print(f"  Prediction: {predictions[i][:100]}...")
        print()

    # Save submission
    submission = pd.DataFrame({"id": test_df["id"], "translation": predictions})

    submission.to_csv("submission_retrieval.csv", index=False)
    print("\nSubmission saved to submission_retrieval.csv")
    print("\n--- Final Submission ---")
    print(submission)

    # Similarity distribution analysis
    max_similarities = np.max(similarities, axis=1)

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(max_similarities)), max_similarities)
    plt.xlabel("Test Example")
    plt.ylabel("Maximum Similarity")
    plt.title("Best Match Similarity for Each Test Example")
    plt.xticks(
        range(len(max_similarities)),
        [f"Test {i}" for i in range(len(max_similarities))],
    )
    plt.axhline(y=0.5, color="r", linestyle="--", label="Threshold 0.5")
    plt.legend()
    plt.tight_layout()
    plt.savefig("similarity_analysis.png", dpi=150)
    plt.close()

    print(f"\nSimilarity statistics:")
    print(f"  Mean: {max_similarities.mean():.3f}")
    print(f"  Min: {max_similarities.min():.3f}")
    print(f"  Max: {max_similarities.max():.3f}")

    # Model comparison summary
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)

    print("""
1. RETRIEVAL-BASELINE (TF-IDF):
   - Method: Character n-grams + cosine similarity
   - Uses augmented dataset (~28K examples)
   - Fast to train and run
   - Works well for similar texts

2. T5-SMALL NEURAL MODEL:
   - Method: Fine-tuned transformer with GPU acceleration
   - Requires GPU and more time
   - Better generalization potential
   - Benefits from more training data

3. IMPROVEMENT SUGGESTIONS:
   - Use larger models (T5-base, mBART, NLLB)
   - Data augmentation (back-translation)
   - Ensemble methods
""")


if __name__ == "__main__":
    main()
