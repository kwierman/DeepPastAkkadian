#!/usr/bin/env python3
"""
Evaluation and Results Analysis

This script analyzes the model predictions and compares them to the expected outputs.

Evaluation Metric: Geometric mean of BLEU and chrF++ scores

Usage:
    python 05_evaluation.py
"""

import pandas as pd
import torch
import sacrebleu
import numpy as np


def main():
    # Device setup (for consistency)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    test_df = pd.read_csv("test.csv")
    submission = pd.read_csv("submission_augmented.csv")
    sample_submission = pd.read_csv("sample_submission.csv")

    print("\nTest data:")
    print(test_df)

    print("\nOur predictions:")
    print(submission)

    print("\nSample submission (reference):")
    print(sample_submission)

    # Calculate BLEU scores
    print("\n" + "=" * 70)
    print("BLEU SCORE CALCULATION")
    print("=" * 70)

    references = sample_submission["translation"].tolist()
    predictions = submission["translation"].tolist()

    bleu_scores = []
    for ref, pred in zip(references, predictions):
        bleu = sacrebleu.sentence_bleu(pred, [ref])
        bleu_scores.append(bleu.score)
        print(f"BLEU: {bleu.score:.2f} | Ref: {ref[:50]}...")
        print(f"         Pred: {pred[:50]}...")
        print()

    print(f"\nMean BLEU: {np.mean(bleu_scores):.2f}")

    # Calculate chrF++ scores
    print("\n" + "=" * 70)
    print("chrF++ SCORE CALCULATION")
    print("=" * 70)

    chrf_scores = []
    for ref, pred in zip(references, predictions):
        chrf = sacrebleu.sentence_chrf(pred, [ref])
        chrf_scores.append(chrf.score)
        print(f"chrF++: {chrf.score:.2f}")

    print(f"\nMean chrF++: {np.mean(chrf_scores):.2f}")

    # Geometric mean
    print("\n" + "=" * 70)
    print("GEOMETRIC MEAN OF BLEU AND chrF++")
    print("=" * 70)

    geometric_means = []
    for bleu, chrf in zip(bleu_scores, chrf_scores):
        gm = np.sqrt((bleu + 0.01) * (chrf + 0.01))
        geometric_means.append(gm)
        print(f"BLEU: {bleu:.2f}, chrF++: {chrf:.2f}, Geometric Mean: {gm:.2f}")

    print(f"\nOverall Geometric Mean: {np.mean(geometric_means):.2f}")

    # Detailed prediction analysis
    print("\n" + "=" * 70)
    print("DETAILED PREDICTION ANALYSIS")
    print("=" * 70)

    for i in range(len(test_df)):
        print(f"\n--- Test {i} ---")
        print(f"Input (transliteration):")
        print(f"  {test_df['transliteration'].iloc[i][:100]}...")
        print(f"\nReference translation:")
        print(f"  {references[i][:150]}...")
        print(f"\nOur prediction:")
        print(f"  {predictions[i][:150]}...")
        print(
            f"\nScores: BLEU={bleu_scores[i]:.2f}, chrF++={chrf_scores[i]:.2f}, GM={geometric_means[i]:.2f}"
        )

    # Key observations
    print("\n" + "=" * 70)
    print("KEY OBSERVATIONS")
    print("=" * 70)

    print("""
1. AUGMENTED DATASET:
   - Training data expanded from ~1,500 to ~28,000 samples
   - Combined original Deep Past Challenge + Akkademia data
   - External resources from ORACC RINAP projects

2. MODEL APPROACHES:
   - Retrieval baseline: TF-IDF + cosine similarity
   - Neural model: T5-small with GPU acceleration
   - Both benefit from augmented dataset

3. DATA CHARACTERISTICS:
   - Test set contains sentences from text_id: 332fda50
   - Training contains full document translations
   - Sentence-level alignment is crucial

4. GPU ACCELERATION:
   - Scripts detect CUDA availability automatically
   - Falls back to CPU if no GPU available
   - Uses fp16 when GPU is available for faster training
""")

    # Conclusion
    print("\n" + "=" * 70)
    print("PROJECT SUMMARY")
    print("=" * 70)

    print("""
ACHIEVEMENTS:
- Created augmented training dataset (~28,000 examples from ~1,500)
- Integrated external Akkademia dataset (Gutherz et al. 2023)
- Implemented both neural (T5) and retrieval-based approaches
- Added GPU acceleration support with automatic detection
- Generated valid submission file

CHALLENGES:
- Low-resource language (Akkadian)
- Complex morphology
- Limited sentence-level aligned data
- GPU time constraints

FUTURE WORK:
- Use larger transformer models (mBART, NLLB)
- Extract more training data from scholarly publications
- Implement data augmentation
- Build ensemble of retrieval + neural models
""")

    print("\nFiles generated:")
    print("  - train_augmented.csv: Augmented training dataset (~28K samples)")
    print("  - akkademia_train.ak/en: External Akkademia parallel corpus")
    print("  - submission_augmented.csv: Neural model predictions")
    print("  - submission_retrieval.csv: Retrieval baseline predictions")
    print("  - similarity_analysis.png: Similarity distribution plot")


if __name__ == "__main__":
    main()
