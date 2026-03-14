#!/usr/bin/env python3
"""
Deep Past Challenge: Old Assyrian to English Translation
Exploratory Data Analysis

This notebook provides an exploratory data analysis of the Deep Past Challenge dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)


def main():
    print("=" * 60)
    print("DEEP PAST CHALLENGE: OLD ASSYRIAN TO ENGLISH TRANSLATION")
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    # Load the augmented training and test datasets
    train_df = pd.read_csv("train_augmented.csv")
    test_df = pd.read_csv("test.csv")

    print(f"\nTraining data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")

    print("\n--- Training Data Columns ---")
    print(train_df.columns.tolist())

    print("\n--- Test Data Columns ---")
    print(test_df.columns.tolist())

    print("\n--- Data Source Breakdown ---")
    if "source" in train_df.columns:
        print(train_df["source"].value_counts())

    # Examine training data
    print("\n" + "=" * 60)
    print("EXAMINING TRAINING DATA")
    print("=" * 60)

    print("\nMissing values in training data:")
    print(train_df.isnull().sum())

    print("\n--- Sample Transliteration (from different sources) ---")
    for src in train_df["source"].unique()[:3]:
        sample = train_df[train_df["source"] == src].iloc[0]
        print(f"\n[{src}]:")
        print(f"  {sample['transliteration'][:200]}...")

    print("\n--- Sample Translation ---")
    print(train_df["translation"].iloc[0][:500])

    # Text length analysis
    print("\n" + "=" * 60)
    print("TEXT LENGTH ANALYSIS")
    print("=" * 60)

    train_df["transliteration_len"] = train_df["transliteration"].str.len()
    train_df["translation_len"] = train_df["translation"].str.len()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(train_df["transliteration_len"], bins=50, edgecolor="black", alpha=0.7)
    axes[0].set_xlabel("Character Length")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Distribution of Transliteration Lengths")
    axes[0].axvline(
        train_df["transliteration_len"].mean(),
        color="red",
        linestyle="--",
        label=f"Mean: {train_df['transliteration_len'].mean():.0f}",
    )
    axes[0].legend()

    axes[1].hist(
        train_df["translation_len"],
        bins=50,
        edgecolor="black",
        alpha=0.7,
        color="green",
    )
    axes[1].set_xlabel("Character Length")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Distribution of Translation Lengths")
    axes[1].axvline(
        train_df["translation_len"].mean(),
        color="red",
        linestyle="--",
        label=f"Mean: {train_df['translation_len'].mean():.0f}",
    )
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("text_length_distributions.png", dpi=150)
    plt.close()

    print("\n--- Text Length Statistics ---")
    print(train_df[["transliteration_len", "translation_len"]].describe())

    # Length ratio analysis
    train_df["length_ratio"] = (
        train_df["translation_len"] / train_df["transliteration_len"]
    )

    plt.figure(figsize=(10, 5))
    plt.hist(
        train_df["length_ratio"], bins=50, edgecolor="black", alpha=0.7, color="purple"
    )
    plt.xlabel("Length Ratio (Translation / Transliteration)")
    plt.ylabel("Frequency")
    plt.title("Translation to Transliteration Length Ratio")
    plt.axvline(
        train_df["length_ratio"].mean(),
        color="red",
        linestyle="--",
        label=f"Mean: {train_df['length_ratio'].mean():.2f}",
    )
    plt.legend()
    plt.savefig("length_ratio_distribution.png", dpi=150)
    plt.close()

    print(f"\nMean length ratio: {train_df['length_ratio'].mean():.2f}")
    print(
        f"This indicates translations are on average {train_df['length_ratio'].mean():.1f}x longer than transliterations"
    )

    # Compare sources
    print("\n" + "=" * 60)
    print("COMPARING SOURCES")
    print("=" * 60)

    print("\n--- Text Statistics by Source ---")
    source_stats = (
        train_df.groupby("source")
        .agg(
            {
                "transliteration_len": ["mean", "std", "min", "max"],
                "translation_len": ["mean", "std", "min", "max"],
                "length_ratio": ["mean", "std"],
            }
        )
        .round(2)
    )
    print(source_stats)

    # Examine test data
    print("\n" + "=" * 60)
    print("TEST DATA")
    print("=" * 60)

    print("\nTest data:")
    print(test_df)

    print("\n--- Sample Test Transliterations ---")
    for i, row in test_df.iterrows():
        print(f"\nID {row['id']}:")
        print(f"  Lines: {row['line_start']}-{row['line_end']}")
        print(f"  Text: {row['transliteration'][:100]}...")

    # Load supplemental data
    print("\n" + "=" * 60)
    print("SUPPLEMENTAL DATA")
    print("=" * 60)

    published_texts = pd.read_csv("published_texts.csv")
    sentences = pd.read_csv("Sentences_Oare_FirstWord_LinNum.csv")

    print(f"\nPublished texts: {len(published_texts)} entries")
    print(f"Sentences with translations: {len(sentences)} entries")

    print("\n--- Published Texts Columns ---")
    print(published_texts.columns.tolist())

    print("\n--- Sentences Columns ---")
    print(sentences.columns.tolist())

    # Key findings summary
    print("\n" + "=" * 60)
    print("KEY FINDINGS FROM EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    print(f"""
1. TRAINING DATA (AUGMENTED):
   - Total samples: {len(train_df)}
   - Average transliteration length: {train_df["transliteration_len"].mean():.0f} characters
   - Average translation length: {train_df["translation_len"].mean():.0f} characters
   - Mean length ratio: {train_df["length_ratio"].mean():.2f}x

2. DATA SOURCES:
   - Original Deep Past Challenge: {(train_df["source"] == "original").sum()} samples
   - Akkademia (Gutherz et al. 2023): {(train_df["source"] == "akkademia").sum()} samples
   - Akkademia Validation: {(train_df["source"] == "akkademia_valid").sum()} samples

3. TEST DATA:
   - Test samples: {len(test_df)}
   - Note: These are sample sentences from one document (text_id: {test_df["text_id"].iloc[0]})

4. SUPPLEMENTAL DATA:
   - Published texts (without translations): {len(published_texts)}
   - Sentences with translations: {len(sentences)}

5. CHALLENGES:
   - Originally low-resource (~1,500 examples), now augmented to ~28,000
   - Morphologically complex language
   - Translations vary significantly in length (high std deviation)
""")


if __name__ == "__main__":
    main()
