#!/usr/bin/env python3
"""
Data Preprocessing and Training Data Creation

This script creates an expanded training dataset by combining:
1. The original Deep Past Challenge training data
2. Sentence-level translations extracted from supplemental data
3. External Akkademia dataset (Gutherz et al. 2023)

The augmented dataset is saved as train_augmented.csv with ~28,000 samples.
"""

import pandas as pd
import numpy as np
import re
import os
import urllib.request
import uuid


def download_akkademia_data():
    """Download Akkademia data from GitHub if not already present."""

    def download_file(url, filename):
        """Download a file if it doesn't exist."""
        if os.path.exists(filename):
            print(f"{filename} already exists, skipping download")
            return True
        print(f"Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, filename)
            print(f"Downloaded {filename} successfully")
            return True
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            return False

    # Base URL for Akkademia data
    base_url = "https://raw.githubusercontent.com/gaigutherz/Akkademia/master/NMT_input"

    # Download training data
    download_file(f"{base_url}/train.ak", "akkademia_train.ak")
    download_file(f"{base_url}/train.en", "akkademia_train.en")

    # Download validation data
    download_file(f"{base_url}/valid.ak", "akkademia_valid.ak")
    download_file(f"{base_url}/valid.en", "akkademia_valid.en")

    # Verify downloads
    if os.path.exists("akkademia_train.ak"):
        with open("akkademia_train.ak", "r") as f:
            train_lines = len(f.readlines())
        print(f"\nAkkademia training data: {train_lines} samples")


def extract_id(name):
    """Extract publication ID from display_name."""
    if pd.isna(name):
        return None
    match = re.search(r"\(([^)]+)\)", str(name))
    if match:
        return match.group(1).strip()
    return None


def find_translit(pub_id, alias_to_translit):
    """Match sentences to transliterations."""
    if pd.isna(pub_id):
        return None
    # Try exact match
    if pub_id in alias_to_translit:
        return alias_to_translit[pub_id]
    # Try without the first word (e.g., 'Kayseri 22' -> '22')
    parts = pub_id.split()
    if len(parts) > 1:
        short_id = " ".join(parts[1:])
        if short_id in alias_to_translit:
            return alias_to_translit[short_id]
    return None


def load_akkademia_data():
    """Load Akkademia parallel corpus."""
    akkademia_data = []

    try:
        with open("akkademia_train.ak", "r", encoding="utf-8") as f:
            akk_lines = f.readlines()
        with open("akkademia_train.en", "r", encoding="utf-8") as f:
            en_lines = f.readlines()

        for akk, eng in zip(akk_lines, en_lines):
            akk = akk.strip()
            eng = eng.strip()
            if len(akk) > 10 and len(eng) > 10:
                akkademia_data.append(
                    {"transliteration": akk, "translation": eng, "source": "akkademia"}
                )

        print(f"Loaded Akkademia training data: {len(akkademia_data)} samples")
    except FileNotFoundError:
        print("Akkademia training data not found. Run with Akkademia data available.")

    try:
        with open("akkademia_valid.ak", "r", encoding="utf-8") as f:
            akk_lines = f.readlines()
        with open("akkademia_valid.en", "r", encoding="utf-8") as f:
            en_lines = f.readlines()

        for akk, eng in zip(akk_lines, en_lines):
            akk = akk.strip()
            eng = eng.strip()
            if len(akk) > 10 and len(eng) > 10:
                akkademia_data.append(
                    {
                        "transliteration": akk,
                        "translation": eng,
                        "source": "akkademia_valid",
                    }
                )

        print(
            f"Loaded Akkademia validation data: {len([x for x in akkademia_data if x['source'] == 'akkademia_valid'])} samples"
        )
    except FileNotFoundError:
        print("Akkademia validation data not found.")

    return akkademia_data


def main():
    print("=" * 60)
    print("DATA PREPROCESSING")
    print("=" * 60)

    # Download Akkademia data
    print("\n--- Downloading Akkademia Data ---")
    download_akkademia_data()

    # Load datasets
    print("\n--- Loading Datasets ---")
    train = pd.read_csv("train.csv")
    sentences = pd.read_csv("Sentences_Oare_FirstWord_LinNum.csv")
    published = pd.read_csv("published_texts.csv")

    print(f"Original train size: {len(train)}")
    print(f"Sentences size: {len(sentences)}")
    print(f"Published texts size: {len(published)}")

    # Extract publication IDs from sentences
    sentences["pub_id"] = sentences["display_name"].apply(extract_id)
    print(f"\nExtracted publication IDs: {sentences['pub_id'].notna().sum()}")

    # Create alias to transliteration mapping
    alias_to_translit = {}
    for idx, row in published.iterrows():
        if pd.notna(row["aliases"]):
            for alias in row["aliases"].split("|"):
                alias_to_translit[alias.strip()] = row["transliteration"]

    print(f"Created mapping for {len(alias_to_translit)} aliases")

    # Match sentences to transliterations
    sentences["transliteration"] = sentences["pub_id"].apply(
        lambda x: find_translit(x, alias_to_translit)
    )

    print(f"Matched transliterations: {sentences['transliteration'].notna().sum()}")
    print(f"Unmatched: {sentences['transliteration'].isna().sum()}")

    # Filter to matched sentences
    matched_sentences = sentences[
        (sentences["translation"].notna())
        & (sentences["transliteration"].notna())
        & (sentences["translation"] != "")
        & (sentences["transliteration"] != "")
    ].copy()

    print(
        f"\nMatched sentences with both translation and transliteration: {len(matched_sentences)}"
    )

    # Remove duplicates with original training data
    existing_pairs = set(zip(train["transliteration"], train["translation"]))

    new_sentences = []
    for idx, row in matched_sentences.iterrows():
        pair = (row["transliteration"], row["translation"])
        if pair not in existing_pairs:
            new_sentences.append(row)

    print(f"New sentences not in original training data: {len(new_sentences)}")

    # Load Akkademia data
    print("\n--- Loading Akkademia Data ---")
    akkademia_data = load_akkademia_data()

    # Create combined training dataset
    print("\n--- Creating Augmented Dataset ---")

    # Original training data
    combined_train = train[["transliteration", "translation"]].copy()
    combined_train["source"] = "original"

    # Add new sentences from supplemental data
    new_train_df = pd.DataFrame(new_sentences)
    if len(new_train_df) > 0:
        new_train_df = new_train_df[["transliteration", "translation"]].copy()
        new_train_df["source"] = "sentences"
        combined_train = pd.concat([combined_train, new_train_df], ignore_index=True)

    # Add Akkademia data
    if akkademia_data:
        akkademia_df = pd.DataFrame(akkademia_data)
        combined_train = pd.concat([combined_train, akkademia_df], ignore_index=True)

    # Remove duplicates
    before_dedup = len(combined_train)
    combined_train = combined_train.drop_duplicates(
        subset=["transliteration", "translation"]
    )
    after_dedup = len(combined_train)

    print(f"Before deduplication: {before_dedup}")
    print(f"After deduplication: {after_dedup}")
    print(f"Removed {before_dedup - after_dedup} duplicates")

    # Add UUID for each entry
    combined_train["oare_id"] = [str(uuid.uuid4()) for _ in range(len(combined_train))]

    # Reorder columns
    combined_train = combined_train[
        ["oare_id", "transliteration", "translation", "source"]
    ]

    # Save augmented training data
    combined_train.to_csv("train_augmented.csv", index=False)
    print("\nSaved augmented training data to train_augmented.csv")

    print(f"\n--- Final Dataset Summary ---")
    print(f"Total samples: {len(combined_train)}")
    print(f"\nSource distribution:")
    print(combined_train["source"].value_counts())
    print(f"\nData increase from original: {len(combined_train) / len(train):.1f}x")


if __name__ == "__main__":
    main()
