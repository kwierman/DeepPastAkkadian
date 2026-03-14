#!/usr/bin/env python3
"""
Ensemble Model Training

This script trains multiple models for ensemble prediction:
1. T5-small: Transformer-based neural machine translation
2. mT5-small: Multilingual T5 (better for low-resource languages)

Usage:
    python 06_ensemble_training.py
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    MT5Tokenizer,
    MT5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from sklearn.model_selection import train_test_split
import os
import json

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TranslationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = "translate Akkadian to English: " + str(self.texts[idx])
        label = str(self.labels[idx])

        source = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        target = self.tokenizer(
            label,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": source["input_ids"].squeeze(),
            "attention_mask": source["attention_mask"].squeeze(),
            "labels": target["input_ids"].squeeze(),
        }


def train_model(model_name, tokenizer, model, train_dataset, val_dataset, output_dir):
    """Train a single model."""
    print(f"\nTraining {model_name}...")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=50,
        weight_decay=0.01,
        logging_steps=200,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        report_to="none",
        learning_rate=3e-4,
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    print(f"{model_name} training completed!")

    # Save model
    model.save_pretrained(f"./ensemble_models/{model_name}")
    tokenizer.save_pretrained(f"./ensemble_models/{model_name}")
    print(f"{model_name} model saved.")

    return True


def main():
    # Device setup
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )

    print("\nLoading augmented data...")
    train_df = pd.read_csv("train_augmented.csv")
    test_df = pd.read_csv("test.csv")

    print(f"Augmented train size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")
    print("\n--- Data Sources ---")
    print(train_df["source"].value_counts())

    # Clean and split data
    train_df = train_df.dropna(subset=["transliteration", "translation"])
    train_texts = train_df["transliteration"].tolist()
    train_labels = train_df["translation"].tolist()

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.05, random_state=42
    )

    print(f"\nTraining samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")

    # Create output directories
    os.makedirs("./ensemble_results", exist_ok=True)
    os.makedirs("./ensemble_models", exist_ok=True)

    # Track trained models
    trained_models = []

    # Train Model 1: T5-small
    print("\n" + "=" * 60)
    print("TRAINING MODEL 1: T5-small")
    print("=" * 60)

    tokenizer_t5 = T5Tokenizer.from_pretrained("t5-small")
    model_t5 = T5ForConditionalGeneration.from_pretrained("t5-small")
    model_t5 = model_t5.to(DEVICE)

    train_dataset_t5 = TranslationDataset(
        train_texts, train_labels, tokenizer_t5, max_len=128
    )
    val_dataset_t5 = TranslationDataset(
        val_texts, val_labels, tokenizer_t5, max_len=128
    )

    train_model(
        "t5_small",
        tokenizer_t5,
        model_t5,
        train_dataset_t5,
        val_dataset_t5,
        "./ensemble_results/t5_small",
    )
    trained_models.append("t5_small")

    # Train Model 2: mT5-small (optional)
    print("\n" + "=" * 60)
    print("TRAINING MODEL 2: mT5-small (optional)")
    print("=" * 60)

    try:
        tokenizer_mt5 = MT5Tokenizer.from_pretrained("google/mt5-small")
        model_mt5 = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
        model_mt5 = model_mt5.to(DEVICE)

        train_dataset_mt5 = TranslationDataset(
            train_texts, train_labels, tokenizer_mt5, max_len=128
        )
        val_dataset_mt5 = TranslationDataset(
            val_texts, val_labels, tokenizer_mt5, max_len=128
        )

        train_model(
            "mt5_small",
            tokenizer_mt5,
            model_mt5,
            train_dataset_mt5,
            val_dataset_mt5,
            "./ensemble_results/mt5_small",
        )
        trained_models.append("mt5_small")
    except Exception as e:
        print(f"mT5 training skipped: {e}")

    # Save ensemble configuration
    ensemble_config = {
        "models": trained_models,
        "device": str(DEVICE),
        "training_samples": len(train_texts),
        "val_samples": len(val_texts),
    }

    with open("ensemble_config.json", "w") as f:
        json.dump(ensemble_config, f, indent=2)

    print("\n" + "=" * 60)
    print("ENSEMBLE TRAINING COMPLETE")
    print("=" * 60)
    print(f"Trained models: {trained_models}")
    print("Run 07_ensemble_predictions.py to generate ensemble predictions.")


if __name__ == "__main__":
    main()
