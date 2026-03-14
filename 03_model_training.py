#!/usr/bin/env python3
"""
Model Training: Neural Machine Translation

This script trains a T5-small transformer model for Old Assyrian to English translation.

Usage:
    python 03_model_training.py

Requirements:
    - train_augmented.csv (run 02_data_preprocessing.py first)
    - GPU recommended for training (will fallback to CPU)
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from sklearn.model_selection import train_test_split
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


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

    # Clean data
    train_df = train_df.dropna(subset=["transliteration", "translation"])
    print(f"\nAfter dropping NA: {len(train_df)}")

    # Split data
    train_texts = train_df["transliteration"].tolist()
    train_labels = train_df["translation"].tolist()

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.05, random_state=42
    )

    print(f"Training samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")

    # Dataset class
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

    # Load tokenizer and model
    print("\nLoading tokenizer and model...")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")

    # Move model to device
    model = model.to(DEVICE)

    print("Creating datasets...")
    train_dataset = TranslationDataset(
        train_texts, train_labels, tokenizer, max_len=128
    )
    val_dataset = TranslationDataset(val_texts, val_labels, tokenizer, max_len=128)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results_augmented",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=100,
        weight_decay=0.01,
        logging_steps=100,
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

    # Train
    print("\nStarting training...")
    trainer.train()
    print("Training completed!")

    # Generate predictions
    print("\nGenerating predictions...")
    test_texts = [
        "translate Akkadian to English: " + str(t)
        for t in test_df["transliteration"].tolist()
    ]

    test_inputs = tokenizer(
        test_texts, max_length=128, padding=True, truncation=True, return_tensors="pt"
    )

    # Move inputs to device
    test_inputs = {k: v.to(DEVICE) for k, v in test_inputs.items()}

    outputs = model.generate(
        input_ids=test_inputs["input_ids"],
        attention_mask=test_inputs["attention_mask"],
        max_length=128,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3,
    )

    decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    print("\nPredictions:")
    for i, pred in enumerate(decoded_preds):
        print(f"\n{i}: {pred}")

    # Save submission
    submission = pd.DataFrame({"id": test_df["id"], "translation": decoded_preds})

    submission.to_csv("submission_augmented.csv", index=False)
    print("\nSubmission saved to submission_augmented.csv")
    print("\n--- Final Submission ---")
    print(submission)


if __name__ == "__main__":
    main()
