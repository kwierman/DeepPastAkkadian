#!/usr/bin/env python3
"""
Load Model and Create Submission

This script loads pre-trained model weights and generates predictions for the test set.

Usage:
    python 08_load_model_submission.py

Options:
    --model MODEL_PATH    Path to model directory or HuggingFace model name
    --output OUTPUT_FILE  Output CSV file name (default: submission_final.csv)
"""

import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os
import argparse
import glob


def load_model(model_path=None):
    """Load model from disk or HuggingFace hub."""

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Determine model path
    if model_path is None:
        # Try different sources in priority order
        if os.path.exists("./ensemble_models/t5_small"):
            model_path = "./ensemble_models/t5_small"
        elif os.path.exists("./results_augmented"):
            checkpoints = glob.glob("./results_augmented/checkpoint-*")
            if checkpoints:
                model_path = sorted(checkpoints)[-1]
        else:
            model_path = "t5-small"

    print(f"\nLoading model from: {model_path}")

    try:
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path)
    except:
        print(f"Failed to load from {model_path}, trying base t5-small...")
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        model = T5ForConditionalGeneration.from_pretrained("t5-small")

    model = model.to(DEVICE)
    print(f"Model loaded successfully!")
    print(f"Model device: {next(model.parameters()).device}")

    return model, tokenizer, DEVICE


def generate_predictions(model, tokenizer, test_df, device):
    """Generate predictions for test data."""

    print("\nGenerating predictions...")

    # Prepare inputs
    test_texts = [
        "translate Akkadian to English: " + str(t)
        for t in test_df["transliteration"].tolist()
    ]

    # Tokenize
    test_inputs = tokenizer(
        test_texts, max_length=128, padding=True, truncation=True, return_tensors="pt"
    )

    # Move to device
    test_inputs = {k: v.to(device) for k, v in test_inputs.items()}

    # Generate
    outputs = model.generate(
        input_ids=test_inputs["input_ids"],
        attention_mask=test_inputs["attention_mask"],
        max_length=128,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3,
    )

    # Decode
    predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    print(f"Generated {len(predictions)} predictions")

    return predictions


def main():
    parser = argparse.ArgumentParser(description="Load model and create submission")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to model or HuggingFace model name",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="submission_final.csv",
        help="Output CSV file name",
    )
    args = parser.parse_args()

    # Load test data
    test_df = pd.read_csv("test.csv")
    print(f"Test data loaded: {len(test_df)} samples")
    print(test_df)

    # Load model
    model, tokenizer, device = load_model(args.model)

    # Generate predictions
    predictions = generate_predictions(model, tokenizer, test_df, device)

    # Print predictions
    print("\n--- Predictions ---")
    for i, pred in enumerate(predictions):
        print(f"\nTest {i}:")
        print(f"  Input:  {test_df['transliteration'].iloc[i][:60]}...")
        print(f"  Output: {pred[:100]}...")

    # Create submission
    submission = pd.DataFrame({"id": test_df["id"], "translation": predictions})

    # Save submission
    submission.to_csv(args.output, index=False)
    print(f"\nSubmission saved to {args.output}")

    # Verify format
    print("\n--- Verification ---")
    sample = pd.read_csv("sample_submission.csv")
    print(f"Sample columns: {sample.columns.tolist()}")
    print(f"Our columns:    {submission.columns.tolist()}")
    print(f"Match: {list(sample.columns) == list(submission.columns)}")

    print("\n" + "=" * 60)
    print("SUBMISSION SUMMARY")
    print("=" * 60)
    print(f"Output file: {args.output}")
    print(f"Test samples: {len(test_df)}")
    print("Done!")


if __name__ == "__main__":
    main()
