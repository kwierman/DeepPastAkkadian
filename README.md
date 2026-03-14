# Deep Past Challenge: Old Assyrian to English Translation

This project contains scripts for training a neural machine translation model to translate Old Assyrian (Akkadian) cuneiform transliterations to English.

## Project Structure

```
.
├── 01_eda.py                      # Exploratory Data Analysis
├── 02_data_preprocessing.py       # Data preprocessing & augmentation
├── 03_model_training.py           # T5 model training
├── 04_baseline_predictions.py     # TF-IDF retrieval baseline
├── 05_evaluation.py               # Model evaluation
├── 06_ensemble_training.py        # Ensemble model training
├── 07_ensemble_predictions.py     # Ensemble predictions
├── 08_load_model_submission.py    # Load model and create submission
│
├── train_augmented.csv            # Augmented training data (~28K samples)
├── akkademia_train.ak/en          # External Akkademia parallel corpus
├── test.csv                       # Test data
├── submission_augmented.csv       # Single model predictions
├── submission_ensemble.csv        # Ensemble predictions
├── submission_final.csv           # Final submission (loaded model)
└── all_predictions.csv            # All model predictions for analysis
```

## Requirements

```bash
pip install pandas numpy torch transformers scikit-learn sacrebleu matplotlib seaborn
```

## Usage

### Basic Pipeline

```bash
# 1. Exploratory Data Analysis
python 01_eda.py

# 2. Data Preprocessing (downloads Akkademia data automatically)
python 02_data_preprocessing.py

# 3. Model Training (requires GPU recommended)
python 03_model_training.py

# 4. Baseline Predictions (optional, for comparison)
python 04_baseline_predictions.py

# 5. Evaluation
python 05_evaluation.py
```

### Ensemble Pipeline

```bash
# Train ensemble models (T5 + optional mT5)
python 06_ensemble_training.py

# Generate ensemble predictions
python 07_ensemble_predictions.py
```

### Load Model and Generate Submission

```bash
# Load pre-trained model and generate submission
python 08_load_model_submission.py

# Specify custom model path
python 08_load_model_submission.py --model ./ensemble_models/t5_small

# Specify output file
python 08_load_model_submission.py --output my_submission.csv
```

## GPU Support

Scripts automatically detect CUDA availability and use GPU acceleration when available:
- Falls back to CPU if no GPU detected
- Uses fp16 mixed precision when GPU available

## Ensemble Method

The ensemble combines:
1. **Neural Model (T5)**: Transformer-based translation
2. **Retrieval (TF-IDF)**: Similarity-based retrieval

The ensemble uses a confidence-based strategy:
- If retrieval similarity > 0.6: use retrieval prediction
- Otherwise: use neural prediction

This hybrid approach leverages the strengths of both methods:
- Retrieval works well when similar examples exist in training data
- Neural model generalizes better for novel inputs

## Dataset Sources

- **Original Deep Past Challenge**: ~1,500 samples
- **Akkademia** (Gutherz et al. 2023): ~27,000 samples from ORACC RINAP projects

Total augmented dataset: ~28,000 samples

## Models

- **T5-small**: 60M parameters, seq2seq transformer
- **mT5-small**: Multilingual T5, better for low-resource languages
- **Ensemble**: Combines neural + retrieval predictions

## Citation

Gutherz, G., Gordin, S., Sáenz, L., Levy, O., Berant, J. (2023). Translating Akkadian to English with neural machine translation. PNAS Nexus.
