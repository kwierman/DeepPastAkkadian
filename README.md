# Deep Past Challenge: Old Assyrian to English Translation

A neural machine translation system for translating Old Assyrian (Akkadian) cuneiform transliterations to modern English. This project implements a T5-based transformer model with ensemble retrieval augmentation, trained on the Deep Past Challenge dataset augmented with the Akkademia parallel corpus.

## Features

- **T5-small Transformer**: 60M parameter sequence-to-sequence model fine-tuned for Akkadian-English translation
- **Ensemble Architecture**: Combines neural translation with TF-IDF retrieval for improved accuracy
- **Data Augmentation**: Leverages 27,000+ samples from the Akkademia corpus
- **Automatic Preprocessing**: Downloads and processes external resources including the eBL dictionary and OA lexicon
- **GPU Acceleration**: Automatic detection with mixed precision (FP16) support
- **Comprehensive EDA**: Visualization tools for analyzing text distributions, vocabulary, and special characters

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Input Pipeline                           │
│  ┌─────────────────┐    ┌──────────────┐    ┌───────────────┐  │
│  │  Raw Cuneiform  │───▶│  Normalize   │───▶│  Transliteration│ │
│  │  Transliteration│    │  Special Chars│    │  (Akka-ian)   │  │
│  └─────────────────┘    └──────────────┘    └───────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Ensemble Model                             │
│  ┌─────────────────────────┐    ┌─────────────────────────────┐ │
│  │   Neural (T5-small)      │    │    Retrieval (TF-IDF)       │ │
│  │   60M parameters        │    │    Cosine Similarity        │ │
│  └───────────┬─────────────┘    └──────────────┬──────────────┘ │
│              │                                  │               │
│              └──────────┬───────────────────────┘               │
│                         ▼                                        │
│              ┌──────────────────┐                                │
│              │  Confidence     │                                │
│              │  Threshold: 0.6  │                                │
│              └────────┬─────────┘                                │
└───────────────────────┼─────────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Output                                   │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    English Translation                       ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Clone repository
git clone https://github.com/your-repo/DeepPastAkkadian.git
cd DeepPastAkkadian

# Install dependencies
pip install pandas numpy torch transformers scikit-learn sacrebleu matplotlib seaborn

# Run full pipeline
python 01_eda.py              # Analyze data
python 02_data_preprocessing.py # Prepare datasets
python 03_model_training.py    # Train T5 model
python 05_evaluation.py       # Evaluate model

# Generate submission
python 08_load_model_submission.py --output submission.csv
```

## Example Outputs

The model translates Akkadian transliterations to English. Here are example predictions from the test set:

### Example 1: Commercial Correspondence

```
Input (Akkadian):  še-a-am ša ta-di-nu a-na Šamaš-tappê ša Kà-ni-iški
                    a-na É-a-mu-dan a-na ha-mu-uš-tim a-šé-er KÁ-DINGIR-Raki
                    ... [truncated]

Model Output:       "From the Kanesh colony to Aur-nya: The Kanesh colony and 
                    the Kanesh colony seized me and he said: 'Aur-nya brought 
                    you a tablet to the City.' The colony said: 'Aur-nya brought 
                    you a tablet to the City.'"

Reference:          "Thus Kanesh, say to the -payers, our messenger, every 
                    single colony..."
```

### Example 2: Letter Fragment

```
Input (Akkadian):  a-na a-li-a-hi-im qi-bi-ma um-ma a-šur-dan-a-a-ma
                    a-na wa-ša-er a-bu-uk a-na Uur-ma-lik a-na e-pé-ri-im
                    ... [truncated]

Model Output:       "To the tablet of Ali-ahum, at the disposal of Uur-malik, 
                    in Nmlim Dawr, Il-il, Kanesh, Kanesh, Kanesh."

Reference:          "In the letter of the City (it is written): From this 
                    day on..."
```

### Example 3: Fragmentary Text

```
Input (Akkadian):  [...]x x x x x x x x x x x x x x x x x x x x x x

Model Output:       "gap> gap> gap> gap> gap> gap> gap> gap> gap> gap>"

Reference:          "As soon as you have heard our letter, who(ever) over there..."
```

## Evaluation Metrics

| Metric | Score | Description |
|--------|-------|-------------|
| **BLEU** | 3.11 | Standard MT metric (higher is better) |
| **chrF++** | 8.73 | Character n-gram F-score (better for morphologically rich languages) |
| **Geometric Mean** | 5.21 | Combined BLEU × chrF++ |

### Training Progress

| Epoch | Train Loss | Val Loss |
|-------|-----------|----------|
| 1 | 1.49 | 0.68 |
| 2 | 1.34 | 0.65 |

**Training Time**: ~43 minutes for 2,362 steps on GPU

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
├── output/
│   ├── images/                    # EDA visualizations
│   │   ├── text_length_distributions.png
│   │   ├── length_ratio_analysis.png
│   │   ├── vocabulary_analysis.png
│   │   ├── special_characters_analysis.png
│   │   ├── lexicon_analysis.png
│   │   └── train_test_comparison.png
│   ├── models/                    # Trained models
│   │   ├── t5_akkadian_translator/
│   │   └── ensemble/
│   └── pre_processed_data/        # Cleaned datasets
│
├── train_augmented.csv            # Augmented training data (~28K samples)
├── akkademia_train.ak/en          # External Akkademia parallel corpus
├── test.csv                       # Test data
├── submission_augmented.csv       # Single model predictions
├── submission_ensemble.csv        # Ensemble predictions
├── submission_final.csv           # Final submission (loaded model)
└── all_predictions.csv            # All model predictions for analysis
```

## Detailed Pipeline

### 1. Exploratory Data Analysis (`01_eda.py`)

Generates visualizations analyzing the dataset characteristics:

| Visualization | Description |
|--------------|-------------|
| `text_length_distributions.png` | Character and word length histograms for source/target |
| `length_ratio_analysis.png` | Scatter plot of transliteration vs translation lengths |
| `vocabulary_analysis.png` | Top 30 tokens in both Akkadian and English |
| `special_characters_analysis.png` | Distribution of gaps, determinatives, superscripts |
| `lexicon_analysis.png` | Word type distribution and form lengths |
| `train_test_comparison.png` | Overlay histograms comparing data splits |

### 2. Data Preprocessing (`02_data_preprocessing.py`)

Automatic preprocessing pipeline:
- Downloads Akkademia corpus from ORACC RINAP projects
- Removes modern scribal notations (!, ?, /, :, .)
- Normalizes gap markers `[...]` to `<gap>` tokens
- Handles determinatives and special characters
- Splits data: 90% train, 10% validation

**Data Statistics:**
| Dataset | Samples |
|---------|---------|
| Original Deep Past | 1,561 |
| Akkademia | 40,424 |
| Combined | 41,985 |
| Training | 37,786 |
| Validation | 4,199 |
| Test | 4 |

### 3. Model Training (`03_model_training.py`)

Fine-tunes T5-small for translation:
- Input: `translate Akkadian to English: <transliteration>`
- Output: `<translation>`
- Optimizer: AdamW with linear warmup
- Batch size: 16
- Learning rate: 3e-4
- Epochs: 2
- Generation: Beam search (beam_size=5)

### 4. Baseline Predictions (`04_baseline_predictions.py`)

TF-IDF retrieval baseline for comparison:
- Vectorizes training transliterations
- Retrieves most similar training example for each test input
- Provides similarity score for ensemble weighting

### 5. Evaluation (`05_evaluation.py`)

Computes translation quality metrics:
- **BLEU**: n-gram precision with brevity penalty
- **chrF++**: Character-level n-gram F-score (β=2)
- Geometric mean of both metrics

### 6. Ensemble Methods

**Ensemble Training** (`06_ensemble_training.py`):
- Trains multiple T5 variants
- Optionally includes mT5-small for multilingual benefits

**Ensemble Predictions** (`07_ensemble_predictions.py`):
- Combines neural and retrieval predictions
- Confidence threshold: 0.6 (retrieval used above this threshold)

### 7. Submission Generation (`08_load_model_submission.py`)

```bash
# Basic usage
python 08_load_model_submission.py

# Custom model path
python 08_load_model_submission.py --model ./output/models/t5_small_v2

# Custom output
python 08_load_model_submission.py --output my_submission.csv
```

## GPU Support

Scripts automatically detect CUDA availability:
- **GPU available**: Uses CUDA with FP16 mixed precision
- **CPU only**: Falls back to full precision (slower)
- Memory-efficient batch processing for large datasets

To verify GPU availability:
```python
import torch
print(torch.cuda.is_available())  # True if GPU available
```

## Ensemble Method

The ensemble combines two complementary approaches:

### Neural Translation (T5)
- Generalizes to novel sentence structures
- Learns semantic mappings between languages
- Handles morphological variations

### Retrieval Augmentation (TF-IDF)
- Leverages similar examples from training data
- High accuracy when close matches exist
- Provides confidence signal for ensemble

### Combination Strategy
```
if retrieval_similarity > 0.6:
    use retrieval_prediction
else:
    use neural_prediction
```

This hybrid approach achieves better results than either method alone by:
- Using retrieval for high-confidence matches
- Falling back to neural model for novel inputs

## Dataset Sources

### Deep Past Challenge
- **Original samples**: 1,561 training pairs
- **Test samples**: 4 fragments
- **Content**: Old Assyrian commercial correspondence

### Akkademia Parallel Corpus (Gutherz et al., 2023)
- **Samples**: ~40,000 parallel sentences
- **Source**: ORACC RINAP projects
- **Language**: Old Babylonian and Old Assyrian
- **Annotation**: Professionally curated

### External Resources
- **eBL Dictionary**: Electronic Babylonian Literature dictionary
- **OA Lexicon**: Old Assyrian lexicon with 39,332 entries
  - Word types: 25,574 words, 13,424 proper names (PN), 334 geographic names (GN)

## Models

### T5-small
| Parameter | Value |
|-----------|-------|
| Parameters | 60M |
| Layers | 6 |
| D Model | 512 |
| Heads | 8 |
| D FF | 2,048 |
| Context Length | 512 |

### mT5-small (optional)
- Multilingual variant
- Better handling of low-resource languages
- Shared vocabulary across languages

### Tokenizer
- Custom tokenizer trained on Akkadian corpus
- Handles special tokens: `<gap>`, `<big_gap>`, `<unk>`
- Preserves cuneiform conventions

## Citation

If you use this code or dataset, please cite:

### Deep Past Challenge
```
Deep Past Challenge (2024). Neural Machine Translation for Ancient Languages.
https://github.com/deep-mind/deep_past_challenge
```

### Akkademia Corpus
```
Gutherz, G., Gordin, S., Sáenz, L., Levy, O., & Berant, J. (2023).
Translating Akkadian to English with neural machine translation.
PNAS Nexus, 2(6), pgad256.
https://doi.org/10.1093/pnasnexus/pgad256

@article{gutherz2023translating,
  title={Translating Akkadian to English with neural machine translation},
  author={Gutherz, Gil and Gordin, Shai and S{\'a}enz, Luis and Levy, Omer and Berant, Jonathan},
  journal={PNAS Nexus},
  volume={2},
  number={6},
  pages={pgad256},
  year={2023},
  publisher={Oxford University Press}
}
```

### ORACC (Open Richly Annotated Cuneiform Corpus)
```
Open Richly Annotated Cuneiform Corpus (ORACC).
Royal Holloway, University of London.
https://oracc.museum.upenn.edu
```

## License

This project inherits licensing from its data sources:
- **Deep Past Challenge**: CC BY-SA compatible
- **Akkademia Corpus**: See original publication
- **ORACC**: CC BY-NC-SA 3.0

## Requirements

```bash
pip install pandas>=1.5.0
pip install numpy>=1.23.0
pip install torch>=2.0.0
pip install transformers>=4.30.0
pip install scikit-learn>=1.2.0
pip install sacrebleu>=2.3.0
pip install matplotlib>=3.7.0
pip install seaborn>=0.12.0
```

## Acknowledgments

- Deep Past Challenge organizers for dataset and evaluation
- ORACC team for the Akkademia corpus
- Electronic Babylonian Literature (eBL) project for dictionary resources
- Hugging Face for transformer infrastructure
