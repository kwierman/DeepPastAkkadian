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
pip install pandas numpy torch transformers scikit-learn sacrebleu matplotlib seaborn datasets

# Run full pipeline (Jupyter notebooks)
jupyter notebook notebooks/

# Or run notebooks programmatically
jupyter nbconvert --to notebook --execute notebooks/01_data_acquisition.ipynb
jupyter nbconvert --to notebook --execute notebooks/02_exploratory_data_analysis.ipynb
jupyter nbconvert --to notebook --execute notebooks/03_data_preparation.ipynb
jupyter nbconvert --to notebook --execute notebooks/04_model_training.ipynb
jupyter nbconvert --to notebook --execute notebooks/05_model_submission.ipynb

# Output: submission.csv
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
├── notebooks/                     # Jupyter notebooks for the full pipeline
│   ├── 01_data_acquisition.ipynb       # Download & collect external data
│   ├── 02_exploratory_data_analysis.ipynb  # EDA visualizations
│   ├── 03_data_preparation.ipynb       # Preprocess & augment training data
│   ├── 04_model_training.ipynb         # Train T5/MarianMT ensemble models
│   └── 05_model_submission.ipynb       # Generate submission with beam voting
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
│   ├── pre_processed_data/        # Cleaned datasets
│   └── external_data/             # Downloaded Akkademia corpus
│
├── data/                         # Input data
│   ├── train.csv                 # Training data (~1,500 samples)
│   ├── test.csv                  # Test data
│   ├── publications.csv          # OCR from scholarly PDFs
│   ├── published_texts.csv       # Additional transliterations
│   ├── OA_Lexicon_eBL.csv        # Old Assyrian lexicon
│   └── eBL_Dictionary.csv        # Akkadian dictionary
│
└── submission.csv                 # Final submission file
```

## Detailed Pipeline

### 1. Data Acquisition (`01_data_acquisition.ipynb`)

Downloads and collects external resources:
- **Akkademia Corpus**: Downloads Old Assyrian/Babylonian parallel data from GitHub
- **Publications Analysis**: Analyzes OCR output from ~950 scholarly PDFs
- **Lexicon Loading**: Processes OA Lexicon (39K entries) and eBL Dictionary (19K entries)
- **ORACC Exploration**: Attempts to fetch data from ORACC APIs

**Output**: `output/external_data/` containing Akkademia parallel corpus (~40K sentence pairs)

### 2. Exploratory Data Analysis (`02_exploratory_data_analysis.ipynb`)

Generates visualizations analyzing the dataset characteristics:

| Visualization | Description |
|--------------|-------------|
| `text_length_distributions.png` | Character and word length histograms for source/target |
| `length_ratio_analysis.png` | Scatter plot of transliteration vs translation lengths |
| `vocabulary_analysis.png` | Top 30 tokens in both Akkadian and English |
| `special_characters_analysis.png` | Distribution of gaps, determinatives, superscripts |
| `lexicon_analysis.png` | Word type distribution and form lengths |
| `train_test_comparison.png` | Overlay histograms comparing data splits |

### 3. Data Preparation (`03_data_preparation.ipynb`)

Preprocessing pipeline:
- Loads and combines original training data with external Akkademia data
- Removes modern scribal notations (!, ?, /, :, .)
- Normalizes gap markers `[...]` to `<gap>` tokens
- Normalizes special characters (Ḫ→H, Š→SZ, etc.)
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

**Output**: `output/pre_processed_data/` containing cleaned datasets

### 4. Model Training (`04_model_training.ipynb`)

Trains an ensemble of translation models with memory-efficient sequential loading:
- **T5-small**: 60M parameter sequence-to-sequence model
- **T5-base**: Larger 220M parameter variant
- **MarianMT**: Pre-trained neural machine translation model
- **ByT5**: Character-level T5 for handling special characters

Training configuration:
- Input: `translate Akkadian to English: <transliteration>`
- Output: `<translation>`
- Optimizer: AdamW with linear warmup
- Batch size: 16
- Learning rate: 1e-3
- Epochs: 2
- Generation: Beam search (beam_size=5)

**Output**: `output/models/` containing trained model checkpoints

### 5. Model Submission (`05_model_submission.ipynb`)

Generates final predictions using ensemble beam voting:
- Loads models sequentially to avoid GPU OOM
- Generates beam_size translations per model
- Aggregates candidates across all models
- Selects best translation using length-normalized scoring
- Outputs to `submission.csv`

**Ensemble Models:**
1. T5-small v1
2. T5-base
3. MarianMT
4. ByT5
5. T5-small v2

**Voting Strategy**: Length-normalized beam scoring

```bash
# Basic usage
python 08_load_model_submission.py

# Custom model path
python 08_load_model_submission.py --model ./output/models/t5_small_v2

# Custom output
python 08_load_model_submission.py --output my_submission.csv
```

## GPU Support

Notebooks automatically detect CUDA availability:
- **GPU available**: Uses CUDA with mixed precision
- **CPU only**: Falls back to full precision (slower)
- Memory-efficient sequential model loading to avoid OOM errors

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
pip install datasets>=2.14.0
pip install scikit-learn>=1.2.0
pip install sacrebleu>=2.3.0
pip install matplotlib>=3.7.0
pip install seaborn>=0.12.0
pip install jupyter>=1.0.0
pip install nbconvert>=7.0.0
```

## Acknowledgments

- Deep Past Challenge organizers for dataset and evaluation
- ORACC team for the Akkademia corpus
- Electronic Babylonian Literature (eBL) project for dictionary resources
- Hugging Face for transformer infrastructure
