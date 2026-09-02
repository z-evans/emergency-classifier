# Emergency Message Classifier

A multi-model machine learning system for hierarchical classification of emergency messages into categories and subcategories. This project compares multiple state-of-the-art NLP architectures (DistilBERT, BERT, T5, LSTM-AWD) for emergency message classification with performance benchmarking.

## Overview

This project provides a complete pipeline for:

- **Hierarchical Classification**: Two-stage classification system (category → subcategory)
- **Multi-Model Training**: Compare performance across different architectures
- **Performance Benchmarking**: End-to-end latency and throughput measurements
- **Inference Utilities**: Reusable classification helpers and evaluation metrics

## Features

- **Multiple Model Architectures**:
  - DistilBERT (efficient, recommended for production)
  - BERT (higher accuracy)
  - T5 (sequence-to-sequence classification)
  - LSTM-AWD (recurrent neural networks)

- **Hierarchical Classification**:
  - Stage 1: Classify into 10 emergency categories
  - Stage 2: Classify into 35+ subcategories (category-conditioned)

- **Comprehensive Evaluation**:
  - Confusion matrices
  - Precision/Recall/F1 scores
  - ROC-AUC curves
  - Training loss tracking

- **Production-Ready Benchmarking**:
  - End-to-end inference timing
  - Throughput measurements
  - Multi-device support (CPU/GPU)
  - Batch processing support

## Project Structure

```
emergency-classifier/
├── bert-base-category-subcategory.ipynb         # BERT training notebook
├── distilbert-base-category-subcategory.ipynb   # DistilBERT training notebook
├── t5-base-category-subcategory.ipynb           # T5-Base training notebook
├── t5-small-category-subcategory.ipynb          # T5-Small training notebook
├── lstm-awd-category-subcategory.ipynb          # LSTM-AWD training notebook
├── classification_utils.py                      # Shared evaluation utilities
├── benchmark_inference.py                       # Inference benchmarking script
├── disaster_messages.csv                        # Training dataset (~1.1 MB)
└── LICENSE
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (optional, for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd emergency-classifier

# Create virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `matplotlib` - Plotting and visualization
- `scikit-learn` - ML metrics and utilities
- `torch` - PyTorch framework
- `transformers` - HuggingFace model library
- `datasets` - Dataset loading and processing

## Dataset

**File**: `disaster_messages.csv` (~1.1 MB)

Contains labeled emergency messages with the following columns:

| Column          | Values                                          | Description                       |
| --------------- | ----------------------------------------------- | --------------------------------- |
| **category**    | medical, fire, flooding, trapped, etc.          | Primary emergency type            |
| **subcategory** | injury, wildfire, flash_flood, in_vehicle, etc. | Specific emergency classification |
| **message**     | Text                                            | The emergency message             |

### Emergency Categories & Subcategories

```
1. medical: injury, illness, heavy_bleeding, light_bleeding, unconscious
2. fire: structural_fire, vehicle_fire, wildfire
3. flooding: flash_flood, house_flooding, street_flooding
4. trapped: in_building, in_vehicle, under_debris
5. missing_person: adult, child, elderly
6. structure_damage: light, moderate, severe
7. utility: gas_leak, power_outage, water_outage
8. supply_request: clothing, food, other, water
9. infrastructure_damage: bridge, other, road, utility
10. evacuation: mandatory, voluntary
```

## Usage

### Training a Model

Open any notebook in Jupyter:

```bash
jupyter notebook distilbert-base-category-subcategory.ipynb
```

Each notebook includes:

1. Data loading and exploration
2. Preprocessing and tokenization
3. Model training with validation
4. Performance evaluation
5. Model saving

**Quick Start - DistilBERT** (recommended for efficiency):

```bash
jupyter notebook distilbert-base-category-subcategory.ipynb
# Run all cells (Shift+Enter)
```

### Benchmarking Inference

Compare model performance across architectures:

```bash
python benchmark_inference.py --models distilbert_base bert_base t5_small --num-runs 100
```

**Options**:

- `--models` - Models to benchmark (default: all)
- `--num-runs` - Number of inference runs (default: 100)
- `--batch-size` - Batch size for inference (default: 1)
- `--device` - Device to use (cuda/cpu, default: auto)
- `--output` - JSON file for results (default: benchmark_results.json)

### Classification Utilities

The `classification_utils.py` module provides reusable functions:

```python
from classification_utils import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    roc_curve
)

# Use with any model predictions
report = classification_report(y_true, y_pred)
print(report)
```

## Model Comparison

| Model      | Size | Speed  | Accuracy | Use Case             |
| ---------- | ---- | ------ | -------- | -------------------- |
| DistilBERT | 268M | ⚡⚡⚡ | ✓✓       | Production, mobile   |
| BERT       | 340M | ⚡⚡   | ✓✓✓      | High accuracy needed |
| T5-Small   | 60M  | ⚡⚡⚡ | ✓✓       | Lightweight          |
| T5-Base    | 220M | ⚡⚡   | ✓✓✓      | Balanced             |
| LSTM-AWD   | 200M | ⚡     | ✓        | Research, custom     |

## Development

### Adding New Categories

1. Update the category/subcategory mappings in `classification_utils.py`
2. Retrain models with updated dataset
3. Re-run benchmarks for performance comparison

### Modifying Training Pipeline

Each notebook is self-contained and can be customized:

- Adjust hyperparameters (learning rate, batch size, epochs)
- Modify data preprocessing
- Change model architecture
- Add custom loss functions

## License

MIT License - see [LICENSE](LICENSE) file for details.

Copyright (c) Zack Evans

## References

- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [DistilBERT Paper](https://arxiv.org/abs/1910.01108)
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)](https://arxiv.org/abs/1910.10683)
