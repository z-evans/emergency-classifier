# Emergency Message Classifier

A research project for hierarchical classification of short emergency messages. It compares five pretrained NLP architectures—AWD-LSTM, BERT-base, DistilBERT-base, T5-small, and T5-base—under the same two-stage classification task.

The first model identifies a broad emergency category. A second model receives that predicted category together with the original message and selects one of its valid subcategories.

```mermaid
flowchart LR
    A[Emergency message] --> B[Stage 1: category model]
    B --> C[Predicted category]
    A --> D[Stage 2: conditioned subcategory model]
    C --> D
    D --> E[Valid category / subcategory path]
```

> [!CAUTION]
> This is an experimental classifier, not a dispatch or medical-triage system. Its predictions should not be used as the sole basis for emergency decisions.

## What is included

- Five end-to-end training and evaluation notebooks.
- A common 70:15:15 stratified train/validation/test split with seed 42.
- Shared preprocessing, scoring, metrics, and visualisation utilities.
- Hierarchically constrained inference that prevents invalid category/subcategory combinations.
- CPU and CUDA latency/throughput benchmarking for saved models.
- Accuracy, macro precision, macro recall, macro F1, confusion-matrix, and one-vs-rest ROC-AUC analysis.
- A synthetic disaster-message CSV generator, with optional reproducible sampling.

## Dataset

[`disaster_messages.csv`](disaster_messages.csv) contains 10,000 labelled messages and the following columns:

| Column | Description |
| --- | --- |
| `type` | Message type recorded by the dataset. |
| `category` | One of ten broad emergency categories. |
| `subcategory` | Fine-grained label within the category. |
| `priority` | Numeric priority supplied with the message. It is not used as a model input. |
| `message` | Text classified by the models. |

The dataset has 10 categories, 32 distinct subcategory names, and 33 valid category/subcategory paths. The path count is larger because `utility` is both a broad category and a subcategory of `infrastructure_damage`.

| Category | Valid subcategories |
| --- | --- |
| `medical` | `injury`, `illness`, `heavy_bleeding`, `light_bleeding`, `unconscious` |
| `fire` | `wildfire`, `structural_fire`, `vehicle_fire` |
| `flooding` | `flash_flood`, `house_flooding`, `street_flooding` |
| `utility` | `power_outage`, `water_outage`, `gas_leak` |
| `missing_person` | `child`, `adult`, `elderly` |
| `supply_request` | `food`, `water`, `clothing`, `other` |
| `evacuation` | `voluntary`, `mandatory` |
| `structure_damage` | `light`, `moderate`, `severe` |
| `infrastructure_damage` | `road`, `bridge`, `utility`, `other` |
| `trapped` | `in_building`, `in_vehicle`, `under_debris` |

The split is stratified by complete category/subcategory path, producing 7,000 training, 1,500 validation, and 1,500 test messages. Stage 2 uses the true category during training and validation, then uses Stage 1's prediction during end-to-end testing.

## Generate synthetic messages

[`generate.py`](generate.py) assembles synthetic disaster messages from phrase templates and category metadata using only the Python standard library. It writes the same five CSV columns described above, with `type` set to `disaster` and priority taken from the selected subcategory's metadata.

Before generating data, supply these files under `data/` beside the script; they are currently absent from the repository:

- `data/disaster_sentence_structure_with_medical.json`: an object containing `opening` and `closing` phrase lists and a `categories` list of objects mapping category IDs to subcategory IDs and their phrase lists.
- `data/categories.json`: a list of category objects with `id` and `sub` fields; each `sub` list contains objects with a subcategory `id` and optional `priority`.

The script currently defines `main()` without calling it, so invoke it explicitly from the repository root:

```bash
python -c 'import generate; generate.main()' 10000 generated/disaster_messages.csv --seed 42
```

The positional arguments are the positive row count (default `100`) and output path (default `disaster_messages.csv`, relative to the working directory). The output must have a `.csv` extension. Parent directories are created automatically, and an existing output file is overwritten. The example uses a separate output directory to preserve the checked-in dataset.

`--seed` makes sampling and output order reproducible for the same inputs. The generator removes duplicate complete records, shuffles the result, and fails if it cannot produce enough unique records within `count × 100` attempts. Categories and then their subcategories are sampled randomly, so label counts are not guaranteed to be balanced. Generated datasets do not necessarily match the checked-in dataset's label coverage or recorded results.

To view the arguments:

```bash
python -c 'import generate; generate.main()' --help
```

## Recorded results

These are the outputs currently stored in the notebooks for the shared 1,500-message test partition. Subcategory results are end-to-end: they include any errors propagated from category prediction.

| Model | Category accuracy | Category macro F1 | End-to-end accuracy | End-to-end macro F1 |
| --- | ---: | ---: | ---: | ---: |
| AWD-LSTM | 98.33% | 98.36% | 97.80% | 97.84% |
| BERT-base | **100.00%** | **100.00%** | **99.93%** | **99.90%** |
| DistilBERT-base | **100.00%** | **100.00%** | 99.80% | 99.74% |
| T5-small | 98.33% | 98.34% | 93.73% | 93.35% |
| T5-base | **100.00%** | **100.00%** | 99.87% | 99.84% |

These figures come from one fixed split and one random seed, so small differences—especially among BERT-base, DistilBERT-base, and T5-base—should not be treated as statistically conclusive. The archived T5-small pre-tuning notebook also evaluated the same split while parameters were being revised; the final T5 results are therefore tuning-aware rather than measurements from a completely untouched confirmation set.

## Installation

Python 3.10 or newer is recommended. A CUDA-capable GPU is optional but substantially reduces Transformer training time.

```bash
git clone git@github.com:z-evans/emergency-classifier.git
cd emergency-classifier

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements include fastai for the AWD-LSTM notebook and SentencePiece support for the T5 notebooks.

The first run of a notebook downloads its pretrained checkpoint and tokenizer. Ensure that the environment has network access and enough disk space for the selected model.

## Train and evaluate a model

Start Jupyter from the repository root so that relative paths to the dataset and shared utility module resolve correctly:

```bash
jupyter notebook
```

Open one of the following notebooks and run its cells in order:

| Notebook | Pretrained model | Training approach |
| --- | --- | --- |
| [`distilbert-base-category-subcategory.ipynb`](distilbert-base-category-subcategory.ipynb) | `distilbert-base-uncased` | Two sequence-classification heads; 128-token limit. |
| [`bert-base-category-subcategory.ipynb`](bert-base-category-subcategory.ipynb) | `bert-base-uncased` | Two sequence-classification heads; 128-token limit. |
| [`t5-small-category-subcategory.ipynb`](t5-small-category-subcategory.ipynb) | `t5-small` | Text-to-text category and subcategory generation. |
| [`t5-base-category-subcategory.ipynb`](t5-base-category-subcategory.ipynb) | `t5-base` | Text-to-text category and subcategory generation. |
| [`lstm-awd-category-subcategory.ipynb`](lstm-awd-category-subcategory.ipynb) | pretrained AWD-LSTM | Frozen-head training followed by gradual unfreezing. |

[`t5-small-category-subcategory-pre-manual-tuning.ipynb`](t5-small-category-subcategory-pre-manual-tuning.ipynb) preserves the initial T5-small configuration for experimental provenance. Use the main T5-small notebook for the final configuration.

Every main notebook performs the same high-level workflow:

1. Validate and encode the labels.
2. Create the shared stratified split.
3. Build separate category and subcategory datasets.
4. Fine-tune one model for each stage.
5. Evaluate category performance and the complete cascade.
6. Save both trained models under `models/`.
7. Define `predict_emergency(text)` and show example predictions.

For example, after running a notebook's training and save cells:

```python
predict_emergency("A person is trapped inside a vehicle")
```

The function returns the predicted category, subcategory, and their confidence values. Confidence for Stage 2 is conditional on the category chosen by Stage 1.

Generated `models/`, `results/`, and `logs/` directories are ignored by Git.

## Training configuration

| Architecture | Input length | Train batch | Effective batch | Learning rate | Epochs |
| --- | ---: | ---: | ---: | ---: | ---: |
| AWD-LSTM | 72 | 64 | 64 | $1 \times 10^{-2}$ | 1 frozen + 3 unfrozen |
| BERT-base | 128 | 16 | 16 | $2 \times 10^{-5}$ | 3 maximum |
| DistilBERT-base | 128 | 16 | 16 | $2 \times 10^{-5}$ | 3 maximum |
| T5-small | 128 input / 12 target | 16 | 64 | $1 \times 10^{-4}$ | 3 maximum |
| T5-base | 128 input / 12 target | 16 | 64 | $1 \times 10^{-4}$ | 3 maximum |

Transformer models use AdamW, weight decay 0.01, epoch-level validation and checkpointing, early-stopping patience of two evaluations, and restoration of the checkpoint with the lowest validation loss. T5 obtains its effective batch of 64 through four gradient-accumulation steps.

## Benchmark saved models

[`benchmark_inference.py`](benchmark_inference.py) measures the complete prediction path: tokenisation, category inference, construction of the conditioned Stage 2 input, subcategory inference, and final selection. Model loading and warm-up are excluded from timed measurements.

Train the requested families first so their artefacts exist under `models/`, then run, for example:

```bash
python benchmark_inference.py \
  --families distilbert_base bert_base t5_small \
  --batch-sizes 1 8 32 \
  --warmups 2 \
  --repeats 10 \
  --threads 8 \
  --device auto \
  --output benchmark_results
```

Supported family names are:

```text
distilbert_base bert_base t5_small t5_base awd_lstm
```

The benchmark writes `inference_benchmark.csv` and `inference_benchmark.json` to the selected output directory. `--device auto` uses CUDA when available and otherwise uses the CPU. Exported AWD-LSTM learners are benchmarked on CPU only.

View all command-line options with:

```bash
python benchmark_inference.py --help
```

## Project structure

```text
emergency-classifier/
├── bert-base-category-subcategory.ipynb
├── distilbert-base-category-subcategory.ipynb
├── lstm-awd-category-subcategory.ipynb
├── t5-base-category-subcategory.ipynb
├── t5-small-category-subcategory.ipynb
├── t5-small-category-subcategory-pre-manual-tuning.ipynb
├── benchmark_inference.py
├── classification_utils.py
├── disaster_messages.csv
├── generate.py
├── requirements.txt
├── LICENSE
└── README.md
```

[`classification_utils.py`](classification_utils.py) contains the shared label mappings, split and tokenisation helpers, metrics, ROC analysis, model scoring, and batched prediction functions used by the notebooks.

## License

This project is available under the [MIT License](LICENSE).
