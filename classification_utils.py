"""Reusable helpers for multiclass classification notebooks.

The functions in this module deliberately know nothing about a particular model.
They can be used with scikit-learn, PyTorch, TensorFlow, or Transformers as long
as predictions are represented by class IDs and optional class probabilities.
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    classification_report,
    precision_recall_fscore_support,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from datasets import Dataset
from transformers import TrainingArguments

category_to_id = {
    "medical": 0, "structure_damage": 1,
    "utility": 2, "missing_person": 3, "supply_request": 4,
    "flooding": 5, "trapped": 6, "fire": 7,
    "infrastructure_damage": 8, "evacuation": 9,
}

subcategory_to_id = {
    # Medical
    "injury": 0,
    "illness": 1,
    "heavy_bleeding": 2,
    "light_bleeding": 3,
    "unconscious": 4,

    # Fire
    "wildfire": 5,
    "structural_fire": 6,
    "vehicle_fire": 7,

    # Flooding
    "flash_flood": 8,
    "house_flooding": 9,
    "street_flooding": 10,

    # Utility
    "power_outage": 11,
    "water_outage": 12,
    "gas_leak": 13,

    # Missing Person
    "child": 14,
    "adult": 15,
    "elderly": 16,

    # Supply Request
    "food": 17,
    "water": 18,
    "clothing": 19,
    "other": 20,

    # Evacuation
    "voluntary": 21,
    "mandatory": 22,

    # Structure Damage
    "light": 23,
    "moderate": 24,
    "severe": 25,

    # Infrastructure Damage
    "road": 26,
    "bridge": 27,
    "utility": 28,
    "other": 29,

    # Trapped
    "in_building": 30,
    "in_vehicle": 31,
    "under_debris": 32,
}

chosen_category_to_id = subcategory_to_id
target_column = "subcategory"

def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch (when installed)."""

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ordered_class_names() -> list[str]:
    """Return class names in numeric ID order after validating the mapping."""
    return [name for name, _ in sorted(chosen_category_to_id.items(), key=lambda item: item[1])]


def load_labeled_csv(
    path: Union[str, Path],
    *,
    text_column: str,
    label_column: str,
    label_to_id: Mapping[str, int],
    encoded_label_column: str = "labels",
) -> pd.DataFrame:
    """Load, validate, and integer-encode a text-classification CSV."""

    frame = pd.read_csv(Path(path))
    required = {text_column, label_column}
    missing_columns = required.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    frame = frame.dropna(subset=[text_column, label_column]).copy()
    frame[text_column] = frame[text_column].astype(str)

    unknown_labels = sorted(set(frame[label_column]).difference(label_to_id))
    if unknown_labels:
        raise ValueError(f"Labels missing from label_to_id: {unknown_labels}")

    frame[encoded_label_column] = frame[label_column].map(label_to_id).astype(int)
    return frame


def stratified_split(
    frame: pd.DataFrame,
    *,
    label_column: str = "labels",
    train_size: float = 0.70,
    validation_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create reproducible train, validation, and test DataFrames."""

    sizes = np.asarray([train_size, validation_size, test_size], dtype=float)
    if np.any(sizes <= 0) or not np.isclose(sizes.sum(), 1.0):
        raise ValueError("Split sizes must be positive and sum to 1.0.")
    if label_column not in frame:
        raise ValueError(f"Missing label column: {label_column!r}")

    train, remainder = train_test_split(
        frame,
        train_size=train_size,
        random_state=seed,
        stratify=frame[label_column],
    )
    relative_test_size = test_size / (validation_size + test_size)
    validation, test = train_test_split(
        remainder,
        test_size=relative_test_size,
        random_state=seed,
        stratify=remainder[label_column],
    )
    return tuple(part.reset_index(drop=True) for part in (train, validation, test))

def tokenize_t5_dataset(
        tokenizer, 
        frame, 
        target_column: str = target_column,
        *,
        MAX_INPUT_LENGTH: int = 128, 
        MAX_TARGET_LENGTH: int = 12,
        INPUT_PREFIX: str = "classify emergency: "
    ):
    dataset = Dataset.from_pandas(
        frame[["message", target_column]].reset_index(drop=True),
        preserve_index=False,
    )

    def tokenize_batch(batch):
        model_inputs = tokenizer(
            [INPUT_PREFIX + text for text in batch["message"]],
            truncation=True,
            max_length=MAX_INPUT_LENGTH,
        )
        targets = tokenizer(
            text_target=batch[target_column],
            truncation=True,
            max_length=MAX_TARGET_LENGTH,
        )
        model_inputs["labels"] = targets["input_ids"]
        return model_inputs

    return dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=["message", target_column],
    )


def tokenize_bert_dataset(
    tokenizer,
    frame: pd.DataFrame,
    *,
    text_column: str = "message",
    label_column: str = "labels",
    max_length: int = 128,
):
    """Tokenize integer-labelled text for BERT sequence classification."""

    required = {text_column, label_column}
    missing_columns = required.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    if max_length <= 0:
        raise ValueError("max_length must be positive.")

    dataset = Dataset.from_pandas(
        frame[[text_column, label_column]].reset_index(drop=True),
        preserve_index=False,
    )

    def tokenize_batch(batch):
        return tokenizer(
            batch[text_column],
            truncation=True,
            max_length=max_length,
        )

    return dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=[text_column],
    )

def create_training_arguments(
    output_dir,
    learning_rate,
    train_batch_size,
    eval_batch_size,
    use_cpu=False,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    weight_decay=0.01,
    seed=42,
):
    """
    Create consistent Hugging Face TrainingArguments for model fine-tuning.

    Parameters intended to vary between models:
        output_dir: Directory for checkpoints/results.
        learning_rate: Model-specific learning rate.
        train_batch_size: Training batch size.
        eval_batch_size: Evaluation batch size. Defaults to train_batch_size.
        num_train_epochs: Maximum number of training epochs.
        warmup_ratio: Fraction of training used for learning-rate warmup.
        weight_decay: Weight decay used by the optimizer.
    """

    if eval_batch_size is None:
        eval_batch_size = train_batch_size

    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        weight_decay=weight_decay,
        use_cpu=use_cpu,

        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        report_to="none",
        seed=seed,
    )

def classification_metrics(
    true_labels: Sequence[int], predicted_labels: Sequence[int]
) -> dict[str, float]:
    """Calculate the shared scalar metrics used by the classifier notebooks."""

    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(macro_f1),
    }


def trainer_compute_metrics(evaluation) -> dict[str, float]:
    """Metric callback compatible with Hugging Face ``Trainer`` predictions."""

    predictions = evaluation.predictions
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    predicted_labels = np.asarray(predictions).argmax(axis=-1)
    return classification_metrics(evaluation.label_ids, predicted_labels)

def score_seq2seq_class_probabilities(
    model,
    tokenizer,
    texts: Sequence[str],
    class_names: Sequence[str],
    *,
    input_prefix: str = "",
    batch_size: int = 8,
    max_input_length: int = 128,
    max_target_length: int = 12,
    device=None,
) -> np.ndarray:
    """Score fixed class names with a Hugging Face seq2seq model.

    Each class name is scored by its mean token log-likelihood, then the class
    scores are normalized with a softmax. This makes generative classifiers
    such as T5 expose the same ``(examples, classes)`` probability matrix used
    by encoder-based classifiers and the analytics helpers below.
    """

    try:
        import torch
    except ImportError as error:
        raise ImportError("PyTorch is required to score a seq2seq model.") from error

    texts = list(texts)
    class_names = list(class_names)
    if not class_names:
        raise ValueError("class_names must contain at least one class.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if tokenizer.pad_token_id is None:
        raise ValueError("The tokenizer must define pad_token_id.")
    if not texts:
        return np.empty((0, len(class_names)), dtype=float)

    if device is None:
        try:
            device = next(model.parameters()).device
        except (AttributeError, StopIteration):
            device = model.device

    probability_batches = []
    was_training = model.training
    model.eval()
    try:
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            expanded_inputs = [
                input_prefix + text
                for text in batch_texts
                for _ in class_names
            ]
            expanded_targets = class_names * len(batch_texts)

            inputs = tokenizer(
                expanded_inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_input_length,
            ).to(device)
            target_tokens = tokenizer(
                text_target=expanded_targets,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_target_length,
            )
            labels = target_tokens["input_ids"].to(device)
            labels = labels.masked_fill(labels == tokenizer.pad_token_id, -100)

            with torch.inference_mode():
                logits = model(**inputs, labels=labels).logits
                token_log_probabilities = logits.log_softmax(dim=-1)

            safe_labels = labels.masked_fill(labels == -100, 0)
            selected_scores = token_log_probabilities.gather(
                2, safe_labels.unsqueeze(-1)
            ).squeeze(-1)
            token_mask = labels.ne(-100)
            sequence_scores = (selected_scores * token_mask).sum(dim=1)
            sequence_scores /= token_mask.sum(dim=1).clamp_min(1)
            sequence_scores = sequence_scores.view(len(batch_texts), len(class_names))
            probability_batches.append(sequence_scores.softmax(dim=1).cpu())
    finally:
        model.train(was_training)

    return torch.cat(probability_batches).numpy()


def score_lstm_awd_lstm_class_probabilities(
    learner,
    texts: Sequence[str],
    class_names: Sequence[str],
    *,
    text_column: str = "message",
    batch_size: int = 64,
) -> np.ndarray:
    """Return fastai AWD-LSTM probabilities in canonical class-name order."""

    try:
        from fastai.text.all import tokenize_df
    except ImportError as error:
        raise ImportError("fastai is required to score the AWD-LSTM model.") from error

    texts = list(texts)
    class_names = list(class_names)
    if not class_names:
        raise ValueError("class_names must contain at least one class.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if not texts:
        return np.empty((0, len(class_names)), dtype=float)

    frame = pd.DataFrame({text_column: [str(text) for text in texts]})
    tokenized_frame, _ = tokenize_df(
        frame, text_cols=text_column, n_workers=0
    )
    test_loader = learner.dls.test_dl(
        tokenized_frame,
        with_labels=False,
        bs=batch_size,
        num_workers=0,
    )
    probabilities = learner.get_preds(dl=test_loader)[0].cpu().numpy()

    learner_class_names = list(learner.dls.vocab[1])
    missing_classes = sorted(set(class_names).difference(learner_class_names))
    if missing_classes:
        raise ValueError(
            f"AWD-LSTM learner is missing configured classes: {missing_classes}"
        )
    canonical_columns = [learner_class_names.index(name) for name in class_names]
    return probabilities[:, canonical_columns]


def lstm_elmo_tokenize(text: str) -> list[str]:
    """Apply lightweight tokenization suitable for ELMo character inputs."""

    tokens = re.findall(r"\w+|[^\w\s]", str(text).lower())
    return tokens or ["<empty>"]


def score_lstm_elmo_class_probabilities(
    model,
    texts: Sequence[str],
    *,
    batch_size: int = 32,
    device=None,
) -> np.ndarray:
    """Return ELMo classifier probabilities with shape ``(rows, classes)``."""

    try:
        import torch
        from allennlp.modules.elmo import batch_to_ids
    except ImportError as error:
        raise ImportError(
            "PyTorch and AllenNLP are required to score the ELMo model."
        ) from error

    texts = list(texts)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    num_classes = int(model.classifier.out_features)
    if not texts:
        return np.empty((0, num_classes), dtype=float)

    if device is None:
        device = next(model.parameters()).device

    probability_batches = []
    was_training = model.training
    model.eval()
    try:
        for start in range(0, len(texts), batch_size):
            tokenized_texts = [
                lstm_elmo_tokenize(text)
                for text in texts[start : start + batch_size]
            ]
            character_ids = batch_to_ids(tokenized_texts).to(device)
            with torch.inference_mode():
                logits = model(character_ids)
            probability_batches.append(logits.softmax(dim=-1).cpu())
    finally:
        model.train(was_training)

    return torch.cat(probability_batches).numpy()


def score_bert_class_probabilities(
    model,
    tokenizer,
    texts: Sequence[str],
    *,
    batch_size: int = 32,
    max_length: int = 128,
    device=None,
) -> np.ndarray:
    """Return BERT classifier probabilities with shape ``(rows, classes)``."""

    try:
        import torch
    except ImportError as error:
        raise ImportError("PyTorch is required to score a BERT model.") from error

    texts = list(texts)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if max_length <= 0:
        raise ValueError("max_length must be positive.")

    num_labels = int(model.config.num_labels)
    if not texts:
        return np.empty((0, num_labels), dtype=float)

    if device is None:
        try:
            device = next(model.parameters()).device
        except (AttributeError, StopIteration):
            device = model.device

    probability_batches = []
    was_training = model.training
    model.eval()
    try:
        for start in range(0, len(texts), batch_size):
            inputs = tokenizer(
                texts[start : start + batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)
            with torch.inference_mode():
                logits = model(**inputs).logits
            probability_batches.append(logits.softmax(dim=-1).cpu())
    finally:
        model.train(was_training)

    return torch.cat(probability_batches).numpy()


def print_evaluation(
    true_labels: Sequence[int],
    predicted_labels: Sequence[int],
    class_names: Sequence[str],
) -> dict[str, float]:
    """Print scalar metrics and a per-class report, then return the metrics."""

    metrics = classification_metrics(true_labels, predicted_labels)
    for name, value in metrics.items():
        print(f"{name.replace('_', ' ').title()}: {value:.4f}")
    print()
    print(
        classification_report(
            true_labels,
            predicted_labels,
            labels=np.arange(len(class_names)),
            target_names=class_names,
            zero_division=0,
        )
    )
    return metrics


def plot_confusion_matrix(
    true_labels: Sequence[int],
    predicted_labels: Sequence[int],
    class_names: Sequence[str],
    *,
    title: str = "Normalized test confusion matrix",
    figsize: tuple[float, float] = (11, 11),
):
    """Plot a row-normalized multiclass confusion matrix and return its axes."""

    _, ax = plt.subplots(figsize=figsize)
    ConfusionMatrixDisplay.from_predictions(
        true_labels,
        predicted_labels,
        labels=np.arange(len(class_names)),
        display_labels=class_names,
        normalize="true",
        xticks_rotation=45,
        cmap="Blues",
        values_format=".2f",
        ax=ax,
    )
    ax.set_title(title)
    plt.tight_layout()
    return ax


def plot_micro_roc(
    true_labels: Sequence[int],
    probabilities: np.ndarray,
    class_names: Sequence[str],
    *,
    probability_class_ids: Optional[Sequence[int]] = None,
    model_name: str = "Model",
    figsize: tuple[float, float] = (7, 6),
) -> tuple[float, object]:
    """Plot micro-averaged one-vs-rest ROC and return ``(AUC, axes)``.

    ``probability_class_ids`` identifies the class represented by each input
    probability column. Supplying it allows evaluation when a training split
    did not contain every configured class (as can happen with small samples).
    Missing class columns are treated as zero probability.
    """

    if len(class_names) < 2:
        raise ValueError("ROC analytics require at least two classes.")

    probabilities = np.asarray(probabilities)
    expected_shape = (len(true_labels), len(class_names))
    if probability_class_ids is not None:
        probability_class_ids = np.asarray(probability_class_ids, dtype=int)
        if probabilities.shape != (len(true_labels), len(probability_class_ids)):
            raise ValueError(
                "probabilities columns must match probability_class_ids; "
                f"got {probabilities.shape} and {len(probability_class_ids)} IDs."
            )
        if (
            len(np.unique(probability_class_ids)) != len(probability_class_ids)
            or np.any(probability_class_ids < 0)
            or np.any(probability_class_ids >= len(class_names))
        ):
            raise ValueError("probability_class_ids contains invalid or duplicate IDs.")
        aligned_probabilities = np.zeros(expected_shape, dtype=probabilities.dtype)
        aligned_probabilities[:, probability_class_ids] = probabilities
        probabilities = aligned_probabilities
    elif probabilities.shape != expected_shape:
        raise ValueError(
            f"probabilities must have shape {expected_shape}; got {probabilities.shape}. "
            "Pass probability_class_ids if the model omitted classes."
        )

    binary_labels = label_binarize(
        true_labels, classes=np.arange(len(class_names))
    )
    if len(class_names) == 2:
        binary_labels = np.column_stack((1 - binary_labels, binary_labels))
    false_positive_rate, true_positive_rate, _ = roc_curve(
        binary_labels.ravel(), probabilities.ravel()
    )
    roc_auc = float(auc(false_positive_rate, true_positive_rate))

    _, ax = plt.subplots(figsize=figsize)
    ax.plot(
        false_positive_rate,
        true_positive_rate,
        linewidth=2,
        label=f"{model_name} (AUC = {roc_auc:.3f})",
    )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Chance")
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1.02),
        xlabel="False positive rate",
        ylabel="True positive rate",
        title=f"{model_name} test ROC curve",
    )
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    plt.tight_layout()
    return roc_auc, ax


def display_classification_analytics(
    true_labels: Sequence[int],
    class_names: Sequence[str],
    *,
    predicted_labels: Optional[Sequence[int]] = None,
    probabilities: Optional[np.ndarray] = None,
    probability_class_ids: Optional[Sequence[int]] = None,
    model_name: str = "Model",
    show: bool = True,
) -> dict[str, object]:
    """Print metrics and display reusable multiclass evaluation plots.

    Pass either predicted class IDs, class probabilities, or both. When only a
    probability matrix is supplied, predictions are derived from its largest
    value. ``probability_class_ids`` maps probability columns to class IDs for
    Hugging Face models trained on a subset or with a custom label order.
    """

    class_names = list(class_names)
    true_labels = np.asarray(true_labels)
    if not class_names:
        raise ValueError("class_names must contain at least one class.")
    if true_labels.ndim != 1:
        raise ValueError("true_labels must be one-dimensional.")
    if predicted_labels is None and probabilities is None:
        raise ValueError("Pass predicted_labels, probabilities, or both.")

    if probabilities is not None:
        probabilities = np.asarray(probabilities)
        if probabilities.ndim != 2 or probabilities.shape[0] != len(true_labels):
            raise ValueError(
                "probabilities must be a two-dimensional array with one row "
                "per true label."
            )
        if probability_class_ids is None:
            probability_class_ids_array = np.arange(len(class_names))
        else:
            probability_class_ids_array = np.asarray(
                probability_class_ids, dtype=int
            )
        if probabilities.shape[1] != len(probability_class_ids_array):
            raise ValueError(
                "probabilities columns must match probability_class_ids."
            )
        if (
            len(np.unique(probability_class_ids_array))
            != len(probability_class_ids_array)
            or np.any(probability_class_ids_array < 0)
            or np.any(probability_class_ids_array >= len(class_names))
        ):
            raise ValueError("probability_class_ids contains invalid or duplicate IDs.")
        if predicted_labels is None:
            predicted_labels = probability_class_ids_array[
                probabilities.argmax(axis=1)
            ]

    predicted_labels = np.asarray(predicted_labels)
    if predicted_labels.ndim != 1 or len(predicted_labels) != len(true_labels):
        raise ValueError("predicted_labels must contain one ID per true label.")

    metrics = print_evaluation(true_labels, predicted_labels, class_names)
    confusion_ax = plot_confusion_matrix(
        true_labels,
        predicted_labels,
        class_names,
        title=f"Normalized {model_name} test confusion matrix",
    )

    results: dict[str, object] = {
        "metrics": metrics,
        "predicted_labels": predicted_labels,
        "confusion_matrix_ax": confusion_ax,
    }
    if probabilities is not None:
        roc_auc, roc_ax = plot_micro_roc(
            true_labels,
            probabilities,
            class_names,
            probability_class_ids=probability_class_ids,
            model_name=model_name,
        )
        print(f"ROC AUC: {roc_auc:.4f}")
        results.update({"roc_auc": roc_auc, "roc_ax": roc_ax})

    if show:
        plt.show()
    return results
