#!/usr/bin/env python3
"""Benchmark end-to-end latency and throughput of the saved classifiers.

The timed path includes tokenization, category inference, construction of the
category-conditioned input, subcategory inference, and final selection. Model
loading and warm-up are deliberately outside the measurements.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent
MODEL_ROOT = ROOT / "models"
DEFAULT_FAMILIES = ("distilbert_base", "bert_base", "t5_small", "t5_base", "awd_lstm")
CATEGORY_TO_SUBCATEGORIES = {
    "evacuation": ["mandatory", "voluntary"],
    "fire": ["structural_fire", "vehicle_fire", "wildfire"],
    "flooding": ["flash_flood", "house_flooding", "street_flooding"],
    "infrastructure_damage": ["bridge", "other", "road", "utility"],
    "medical": ["heavy_bleeding", "illness", "injury", "light_bleeding", "unconscious"],
    "missing_person": ["adult", "child", "elderly"],
    "structure_damage": ["light", "moderate", "severe"],
    "supply_request": ["clothing", "food", "other", "water"],
    "trapped": ["in_building", "in_vehicle", "under_debris"],
    "utility": ["gas_leak", "power_outage", "water_outage"],
}
CATEGORY_NAMES = list(CATEGORY_TO_SUBCATEGORIES)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values), q))


def cpu_model_name() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


class BertPipeline:
    def __init__(self, family: str, device: torch.device):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        category_path = MODEL_ROOT / f"{family}_category"
        subcategory_path = MODEL_ROOT / f"{family}_subcategory"
        self.tokenizer = AutoTokenizer.from_pretrained(category_path, local_files_only=True)
        self.category_model = AutoModelForSequenceClassification.from_pretrained(
            category_path, local_files_only=True
        ).to(device).eval()
        self.subcategory_model = AutoModelForSequenceClassification.from_pretrained(
            subcategory_path, local_files_only=True
        ).to(device).eval()
        self.device = device

    def __call__(self, texts: list[str]) -> list[tuple[str, str]]:
        encoded = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=128
        ).to(self.device)
        with torch.inference_mode():
            category_ids = self.category_model(**encoded).logits.argmax(dim=-1).tolist()
        categories = [self.category_model.config.id2label[i] for i in category_ids]
        stage_two = [f"category: {category} message: {text}" for category, text in zip(categories, texts)]
        encoded = self.tokenizer(
            stage_two, return_tensors="pt", padding=True, truncation=True, max_length=128
        ).to(self.device)
        with torch.inference_mode():
            logits = self.subcategory_model(**encoded).logits
        results = []
        for row, category in zip(logits, categories):
            allowed = [
                i for i, label in self.subcategory_model.config.id2label.items()
                if label.startswith(category + "/")
            ]
            winner = allowed[int(row[allowed].argmax())]
            results.append((category, self.subcategory_model.config.id2label[winner].split("/", 1)[1]))
        return results


class T5Pipeline:
    CATEGORY_PREFIX = "classify emergency category: "
    SUBCATEGORY_PREFIX = "classify emergency subcategory: "

    def __init__(self, family: str, device: torch.device):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        category_path = MODEL_ROOT / f"{family}_category"
        subcategory_path = MODEL_ROOT / f"{family}_subcategory"
        self.tokenizer = AutoTokenizer.from_pretrained(category_path, local_files_only=True)
        self.category_model = AutoModelForSeq2SeqLM.from_pretrained(
            category_path, local_files_only=True
        ).to(device).eval()
        self.subcategory_model = AutoModelForSeq2SeqLM.from_pretrained(
            subcategory_path, local_files_only=True
        ).to(device).eval()
        self.device = device

    def _choose(self, model, texts: list[str], candidates: list[str], prefix: str) -> list[str]:
        expanded_inputs = [prefix + text for text in texts for _ in candidates]
        expanded_targets = candidates * len(texts)
        encoded = self.tokenizer(
            expanded_inputs, return_tensors="pt", padding=True, truncation=True, max_length=128
        ).to(self.device)
        targets = self.tokenizer(
            text_target=expanded_targets,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=12,
        )["input_ids"].to(self.device)
        labels = targets.masked_fill(targets == self.tokenizer.pad_token_id, -100)
        with torch.inference_mode():
            logits = model(**encoded, labels=labels).logits.log_softmax(dim=-1)
            safe_labels = labels.masked_fill(labels == -100, 0)
            selected = logits.gather(2, safe_labels.unsqueeze(-1)).squeeze(-1)
            mask = labels.ne(-100)
            scores = (selected * mask).sum(1) / mask.sum(1).clamp_min(1)
            winners = scores.view(len(texts), len(candidates)).argmax(dim=1).tolist()
        return [candidates[i] for i in winners]

    def __call__(self, texts: list[str]) -> list[tuple[str, str]]:
        categories = self._choose(self.category_model, texts, CATEGORY_NAMES, self.CATEGORY_PREFIX)
        grouped: dict[str, list[tuple[int, str]]] = defaultdict(list)
        for index, (category, text) in enumerate(zip(categories, texts)):
            grouped[category].append((index, f"category: {category} message: {text}"))
        subcategories = [""] * len(texts)
        for category, indexed_texts in grouped.items():
            choices = self._choose(
                self.subcategory_model,
                [text for _, text in indexed_texts],
                CATEGORY_TO_SUBCATEGORIES[category],
                self.SUBCATEGORY_PREFIX,
            )
            for (index, _), choice in zip(indexed_texts, choices):
                subcategories[index] = choice
        return list(zip(categories, subcategories))


class AwdLstmPipeline:
    def __init__(self, _family: str, device: torch.device):
        if device.type != "cpu":
            raise ValueError("The exported AWD-LSTM learners are benchmarked on CPU only.")
        from fastai.text.all import load_learner

        self.category_learner = load_learner(MODEL_ROOT / "awd_lstm_category.pkl", cpu=True)
        self.subcategory_learner = load_learner(MODEL_ROOT / "awd_lstm_subcategory.pkl", cpu=True)

    @staticmethod
    def _scores(learner, texts: list[str], batch_size: int) -> tuple[np.ndarray, list[str]]:
        from fastai.text.all import tokenize_df

        frame = pd.DataFrame({"message": texts})
        tokenized, _ = tokenize_df(frame, text_cols="message", n_workers=0)
        loader = learner.dls.test_dl(
            tokenized, with_labels=False, bs=batch_size, num_workers=0
        )
        with learner.no_bar():
            scores = learner.get_preds(dl=loader)[0].cpu().numpy()
        return scores, list(learner.dls.vocab[1])

    def __call__(self, texts: list[str]) -> list[tuple[str, str]]:
        category_scores, category_vocab = self._scores(self.category_learner, texts, len(texts))
        categories = [category_vocab[i] for i in category_scores.argmax(axis=1)]
        stage_two = [f"category: {category} message: {text}" for category, text in zip(categories, texts)]
        subcategory_scores, subcategory_vocab = self._scores(
            self.subcategory_learner, stage_two, len(texts)
        )
        results = []
        for row, category in zip(subcategory_scores, categories):
            allowed = [i for i, label in enumerate(subcategory_vocab) if label.startswith(category + "/")]
            winner = allowed[int(row[allowed].argmax())]
            results.append((category, subcategory_vocab[winner].split("/", 1)[1]))
        return results


def make_pipeline(family: str, device: torch.device):
    if family in {"bert_base", "distilbert_base"}:
        return BertPipeline(family, device)
    if family in {"t5_small", "t5_base"}:
        return T5Pipeline(family, device)
    if family == "awd_lstm":
        return AwdLstmPipeline(family, device)
    raise ValueError(f"Unknown family: {family}")


def benchmark_family(
    family: str,
    messages: list[str],
    batch_sizes: list[int],
    warmups: int,
    repeats: int,
    device: torch.device,
) -> list[dict]:
    print(f"Loading {family} ...", flush=True)
    pipeline = make_pipeline(family, device)
    rows = []
    for batch_size in batch_sizes:
        batch = [messages[i % len(messages)] for i in range(batch_size)]
        for _ in range(warmups):
            pipeline(batch)
        sync(device)
        durations = []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            predictions = pipeline(batch)
            sync(device)
            durations.append((time.perf_counter_ns() - start) / 1e9)
        total_seconds = sum(durations)
        per_message_ms = [duration * 1000 / batch_size for duration in durations]
        row = {
            "model": family,
            "device": str(device),
            "batch_size": batch_size,
            "warmups": warmups,
            "repeats": repeats,
            "samples": batch_size * repeats,
            "total_seconds": total_seconds,
            "throughput_messages_per_second": batch_size * repeats / total_seconds,
            "latency_mean_ms_per_message": statistics.mean(per_message_ms),
            "latency_median_ms_per_message": statistics.median(per_message_ms),
            "latency_p95_ms_per_message": percentile(per_message_ms, 95),
            "batch_latency_mean_ms": statistics.mean(durations) * 1000,
            "example_prediction": "/".join(predictions[0]),
        }
        rows.append(row)
        print(
            f"  batch={batch_size:<3} mean={row['latency_mean_ms_per_message']:.2f} ms/msg "
            f"p95={row['latency_p95_ms_per_message']:.2f} ms/msg "
            f"throughput={row['throughput_messages_per_second']:.2f} msg/s",
            flush=True,
        )
    del pipeline
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", nargs="+", choices=DEFAULT_FAMILIES, default=list(DEFAULT_FAMILIES))
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8, 32])
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmark_results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.batch_sizes) <= 0 or args.warmups < 0 or args.repeats <= 0 or args.threads <= 0:
        raise SystemExit("Batch sizes, repeats, and threads must be positive; warmups may be zero.")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable.")
    device = torch.device(device_name)
    messages = pd.read_csv(ROOT / "disaster_messages.csv")["message"].dropna().astype(str).tolist()
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "processor": cpu_model_name(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "timing_scope": "tokenization + category inference + conditioned subcategory inference + post-processing",
        "model_loading_timed": False,
        "input_source": "disaster_messages.csv",
    }
    rows = []
    for family in args.families:
        rows.extend(
            benchmark_family(
                family, messages, args.batch_sizes, args.warmups, args.repeats, device
            )
        )
    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "inference_benchmark.csv"
    json_path = args.output / "inference_benchmark.json"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps({"metadata": metadata, "results": rows}, indent=2) + "\n")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
