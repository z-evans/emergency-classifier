#!/usr/bin/env python3
"""Generate a disaster-only synthetic emergency-message dataset."""

import argparse
import csv
import json
import random
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
DEFAULT_STRUCTURE = DATA_DIR / "disaster_sentence_structure_with_medical.json"
DEFAULT_CATEGORIES = DATA_DIR / "categories.json"
FIELDNAMES = ["type", "category", "subcategory", "priority", "message"]


def load_json(path):
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def load_disaster_data(
    structure_path=DEFAULT_STRUCTURE, categories_path=DEFAULT_CATEGORIES
):
    """Load and validate the disaster phrases and their category metadata."""
    structure = load_json(Path(structure_path))
    category_data = load_json(Path(categories_path))

    disaster_metadata = {
        category["id"]: category
        for category in category_data
    }
    disaster_phrases = {
        category_id: subcategories
        for category_group in structure["categories"]
        for category_id, subcategories in category_group.items()
    }

    missing_metadata = disaster_phrases.keys() - disaster_metadata.keys()
    if missing_metadata:
        names = ", ".join(sorted(missing_metadata))
        raise ValueError(f"Missing category metadata for: {names}")

    missing_phrases = disaster_metadata.keys() - disaster_phrases.keys()
    if missing_phrases:
        names = ", ".join(sorted(missing_phrases))
        raise ValueError(f"Missing sentence structures for: {names}")

    for category_id, subcategories in disaster_phrases.items():
        known_subcategories = {
            subcategory["id"] for subcategory in disaster_metadata[category_id]["sub"]
        }
        unknown = subcategories.keys() - known_subcategories
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown {category_id} subcategories: {names}")
        if not subcategories or any(not phrases for phrases in subcategories.values()):
            raise ValueError(f"Every {category_id} subcategory needs at least one phrase")

    return structure, disaster_metadata, disaster_phrases


def clean_part(part):
    """Remove sentence-edge punctuation so joined messages have consistent separators."""
    return part.strip().rstrip(".!? ")


def generate_disaster_message(structure, metadata, phrases, rng=random):
    """Generate one disaster message without selecting any medical content."""
    category_id = rng.choice(list(phrases))
    subcategory_id = rng.choice(list(phrases[category_id]))
    candidates = phrases[category_id][subcategory_id]

    parts = []
    opening = clean_part(rng.choice(structure["opening"]))
    if opening:
        parts.append(opening)

    first_phrase = rng.choice(candidates)
    parts.append(clean_part(first_phrase))

    # A second description gives some messages the repetition and fragmented
    # context commonly found in real emergency reports.
    if len(candidates) > 1 and rng.random() < 0.4:
        alternatives = [phrase for phrase in candidates if phrase != first_phrase]
        if alternatives:
            parts.append(clean_part(rng.choice(alternatives)))

    closing = clean_part(rng.choice(structure["closing"]))
    if closing:
        parts.append(closing)

    priority_by_subcategory = {
        item["id"]: item.get("priority")
        for item in metadata[category_id]["sub"]
    }

    return {
        "type": "disaster",
        "category": category_id,
        "subcategory": subcategory_id,
        "priority": priority_by_subcategory[subcategory_id],
        "message": ". ".join(parts) + ".",
    }


def generate_dataset(count, output_file, seed=None):
    """Write ``count`` disaster-only records to a CSV file."""
    if count <= 0:
        raise ValueError("count must be a positive integer")

    output_path = Path(output_file)
    if output_path.suffix.lower() != ".csv":
        raise ValueError("output_file must use the .csv extension")

    structure, metadata, phrases = load_disaster_data()
    rng = random.Random(seed)
    messages = []
    seen_records = set()
    max_attempts = count * 100
    attempts = 0
    progress_interval = max(1, count // 100)

    while len(messages) < count and attempts < max_attempts:
        message = generate_disaster_message(structure, metadata, phrases, rng)
        record = tuple(message[field] for field in FIELDNAMES)
        attempts += 1

        if record not in seen_records:
            seen_records.add(record)
            messages.append(message)

        if attempts % progress_interval == 0 or len(messages) == count:
            print(
                f"\rGenerated rows: {len(messages)}/{count} | Attempts: {attempts}",
                end="",
                flush=True,
            )

    print()

    if len(messages) < count:
        raise ValueError(
            f"Could only generate {len(messages)} unique records after "
            f"{max_attempts} attempts; request a smaller dataset"
        )

    rng.shuffle(messages)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(messages)

    return output_path, messages


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a CSV dataset containing disaster messages only."
    )
    parser.add_argument("count", nargs="?", type=int, default=100)
    parser.add_argument("output_file", nargs="?", default="disaster_messages.csv")
    parser.add_argument("--seed", type=int, help="Make generation reproducible")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        output_path, messages = generate_dataset(args.count, args.output_file, args.seed)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise SystemExit(f"Error: {error}") from error

    print(f"Generated {len(messages)} disaster messages in {output_path}")


if __name__ == "__main__":
    main()
