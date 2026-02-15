# Emergency Message Classifier

A machine learning project that generates synthetic emergency messages and trains a DistilBERT model to classify them into various emergency categories (medical, fire, flooding, trapped, etc.).

## Overview

This project provides tools for:

- **Synthetic Data Generation**: Create realistic emergency messages with varying priorities and contexts
- **Model Training**: Train a DistilBERT-based classifier to categorize emergency messages

## Features

- Generates realistic synthetic emergency messages
- Fine-tuned DistilBERT model for emergency classification
- Supports multiple emergency categories with priority levels
- Hierarchical classification (type → category → subcategory)
- Jupyter notebook for training and evaluation

## Project Structure

```
emergency-classifier/
├── generate.py                      # Synthetic message generator
├── distilbert-classifier.ipynb      # Model training notebook
├── sample.csv                       # Sample dataset (10,000 messages)
├── data/
│   ├── categories.json              # Emergency category definitions
│   ├── disaster_sentence_structure.json
│   └── medical_sentence_structure.json
```

## Installation

### Dependencies

- Python 3.7+
- pandas
- numpy
- matplotlib
- scikit-learn
- transformers (HuggingFace)
- datasets
- torch (PyTorch)

## Usage

### Generating Synthetic Messages

#### Generate Sample Messages (Demo Mode)

```bash
python generate.py
```

This will print 3 sample messages (medical, disaster, and random).

#### Generate Dataset

```bash
python generate.py <num_medical> <num_disaster> <output_file>
```

**Example:**

```bash
python generate.py 5000 5000 emergency_messages.csv
```

This creates a CSV file with 10,000 messages (5,000 medical + 5,000 disaster).

**Output Format:**

```csv
type,category,subcategory,priority,message
medical,medical,heavy_bleeding,1,"Someone is hurt!. person bleeding heavily. Patient is middle-aged person..."
disaster,flooding,flash_flood,1,"hlp pls. flash flood coming. sudden flood waters rising..."
```

### Training the Classifier

1. Open [distilbert-classifier.ipynb](distilbert-classifier.ipynb) in Jupyter Notebook or JupyterLab
2. Ensure `sample.csv` (or your generated dataset) is in the project root
3. Run all cells to:
   - Load and explore the data
   - Preprocess messages
   - Train the DistilBERT model
   - Evaluate performance
   - Save the trained model

## Data Format

The generated CSV contains the following columns:

- **type**: `medical` or `disaster`
- **category**: Main emergency category (e.g., `medical`, `fire`, `flooding`)
- **subcategory**: Specific emergency type (e.g., `injury`, `wildfire`, `flash_flood`)
- **priority**: Urgency level (1=highest, 5=lowest, or empty)
- **message**: The emergency message text

## Customization

### Adding New Categories

Edit `data/categories.json` to add new emergency types:

```json
{
  "id": "earthquake",
  "name": "Earthquake",
  "sub": [
    {
      "id": "major",
      "name": "Major Earthquake",
      "priority": 1
    }
  ]
}
```

Then update the corresponding sentence structure files in `data/`.

### Adjusting Message Generation

Modify the sentence structure files to customize generated messages:

- `data/medical_sentence_structure.json` - Medical message templates
- `data/disaster_sentence_structure.json` - Disaster message templates

## License

MIT License - see [LICENSE](LICENSE) file for details.

Copyright (c) Zack Evans

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Acknowledgments

- Built with [HuggingFace Transformers](https://huggingface.co/transformers/)
- Uses [DistilBERT](https://huggingface.co/distilbert-base-uncased) for efficient text classification
