# pymisinfo

## Overview
pymisinfo is a research-driven machine-learning project focused on detecting and addressing misinformation on social media platforms.  
The system aims to enhance online safety and contribute to cybersecurity by analyzing linguistic and contextual features that distinguish factual information from misinformation.

This project is part of an academic initiative exploring the intersection of artificial intelligence, cybersecurity, and digital ethics.

## Objectives
- Develop multiple machine-learning pipelines to detect misinformation on social media.  
- Evaluate linguistic and semantic indicators contributing to misinformation classification.  
- Promote safer online environments through data-driven cybersecurity research.

## Technology Stack
- Language: Python  
- Frameworks/Libraries: NumPy, pandas, scikit-learn, TensorFlow/Keras, joblib, ftfy 

## Project Structure
```
pymisinfo/
├── src/
│   ├── core/                  # Core logic and text cleansing utilities
│   ├── data/                  # Raw and processed datasets
│   ├── models/                # Model implementations and training scripts
│   │   ├── model_1/           # Logistic Regression (baseline)
│   │   ├── model_2/           # BiLSTM (deep learning)
│   │   └── model_3/           # Random Forest / Decision Forests
│   ├── utils/                 # Shared utilities
├── requirements.txt
├── README.txt
└── LICENSE
```

## Environment Setup (venv)
To ensure reproducibility and isolate dependencies, this project uses a Python virtual environment (venv).

1. Create the Virtual Environment

    ```sh
    python3 -m venv venv
    ```

2. Activate the Virtual Environment

    ```sh
    # Linux/Mac
    source venv/bin/activate
    ```

    ```powershell
    # On Windows
    venv\Scripts\activate
    ```

3. Install Dependencies
   ```sh
   pip install -r requirements.txt
   ```

## Data Processing

Clean Dataset:

```python
python -m src.core.data_cleansing
python -m src.core.data_cleansing --preview
```

Find Max Length:

```python
python -m src.utils.suggest_maxlen
```

## Model Training

### Prepare Features

You can simpily run `python -m python -m src.models.model_2.prepare_features --preview`. You can pass in the `-h` flag to view all command line options as well.

```
usage: prepare_features.py [-h] [--in IN_PATH] [--out OUT_DIR] [--maxlen MAXLEN] [--preview]

Tokenize, encode, and split the cleaned dataset.

options:
  -h, --help       show this help message and exit
  --in IN_PATH     Path to cleaned CSV file
  --out OUT_DIR    Output directory for tokenized data
  --maxlen MAXLEN
  --preview        Print output summary to console
```

### Train model:

You can train the model by running `python -m src.models.model_2.train_bilstm --epochs 12 --batch 64`. You can pass in the `-h` flag to view all command line options. 

```python
usage: train_bilstm.py [-h] [--features FEATURES] [--artifacts ARTIFACTS] [--maxlen MAXLEN] [--epochs EPOCHS] [--batch BATCH] [--no-class-weight] [--out OUT_MODEL_PATH] [--use-date]

Train BiLSTM + Attention on processed misinformation dataset.

options:
  -h, --help            show this help message and exit
  --features FEATURES
  --artifacts ARTIFACTS
  --maxlen MAXLEN
  --epochs EPOCHS
  --batch BATCH
  --no-class-weight
  --out OUT_MODEL_PATH
  --use-date            Include normalized date feature in training

```

Evaluate model:

```python
python -m src.models.model_2.evaluate_bilstm
```

## Using the Model for Prediction

Input json string:

```python
python -m src.models.model_2.predict_input --json '{}'
```
