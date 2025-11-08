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

### Download Dataset

Find the latest release, and follow the instructions to download the most recent dataset. Then, continue reading.

### Clean Dataset

You can simpliy run the following command to clean the dataset: `python -m src.core.data_cleansing`.

```
usage: data_cleansing.py [-h] --mode {dataset,json} --in IN_PATH [--out OUT_PATH] [--preview] [--save]

Clean misinformation data (CSV dataset or single JSON input).

options:
  -h, --help            show this help message and exit
  --mode {dataset,json}
                        Choose whether to clean a dataset CSV or a single JSON input.
  --in IN_PATH          Input CSV file or JSON string/file path.
  --out OUT_PATH        Output file or directory path.
  --preview             Show preview summary (dataset mode only).
  --save                Save cleaned output to file (JSON mode only).
```

### Find Max Length:

```sh
python -m src.utils.suggest_maxlen
```

## Model Training

### Prepare Features

You can simpily run `python -m src.models.model_2.prepare_features --preview`. You can pass in the `-h` flag to view all command line options as well.

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

```
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

### Evaluate model:

```python
python -m src.models.model_2.evaluate_bilstm
```

## Using the Model for Prediction

Input json string:

```python
python -m src.models.model_2.predict_input --json '{}'
```
