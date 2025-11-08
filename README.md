pymisinfo
==========

Overview
--------
pymisinfo is a research-driven machine-learning project focused on detecting and addressing misinformation on social media platforms.  
The system aims to enhance online safety and contribute to cybersecurity by analyzing linguistic and contextual features that distinguish factual information from misinformation.

This project is part of an academic initiative exploring the intersection of artificial intelligence, cybersecurity, and digital ethics.

Objectives
----------
- Develop multiple machine-learning pipelines to detect misinformation on social media.  
- Evaluate linguistic and semantic indicators contributing to misinformation classification.  
- Promote safer online environments through data-driven cybersecurity research.

Technology Stack
----------------
- Language: Python  
- Frameworks/Libraries: NumPy, pandas, scikit-learn, TensorFlow/Keras, joblib, ftfy 

Project Structure
-----------------
pymisinfo/\
├── src/\
│   ├── core/                  # Core logic and text cleansing utilities\
│   ├── data/                  # Raw and processed datasets\
│   ├── models/                # Model implementations and training scripts\
│   │   ├── model_1/           # Logistic Regression (baseline)\
│   │   ├── model_2/           # BiLSTM (deep learning)\
│   │   └── model_3/           # Random Forest / Decision Forests\
│   ├── utils/                 # Shared utilities\
├── requirements.txt\
├── README.txt\
└── LICENSE\

Environment Setup (venv)
-------------------------
To ensure reproducibility and isolate dependencies, this project uses a Python virtual environment (venv).

Step 1 — Create the Virtual Environment \
    python3 -m venv venv

Step 2 — Activate the Virtual Environment \
On Linux or macOS:\
    source venv/bin/activate

On Windows:\
    venv\Scripts\activate

Step 3 — Install Dependencies \
    pip install -r requirements.txt

Data Processing
---------------
Clean Dataset:\
python -m src.core.data_cleansing \
python -m src.core.data_cleansing --preview

Find Max Length:\
python -m src.utils.suggest_maxlen

Model Training
--------------
Prepare Features:\
python -m src.models.model_2.prepare_features \
python -m src.models.model_2.prepare_features --preview

Train model:\
python -m src.models.model_2.train_bilstm --epochs 12 --batch 64

Evaluate model:\
python -m src.models.model_2.evaluate_bilstm

Using the Model for Prediction
------------------------------
Input json string:\
python -m src.models.model_2.predict_input --json '{}'
