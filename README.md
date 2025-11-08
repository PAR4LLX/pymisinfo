# pymisinfo

Clean Dataset:
python -m src.core.data_cleansing
python -m src.core.data_cleansing --preview

Find Max Length:
python -m src.utils.suggest_maxlen

Prepare Features:
python -m src.models.model_2.prepare_features
python -m src.models.model_2.prepare_features --preview

Train model:
python -m src.models.model_2.train_bilstm --epochs 12 --batch 64

Evaluate model: (Currently Broken)
python -m src.models.model_2.evaluate_bilstm

Test Pipeline: (Currently Broken)
python -m src.models.model_2.test_pipeline_integrity

Input json string:
python -m src.models.model_2.predict_input --json '{}'
