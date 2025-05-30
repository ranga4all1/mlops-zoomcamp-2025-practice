## Conda env
```
conda create -n mlops python=3.12
conda activate mlops

pip install -r requirements.txt
pip list | grep mlflow

mlflow --version
```

## Data prep
```
python homework-scripts/preprocess_data.py --raw_data_path ./TAXI_DATA_FOLDER --dest_path ./output

ls -l output/
```

## Model training
```
python homework-scripts/train.py

mlflow ui
```

## Local tracking server
```
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root artifacts

python homework-scripts/train.py
```

## Tune model hyperparameters
```
python homework-scripts/hpo.py
```

