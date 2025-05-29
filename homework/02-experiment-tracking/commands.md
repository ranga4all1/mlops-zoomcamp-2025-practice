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
cd homework-scripts
python preprocess_data.py --raw_data_path ../TAXI_DATA_FOLDER --dest_path ../output

cd ../output
ls -l
```

```
mlflow ui --backend-store-uri sqlite:///mlflow.db 
```