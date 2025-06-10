```bash
pip install mlflow jupyter pandas numpy scikit-learn xgboost hyperopt 
wget https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet


jupyter notebook

mlflow server \
    --backend-store-uri sqlite:///mlflow.db
```


```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("nyc-taxi-experiment-homework")
```

```python
URL = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet'
```

```python
jupyter nbconvert --to script 03-orchestration-homework-duration-prediction.ipynb
```

### Usage
```python
python 03-orchestration-homework-duration-prediction.py --year YEAR --month MONTH
```