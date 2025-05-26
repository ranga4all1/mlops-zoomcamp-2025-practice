```
conda create -n mlops python=3.12
conda activate mlops

pip install -r requirements.txt
pip list | grep mlflow
```


```
mlflow ui --backend-store-uri sqlite:///mlflow.db 
```