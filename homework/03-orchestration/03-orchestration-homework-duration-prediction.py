#!/usr/bin/env python
# coding: utf-8

import pickle
from pathlib import Path

import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("nyc-taxi-experiment-homework")

models_folder = Path('models')
models_folder.mkdir(exist_ok=True)


def read_dataframe(year, month):

    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet"
    df = pd.read_parquet(url)

    print(f"Number of records in raw df: {df.shape}")

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    print(f"Number of records in df after filtering: {df.shape}")

    return df


def create_X(df, dv=None):
    categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')

    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv


def train_model(X_train, y_train, dv):

    mlflow.sklearn.autolog()

    with mlflow.start_run() as run:

        # mlflow.log_param("train-data-path", "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet")

        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        intercept = lr_model.intercept_

        mlflow.log_metric("intercept", intercept)

        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)

        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        mlflow.sklearn.log_model(lr_model, artifact_path="models_mlflow")

        return run.info.run_id


def run(year, month):
    df_train = read_dataframe(year=year, month=month)

    # next_year = year if month < 12 else year + 1
    # next_month = month + 1 if month < 12 else 1

    X_train, dv = create_X(df_train)

    target = 'duration'
    y_train = df_train[target].values
  
    run_id = train_model(X_train, y_train, dv)
    print(f"MLflow run_id: {run_id}")
    return run_id


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train a model to predict taxi trip duration.')
    parser.add_argument('--year', type=int, required=True, help='Year of the data to train on')
    parser.add_argument('--month', type=int, required=True, help='Month of the data to train on')
    args = parser.parse_args()

    run_id = run(year=args.year, month=args.month)

    with open("run_id.txt", "w") as f:
        f.write(run_id)

