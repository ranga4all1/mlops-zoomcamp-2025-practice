import os
import pickle
import click

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("rf-nyc-taxi-experiment")


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):
    print(f"Loading data from {data_path}")
    print(f"Using MLflow tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Using MLflow experiment: {mlflow.get_experiment_by_name('rf-nyc-taxi-experiment').name}")
    print(f"Using MLflow experiment ID: {mlflow.get_experiment_by_name('rf-nyc-taxi-experiment').experiment_id}")
    print(f"Using MLflow run ID: {mlflow.active_run().info.run_id if mlflow.active_run() else 'No active run'}")

    mlflow.sklearn.autolog()

    with mlflow.start_run():

        mlflow.log_param("train-data-path", "./output/train.pkl")
        mlflow.log_param("valid-data-path", "./output/val.pkl")
        # mlflow.log_param("model-type", "RandomForestRegressor")
        # mlflow.log_param("max-depth", 10)
        # mlflow.log_param("random-state", 0)
        
        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = root_mean_squared_error(y_val, y_pred)
        print(f"logging RMSE: {rmse:.3f}")
        mlflow.log_metric("rmse", rmse)

        print(f"Logging model to MLflow...")
        mlflow.sklearn.log_model(rf, "model")
        
        print("Model training completed and logged to MLflow.")
        print(f"Run ID: {mlflow.active_run().info.run_id}")
        
if __name__ == '__main__':
    run_train()
