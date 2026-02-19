import mlflow
import mlflow.sklearn
from mlflow.client import MlflowClient
from mlflow.models import infer_signature

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import column

from utils.config import MLFLOW_TRACKING_URI, EXPERIMENT_NAME, MODEL_NAME
from utils.logger import get_logger

from components.model_builder import build_grid, get_models
from components.evaluation import evaluate_model

logger = get_logger(__name__)


def train_pipeline(df):
    logger.info("Starting training pipeline...")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    logger.info("Splitting data...")

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = get_models()

    best_accuracy = 0
    best_model = None
    best_model_name = None

    for model_name, config in models.items():

        logger.info(f"Training {model_name}...")

        with mlflow.start_run(run_name=model_name):

            pipeline = Pipeline(
                [("scaler", StandardScaler()), ("model", config["model"])]
            )

            grid = build_grid(pipeline, config["params"], is_pipeline=True)

            grid.fit(X_train, y_train)

            model = grid.best_estimator_

            accuracy, report = evaluate_model(model, X_test, y_test)

            mlflow.log_params(grid.best_params_)
            mlflow.log_metric("Accuracy", accuracy)
            mlflow.log_text(report, "classification_report.txt")

            signature = infer_signature(X_train, model.predict(X_test))
            mlflow.sklearn.log_model(
                sk_model=model, artifact_path="model", signature=signature
            )

            logger.info(f"{model_name} Accuracy: {accuracy}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_model_name = model_name

    logger.info(
        f"Best Model Selected: {best_model_name} with Accuracy: {best_accuracy}"
    )

    # Register only the best model
    with mlflow.start_run(run_name="Best_Model_Registration"):

        signature = infer_signature(X_train, best_model.predict(X_test))

        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="best_model",
            signature=signature,
            registered_model_name=MODEL_NAME,
        )

    client = MlflowClient()

    latest_version = client.get_latest_versions(MODEL_NAME)[-1].version

    logger.info(
        f"Champion Model Version: {latest_version} with Accuracy: {best_accuracy}"
    )

    return best_model
