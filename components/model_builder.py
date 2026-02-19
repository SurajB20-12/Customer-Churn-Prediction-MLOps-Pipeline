from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

from utils.logger import get_logger

logger = get_logger(__name__)


def get_models():
    logger.info("Initializing models and Hyperparameters")

    models = {
        "LogisticRegression": {
            "model": LogisticRegression(),
            "params": {
                "C": [0.01, 0.1, 1, 10],
                "solver": ["lbfgs", "liblinear"],
                "max_iter": [100, 200, 500],
            },
        },
        "DecisionTree": {
            "model": DecisionTreeClassifier(),
            "params": {"max_depth": [5, 10, None], "min_samples_split": [2, 5]},
        },
        "RandomForest": {
            "model": RandomForestClassifier(),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [5, 10, None],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
            },
        },
        "XGBoost": {
            "model": XGBClassifier(),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [3, 6],
                "learning_rate": [0.01, 0.1],
            },
        },
    }

    return models


def build_grid(model, params, is_pipeline=False):
    logger.info(f"Building GridSearchCV for {model.__class__.__name__}")

    # If using Pipeline, prefix parameters with the model step name
    if is_pipeline:
        params = {f"model__{key}": value for key, value in params.items()}

    return GridSearchCV(
        estimator=model,
        param_grid=params,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )
