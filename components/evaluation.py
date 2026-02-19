from sklearn.metrics import accuracy_score, classification_report
from utils.logger import get_logger

from sklearn.metrics import accuracy_score, classification_report
from utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)

    logger.info(f"Model Accuracy: {accuracy}")

    return accuracy, report
