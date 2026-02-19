import pandas as pd
from utils.logger import get_logger

logger = get_logger(__name__)


def preprocess_data(df):
    try:
        logger.info("Starting data preprocessing...")

        df["Age"] = df["Age"].astype("int64")
        df["Num_Sites"] = df["Num_Sites"].astype("int64")

        df = df.drop(
            columns=["Names", "Onboard_date", "Location", "Company"], errors="ignore"
        )

        logger.info("Basic preprocessing completed successfully.")

        return df

    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise e
