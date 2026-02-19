from pipeline.training_pipeline import train_pipeline
from data_processing.feature_engineering import preprocess_data
from data_processing.load_data import load_data
from utils.logger import get_logger

logger = get_logger(__name__)

if __name__ == "__main__":

    df = load_data()

    processed_df = preprocess_data(df)
    train_pipeline(processed_df)

    logger.info("Training pipeline completed successfully.")
