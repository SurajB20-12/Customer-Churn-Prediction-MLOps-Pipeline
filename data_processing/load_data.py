import pandas as pd
import numpy as np
from utils.config import RAW_DATA_PATH
from utils.logger import get_logger

logger = get_logger(__name__)


def load_data():
    try:
        logger.info("Loading data...")
        df = pd.read_csv(RAW_DATA_PATH)
        logger.info(f"Data loaded successfully with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error while loading data:{e}")
        raise e
