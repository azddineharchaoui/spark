import os
from dotenv import load_dotenv

load_dotenv()

SPARK_APP_NAME = "BankAttritionAnalysis"
SPARK_MASTER = "local[*]"

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
MONGODB_DATABASE = "bank_attrition"
MONGODB_COLLECTION = "processed_data"

DATA_RAW_PATH = "data/raw/dataset.csv"
DATA_PROCESSED_PATH = "data/processed/"
MODEL_PATH = "models/trained_model/"

RANDOM_SEED = 42
TEST_SIZE = 0.2
