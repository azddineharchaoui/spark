from pyspark.sql import SparkSession
from src.config import DATA_RAW_PATH

def load_data(spark, path=DATA_RAW_PATH):
    df = spark.read.csv(path, header=True, inferSchema=True)
    return df

def display_schema(df):
    print("\n=== DataFrame Schema ===")
    df.printSchema()

def display_sample(df, n=5):
    print(f"\n=== First {n} Rows ===")
    df.show(n, truncate=False)

def get_data_shape(df):
    row_count = df.count()
    col_count = len(df.columns)
    print(f"\nDataFrame Shape: ({row_count}, {col_count})")
    return row_count, col_count
