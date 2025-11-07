from pyspark.sql import SparkSession
from src.config import SPARK_APP_NAME, SPARK_MASTER

def initialize_spark():
    spark = SparkSession.builder \
        .appName(SPARK_APP_NAME) \
        .master(SPARK_MASTER) \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.driver.memory", "2g") \
        .config("spark.executor.memory", "2g") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    return spark

def check_spark_version(spark):
    version = spark.version
    print(f"Spark Version: {version}")
    return version
