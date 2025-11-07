from pyspark.sql.functions import col, when, isnull
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline

def handle_missing_values(df):
    numeric_columns = [f.name for f in df.schema.fields 
                       if str(f.dataType) in ['IntegerType', 'DoubleType']]
    
    for col_name in numeric_columns:
        df = df.fillna({col_name: df.agg({col_name: "avg"}).collect()[0][0]})
    
    categorical_columns = [f.name for f in df.schema.fields 
                           if str(f.dataType) in ['StringType']]
    
    for col_name in categorical_columns:
        df = df.fillna({col_name: "Unknown"})
    
    return df

def remove_duplicates(df):
    return df.dropDuplicates()

def remove_outliers(df, column, lower_quantile=0.01, upper_quantile=0.99):
    quantiles = df.approxQuantile(column, [lower_quantile, upper_quantile], 0.05)
    lower_bound, upper_bound = quantiles[0], quantiles[1]
    return df.filter((col(column) >= lower_bound) & (col(column) <= upper_bound))

def encode_categorical(df, categorical_columns):
    indexers = [StringIndexer(inputCol=col_name, outputCol=f"{col_name}_indexed") 
                for col_name in categorical_columns]
    
    pipeline = Pipeline(stages=indexers)
    df_indexed = pipeline.fit(df).transform(df)
    
    return df_indexed

def scale_numeric_features(df, numeric_columns):
    from pyspark.ml.feature import StandardScaler, VectorAssembler
    
    assembler = VectorAssembler(inputCols=numeric_columns, outputCol="features_temp")
    df = assembler.transform(df)
    
    scaler = StandardScaler(inputCol="features_temp", outputCol="scaled_features")
    df = scaler.fit(df).transform(df)
    
    return df

def preprocess_pipeline(df, target_column="Exited"):
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    
    categorical_cols = [f.name for f in df.schema.fields 
                        if str(f.dataType) == 'StringType']
    
    if categorical_cols:
        df = encode_categorical(df, categorical_cols)
    
    return df
