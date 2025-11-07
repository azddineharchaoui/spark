from src.spark_utils import initialize_spark, check_spark_version
from src.data_loader import load_data, display_schema, display_sample, get_data_shape
# from src.eda import eda_summary, grouped_analysis, missing_values_analysis, outliers_analysis
from src.preprocess import preprocess_pipeline
# from src.ml_pipeline import prepare_features, split_data, train_random_forest, train_logistic_regression, evaluate_model
# from src.evaluate import calculate_metrics, confusion_matrix, print_evaluation_report
# from src.deploy import save_model, export_model_metrics
from src.config import DATA_RAW_PATH

def main():
    print("\n" + "="*60)
    print("BANK ATTRITION PREDICTION - PYSPARK ML PIPELINE")
    print("="*60)
    
    spark = initialize_spark()
    check_spark_version(spark)
    
    print("\n" + "="*60)
    print("STEP 2: LOADING DATA")
    print("="*60)
    try:
        df = load_data(spark, DATA_RAW_PATH)
        get_data_shape(df)
        display_schema(df)
        display_sample(df, 5)
    except FileNotFoundError:
        print(f"ERROR: Data file not found at {DATA_RAW_PATH}")
        print("Please ensure the CSV file is in the data/raw/ directory")
        spark.stop()
        return
    
    print("\n" + "="*60)
    print("STEP 3: EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*60)
    eda_summary(df)
    
    print("\n" + "="*60)
    print("STEP 4: DATA PREPROCESSING")
    print("="*60)
    df_processed = preprocess_pipeline(df, target_column="Exited")
    print("Data preprocessing completed")
    get_data_shape(df_processed)
    
    print("\n" + "="*60)
    print("STEP 5: FEATURE ENGINEERING")
    print("="*60)
    df_features = prepare_features(df_processed, target_column="Exited")
    print("Features prepared")
    
    print("\n" + "="*60)
    print("STEP 6: DATA SPLITTING")
    print("="*60)
    train_df, test_df = split_data(df_features, test_size=0.2)
    print(f"Train set size: {train_df.count()}")
    print(f"Test set size: {test_df.count()}")
    
    print("\n" + "="*60)
    print("STEP 7: MODEL TRAINING")
    print("="*60)
    
    print("\nTraining Random Forest Model...")
    rf_model = train_random_forest(train_df, target_column="Exited", num_trees=100)
    print("Random Forest model trained")
    
    print("\nTraining Logistic Regression Model...")
    lr_model = train_logistic_regression(train_df, target_column="Exited")
    print("Logistic Regression model trained")
    
    print("\n" + "="*60)
    print("STEP 8: MODEL EVALUATION")
    print("="*60)
    
    print("\nRandom Forest Evaluation:")
    rf_predictions = rf_model.transform(test_df)
    rf_metrics = calculate_metrics(rf_predictions, target_column="Exited")
    rf_cm = confusion_matrix(rf_predictions, target_column="Exited")
    print_evaluation_report(rf_metrics, rf_cm)
    
    print("\nLogistic Regression Evaluation:")
    lr_predictions = lr_model.transform(test_df)
    lr_metrics = calculate_metrics(lr_predictions, target_column="Exited")
    lr_cm = confusion_matrix(lr_predictions, target_column="Exited")
    print_evaluation_report(lr_metrics, lr_cm)
    
    print("\n" + "="*60)
    print("STEP 9: MODEL DEPLOYMENT")
    print("="*60)
    save_model(rf_model, "random_forest_model")
    export_model_metrics(rf_metrics, "rf_model_metrics.txt")
    
    save_model(lr_model, "logistic_regression_model")
    export_model_metrics(lr_metrics, "lr_model_metrics.txt")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)
    
    spark.stop()

if __name__ == "__main__":
    main()
