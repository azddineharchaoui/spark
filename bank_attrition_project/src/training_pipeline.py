import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, isnan, split
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler, Imputer, StandardScaler
)
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import pymongo
import pandas as pd
import sys

# --- Configuration des chemins  ---
RAW_DATA_PATH = "data/raw/dataset-spark.csv"
MODEL_OUTPUT_PATH = "models/spark_lr_pipeline_model"
PROCESSED_DATA_PATH = "data/processed/processed_data.parquet"

# --- Configuration MongoDB ---
MONGO_URI = "mongodb://localhost:27017/"
MONGO_DB = "bank_attrition_db"
MONGO_COLLECTION = "processed_customers"
SAVE_TO_MONGO = False  # Mettre à True pour activer la sauvegarde MongoDB

def create_spark_session():
    """Étape 1 : Configuration et Initialisation de Spark"""
    print("Étape 1 : Initialisation de SparkSession...")
    spark = (
        SparkSession.builder.appName("BankAttritionPrediction")
        .master("local[*]")  # Utiliser tous les cœurs disponibles localement
        .config("spark.driver.memory", "4g")
        .config("spark.mongodb.output.uri", f"{MONGO_URI}{MONGO_DB}.{MONGO_COLLECTION}")
        .getOrCreate()
    )
    print(f"Spark version: {spark.version}")
    return spark

def load_data(spark, path):
    """Étape 2 : Chargement des Données"""
    print(f"\nÉtape 2 : Chargement des données depuis {path}...")
    df = (
        spark.read.csv(
            path,
            header=True,
            inferSchema=True  # Infère automatiquement les types de données
        )
    )
    
    print("Schéma des données :")
    df.printSchema()
    print("Aperçu des données (5 lignes) :")
    df.show(5)
    
    # Renommer la colonne cible pour plus de clarté
    if "Exited" in df.columns:
        df = df.withColumnRenamed("Exited", "label")
        
    return df

def exploratory_data_analysis(df):
    """Étape 3 : Analyse Exploratoire des Données (EDA)"""
    print("\nÉtape 3 : Analyse Exploratoire des Données (EDA)...")
    
    print("Statistiques descriptives :")
    df.describe().show()
    
    print("Comptage des valeurs manquantes par colonne :")
    df.select(
        [
            count(when(isnan(c) | col(c).isNull(), c)).alias(c)
            for c in df.columns
        ]
    ).show()
    
    print("Distribution de la variable cible 'label' (Attrition) :")
    df.groupBy("label").count().show()
    # Note : Cela montrera le déséquilibre des classes

    print("Analyse groupée (Age moyen par Géographie et Attrition) :")
    df.groupBy("Geography", "label").avg("Age", "Balance").show()

    # Détection simple d'outliers (exemple sur 'CreditScore')
    # quantiles = df.stat.approxQuantile("CreditScore", [0.01, 0.99], 0.0)
    # print(f"Quantiles 1% et 99% pour CreditScore : {quantiles}")
    
    return df

def preprocess_data(df):
    """Étape 4 : Prétraitement des Données"""
    print("\nÉtape 4 : Prétraitement et Feature Engineering...")

    # Suppression des colonnes non pertinentes
    cols_to_drop = ["RowNumber", "CustomerId", "Surname"]
    df = df.drop(*cols_to_drop)
    
    # Identification des types de colonnes
    categorical_cols = ["Geography", "Gender"]
    # Toutes les autres colonnes sauf 'label' sont numériques
    numerical_cols = [
        c
        for c in df.columns
        if c not in categorical_cols + ["label"]
    ]
    
    print(f"Colonnes catégorielles : {categorical_cols}")
    print(f"Colonnes numériques : {numerical_cols}")

    # --- Pipeline de Prétraitement ---
    # Nous allons créer un pipeline MLlib pour le prétraitement
    # Cela garantit que les mêmes transformations sont appliquées
    # lors de l'inférence.
    
    stages = []
    
    # 1. Imputation des valeurs manquantes (Exemple sur 'CreditScore' si nécessaire)
    # Pour cet exemple, nous supposons qu'il n'y a pas de NA,
    # mais voici comment on le ferait :
    # imputer = Imputer(
    #     inputCols=["CreditScore"], 
    #     outputCols=["CreditScore_imputed"], 
    #     strategy="median"
    # )
    # stages.append(imputer)
    # numerical_cols = ["CreditScore_imputed"] + [c for c in numerical_cols if c != "CreditScore"]

    # 2. Encodage des variables catégorielles (StringIndexer + OneHotEncoder)
    # 'handleInvalid="keep"' gère les nouvelles catégories à l'inférence
    for cat_col in categorical_cols:
        string_indexer = StringIndexer(
            inputCol=cat_col, 
            outputCol=cat_col + "_index", 
            handleInvalid="keep"
        )
        one_hot_encoder = OneHotEncoder(
            inputCols=[string_indexer.getOutputCol()],
            outputCols=[cat_col + "_ohe"]
        )
        stages += [string_indexer, one_hot_encoder]
        
    # 3. Assemblage des features numériques et encodées
    feature_cols = numerical_cols + [c + "_ohe" for c in categorical_cols]
    
    vector_assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="unscaled_features",
        handleInvalid="keep"
    )
    stages.append(vector_assembler)
    
    # 4. Normalisation des features
    scaler = StandardScaler(
        inputCol="unscaled_features",
        outputCol="features"
        # withStd=True, withMean=False # Configuration par défaut
    )
    stages.append(scaler)
    
    # Créer le pipeline de transformation
    transform_pipeline = Pipeline(stages=stages)
    
    # Appliquer le pipeline de transformation
    print("Ajustement du pipeline de transformation...")
    transform_model = transform_pipeline.fit(df)
    
    print("Transformation des données...")
    processed_df = transform_model.transform(df)
    
    # Mettre en cache le DataFrame transformé
    processed_df.cache()
    
    print("Données prétraitées (sélection de 'label' et 'features') :")
    processed_df.select("label", "features").show(5, truncate=False)
    
    return processed_df, transform_model

def store_in_mongodb(df_to_store):
    """Étape 5 : Stockage Intermédiaire dans MongoDB"""
    if not SAVE_TO_MONGO:
        print("\nÉtape 5 : Stockage MongoDB ignoré (SAVE_TO_MONGO=False).")
        return

    print("\nÉtape 5 : Stockage des données prétraitées dans MongoDB...")
    
    try:
        # Connexion au client MongoDB
        client = pymongo.MongoClient(MONGO_URI)
        db = client[MONGO_DB]
        collection = db[MONGO_COLLECTION]
        
        # Vider la collection existante
        collection.delete_many({})
        
        # Conversion en Pandas (pour les petits/moyens datasets)
        # ATTENTION : Ne pas faire cela sur de très gros volumes de données.
        # Pour le Big Data, utiliser le connecteur Spark-MongoDB.
        print("Conversion en Pandas (peut être long)...")
        pandas_df = df_to_store.toPandas()
        
        print(f"Conversion en dictionnaires et insertion de {len(pandas_df)} documents...")
        records = pandas_df.to_dict("records")
        
        # Insertion en masse
        collection.insert_many(records)
        
        print(f"Stockage dans MongoDB ({MONGO_DB}.{MONGO_COLLECTION}) terminé.")
        client.close()
        
    except Exception as e:
        print(f"Erreur lors du stockage dans MongoDB : {e}", file=sys.stderr)
        print("Vérifiez si votre instance MongoDB est en cours d'exécution.")

def save_processed_data(df, path):
    """Sauvegarde les données traitées au format Parquet pour une réutilisation."""
    print(f"\nSauvegarde des données traitées au format Parquet : {path}")
    (
        df.select("label", "features")
        .write.mode("overwrite")
        .parquet(path)
    )

def handle_class_imbalance(df):
    """Gère le déséquilibre de classes en ajoutant une colonne de poids."""
    print("\nÉtape 6b : Gestion du déséquilibre de classes (Pondération)...")
    
    # Compter les instances de chaque classe
    balance_df = df.groupBy("label").count().collect()
    count_0 = next(row['count'] for row in balance_df if row['label'] == 0)
    count_1 = next(row['count'] for row in balance_df if row['label'] == 1)
    total_count = count_0 + count_1
    
    print(f"Classe 0 (Non-Attr.): {count_0}")
    print(f"Classe 1 (Attrition) : {count_1}")

    # Calculer les poids : Poids = Total / (N_Classes * N_Instances)
    # Poids pour la classe 1 (minoritaire)
    weight_1 = total_count / (2.0 * count_1)
    # Poids pour la classe 0 (majoritaire)
    weight_0 = total_count / (2.0 * count_0)

    print(f"Poids Classe 0 : {weight_0:.2f}")
    print(f"Poids Classe 1 : {weight_1:.2f}")
    
    # Ajouter la colonne de poids au DataFrame
    df_weighted = df.withColumn(
        "weightCol",
        when(col("label") == 1, weight_1).otherwise(weight_0)
    )
    
    return df_weighted

def train_evaluate_model(processed_df, transform_model):
    """Étapes 6, 7 et 8 : Construction du Pipeline ML, Entraînement et Évaluation"""
    print("\nÉtape 6 : Construction du Pipeline de Machine Learning...")

    # Gestion du déséquilibre
    df_weighted = handle_class_imbalance(processed_df)

    # Séparation des données
    (train_df, test_df) = df_weighted.randomSplit([0.8, 0.2], seed=42)
    print(f"Taille Training Set : {train_df.count()}")
    print(f"Taille Test Set : {test_df.count()}")

    # Choix du modèle (Régression Logistique)
    # Nous utilisons 'weightCol' pour gérer le déséquilibre
    lr = LogisticRegression(
        featuresCol="features", 
        labelCol="label", 
        weightCol="weightCol",
        elasticNetParam=0.5 # Combinaison L1/L2
    )
    
    # Le pipeline de transformation est déjà appliqué.
    # Pour le déploiement, nous voulons sauvegarder l'ensemble du pipeline :
    # Transformations (de l'étape 4) + Modèle (lr)
    
    # Nous recréons le pipeline de transformation (stages)
    # et y ajoutons le modèle 'lr'
    
    all_stages = transform_model.stages + [lr]
    ml_pipeline = Pipeline(stages=all_stages)

    print("\nÉtape 7 : Entraînement et Validation Croisée...")
    
    # Grille d'hyperparamètres
    paramGrid = (
        ParamGridBuilder()
        .addGrid(lr.regParam, [0.01, 0.1, 0.5])
        .addGrid(lr.aggregationDepth, [2, 5])
        .build()
    )
    
    # Évaluateur (AUC-ROC)
    evaluator = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )
    
    # Validation croisée
    crossval = CrossValidator(
        estimator=ml_pipeline, # Nous validons le pipeline complet
        estimatorParamMaps=paramGrid,
        evaluator=evaluator,
        numFolds=3  # 3 folds pour la rapidité (5 ou 10 sont courants)
    )
    
    print("Lancement de la validation croisée (peut prendre du temps)...")
    cv_model = crossval.fit(train_df)
    
    # Meilleur modèle
    best_pipeline_model = cv_model.bestModel
    print("Validation croisée terminée.")

    print("\nÉtape 8 : Évaluation du Modèle...")
    
    # Prédictions sur l'ensemble de test
    predictions = best_pipeline_model.transform(test_df)
    
    print("Aperçu des prédictions :")
    predictions.select("label", "probability", "prediction").show(10)
    
    # Calcul des métriques
    auc_roc = evaluator.evaluate(predictions)
    print(f"AUC-ROC sur le Test Set : {auc_roc:.4f}")
    
    # Autres métriques (Accuracy, Precision, Recall, F1)
    multi_evaluator = MulticlassClassificationEvaluator(
        labelCol="label", 
        predictionCol="prediction"
    )
    
    accuracy = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "accuracy"})
    precision = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "weightedPrecision"})
    recall = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "weightedRecall"})
    f1 = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "f1"})
    
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision (pondérée) : {precision:.4f}")
    print(f"Recall (pondéré) : {recall:.4f}")
    print(f"F1-Score (pondéré) : {f1:.4f}")

    print("Matrice de confusion :")
    predictions.groupBy("label", "prediction").count().show()
    
    return best_pipeline_model

def save_model(model):
    """Étape 9 : Sauvegarde du Modèle"""
    print(f"\nÉtape 9 : Sauvegarde du pipeline ML complet dans {MODEL_OUTPUT_PATH}...")
    try:
        model.write().overwrite().save(MODEL_OUTPUT_PATH)
        print("Modèle sauvegardé avec succès.")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde du modèle : {e}")

def main():
    spark = create_spark_session()
    
    df = load_data(spark, RAW_DATA_PATH)
    
    df_eda = exploratory_data_analysis(df)
    
    # Nous passons df_eda, qui est le même que df mais après analyse
    (processed_df, transform_model) = preprocess_data(df_eda)

    # Sauvegarde Parquet (optionnelle, pour réutilisation rapide)
    # save_processed_data(processed_df, PROCESSED_DATA_PATH)
    
    # Stockage MongoDB (optionnel)
    # Nous sélectionnons les colonnes originales + la cible
    store_in_mongodb(df_eda) 
    
    # Entraînement et Évaluation
    # Nous passons processed_df (qui contient 'label' et 'features')
    # ET le transform_model (pour reconstruire le pipeline complet)
    best_model = train_evaluate_model(processed_df, transform_model)
    
    # Sauvegarde du modèle
    save_model(best_model)
    
    print("\nPipeline d'entraînement terminé.")
    spark.stop()

if __name__ == "__main__":
    main()