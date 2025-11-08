import streamlit as st
import pandas as pd
import findspark
findspark.init()
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
import time

# --- Configuration ---
MODEL_PATH = "models/spark_lr_pipeline_model"
APP_TITLE = "Prédiction d'Attrition Bancaire"

# --- Initialisation de Spark ---
@st.cache_resource
def get_spark_session():
    print("Initialisation de SparkSession pour Streamlit...")
    try:
        spark = (
            SparkSession.builder
            .appName("BankAttritionInferenceApp")
            .master("local[*]")
            .config("spark.driver.memory", "2g")
            .getOrCreate()
        )
        return spark
    except Exception as e:
        st.error(f"Erreur lors de l'initialisation de Spark : {e}")
        st.error("Veuillez vérifier votre installation Java et Spark.")
        return None

# --- Chargement du Modèle (une seule fois) ---
@st.cache_resource
def load_model(path):
    print(f"Chargement du modèle depuis {path}...")
    try:
        model = PipelineModel.load(path)
        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle depuis {path}.")
        st.error(f"Détails : {e}")
        st.error("Assurez-vous d'avoir exécuté 'training_pipeline.py' d'abord.")
        return None

# --- Interface Streamlit ---
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.markdown(
    "Cette application utilise un modèle de Machine Learning (PySpark MLlib) "
    "pour prédire si un client est susceptible de quitter la banque (attrition)."
)

# Initialiser Spark et charger le modele
spark = get_spark_session()
model = load_model(MODEL_PATH)

if spark is None or model is None:
    st.error("L'application n'a pas pu démarrer. Vérifiez les logs.")
else:
    # --- Formulaire de saisie ---
    st.sidebar.header("Saisir les informations du client")
    
    col1, col2 = st.sidebar.columns(2)

    with col1:
        credit_score = st.number_input("Score de Crédit", min_value=300, max_value=850, value=650)
        geography = st.selectbox("Pays", ("France", "Spain", "Germany"))
        gender = st.selectbox("Genre", ("Male", "Female"))
        age = st.number_input("Age", min_value=18, max_value=100, value=40)
        tenure = st.number_input("Ancienneté (années)", min_value=0, max_value=10, value=5)

    with col2:
        balance = st.number_input("Solde du compte", min_value=0.0, value=50000.0, format="%.2f")
        num_of_products = st.number_input("Nombre de produits", min_value=1, max_value=4, value=1)
        has_cr_card = st.radio("Possède une carte de crédit ?", (1, 0), format_func=lambda x: "Oui" if x == 1 else "Non")
        is_active_member = st.radio("Membre actif ?", (1, 0), format_func=lambda x: "Oui" if x == 1 else "Non")
        estimated_salary = st.number_input("Salaire estimé (€)", min_value=0.0, value=100000.0, format="%.2f")

    # Bouton de prediction
    if st.sidebar.button("Prédire l'attrition", type="primary"):
        
        input_data = {
            "CreditScore": [credit_score],
            "Geography": [geography],
            "Gender": [gender],
            "Age": [age],
            "Tenure": [tenure],
            "Balance": [balance],
            "NumOfProducts": [num_of_products],
            "HasCrCard": [has_cr_card],
            "IsActiveMember": [is_active_member],
            "EstimatedSalary": [estimated_salary],
        }
        
        input_data["label"] = [0] 

        try:
            # Conversion en DataFrame Spark
            input_df_spark = spark.createDataFrame(pd.DataFrame(input_data))

            # 2. Appliquer le pipeline (Transformation + Prédiction)
            with st.spinner("Prédiction en cours..."):
                time.sleep(1) # Petite pause pour la démo
                prediction_result = model.transform(input_df_spark)
            
            # 3. Récupérer les résultats
            result_row = prediction_result.select("probability", "prediction").first()
            
            probability = result_row["probability"][1]  # Probabilité d'attrition (classe 1)
            prediction = result_row["prediction"]      # Classe prédite (0 ou 1)

            st.subheader("Résultat de la Prédiction")
            
            if prediction == 1:
                st.error(f"**Risque d'attrition ÉLEVÉ (Prédiction : OUI)**")
            else:
                st.success(f"**Risque d'attrition FAIBLE (Prédiction : NON)**")
            
            st.metric(
                label="Probabilité d'attrition (Classe 1)", 
                value=f"{probability * 100:.2f} %"
            )
            
            st.info("Cette prédiction est basée sur le modèle ML entraîné. Elle indique la probabilité que le client quitte la banque.")

            with st.expander("Détails de la prédiction (données brutes)"):
                st.dataframe(prediction_result.toPandas())

        except Exception as e:
            st.error(f"Une erreur est survenue lors de la prédiction : {e}")
            st.error("Vérifiez que les données d'entrée sont correctes.")