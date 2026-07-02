import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# 1. Chargement des données
df = pd.read_csv("get_around_pricing_project.csv")

# Suppression de la colonne d'index inutile si elle existe
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# 2. Séparation des fonctionnalités (X) et de la cible (y)
X = df.drop(columns=["rental_price_per_day"])
y = df["rental_price_per_day"]

# On isole les colonnes de texte/catégorielles et les colonnes numériques/booléennes
categorical_features = ["model_key", "fuel", "paint_color", "car_type"]
numeric_features = [col for col in X.columns if col not in categorical_features]

# 3. Création du Préprocesseur (One-Hot Encoding pour le texte)
# handle_unknown='ignore' est CRUCIAL pour éviter que l'API ne plante si elle reçoit une marque inconnue
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='passthrough' # Garde les colonnes numériques et booléennes intactes
)

# 4. Création du Pipeline complet (Prétraitement + Modèle)
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Séparation Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Entraînement du Pipeline (il applique le One-Hot Encoding puis entraîne le RandomForest)
model_pipeline.fit(X_train, y_train)

# Sauvegarde du pipeline complet sous format joblib
# L'avantage du Pipeline est qu'il sauvegarde le OneHotEncoder ET le modèle ensemble !
joblib.dump(model_pipeline, "model.joblib")

# Optionnel : pour voir combien de colonnes ont été générées après le One-Hot Encoding
X_train_transformed = preprocessor.transform(X_train)
print(f"Entraînement terminé avec succès ! Nombre maximal de variables utilisées : {X_train_transformed.shape[1]}")