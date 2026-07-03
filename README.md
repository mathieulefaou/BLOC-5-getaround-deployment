# 🚗 GetAround - Pricing API & Business Intelligence Dashboard

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python)](https://www.python.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow?style=for-the-badge)](https://huggingface.co/spaces)

Ce projet propose une solution complète de Data Science et MLOps pour optimiser la tarification de location quotidienne des véhicules sur la plateforme **GetAround**, tout en analysant l'impact des retards de restitution de la flotte.

L'architecture est séparée en deux briques indépendantes :
1. **Une API de prédiction (`main.py`)** embarquant un modèle de Machine Learning et un simulateur interactif.
2. **Un Dashboard d'analyse (`app.py`)** destiné aux équipes produit et décisionnaires.

---

## 🚀 Fonctionnalités Clés

* **Simulateur Interactif (Page d'accueil API) :** Une interface web HTML/JS intégrée permettant de choisir les options d'une voiture et de calculer instantanément le prix suggéré en envoyant une requête en temps réel.
* **API de Prédiction en Batch :** Un endpoint `/predict` capable de traiter les prédictions pour un ou plusieurs véhicules simultanément (format de liste de listes).
* **Business Dashboard (`app.py`) :** Un tableau de bord décisionnel (Streamlit) pour analyser les retards entre deux locations, aider à choisir la durée idéale du seuil de tolérance (threshold) et mesurer l'impact sur le business de GetAround.
* **Documentation Interactive :** Une interface Swagger UI pour tester techniquement les routes.

---

## 📊 Performances du Modèle

Le modèle de Machine Learning a été entraîné sur un historique de **4 843 véhicules** à l'aide d'un algorithme de **Gradient Boosting Regressor** (scikit-learn) configuré avec 200 estimateurs et une profondeur maximale de 5.

| Métrique | Régression Linéaire (Baseline) | Gradient Boosting (Retenu) |
| :--- | :---: | :---: |
| **R² Score** | 0.6937 | **0.7504** |
| **MAE** | 12.12 EUR | **10.29 EUR** |
| **RMSE** | 17.96 EUR | **16.22 EUR** |

### Importance des Caractéristiques (Top 3)
1. **Puissance moteur :** 46%
2. **Kilométrage :** 27%
3. **Système GPS :** 5%

---

## 🛠️ Structure des Fichiers

```text
├── main.py              # Code de l'API FastAPI (Moteur de calcul & simulateur)
├── app.py               # Code du Dashboard Streamlit (Visualisation & BI)
├── model.joblib         # Pipeline scikit-learn entraîné et sérialisé
├── feature_info.json    # Métadonnées du modèle, colonnes et performances
├── requirements.txt     # Dépendances Python requises
└── Dockerfile           # Configuration pour le déploiement conteneurisé de l'API
