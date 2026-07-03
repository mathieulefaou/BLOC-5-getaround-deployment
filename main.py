from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import pandas as pd

# 1. On initialise l'API en désactivant TOTALEMENT la doc automatique native
app = FastAPI(
    title="GetAround - Car Pricing API",
    description="API de prédiction du prix de location quotidien.",
    version="1.0.0",
    docs_url=None, 
    redoc_url=None
)

# Chargement du modèle de Machine Learning
model = joblib.load("model.joblib")

# Définition du format des données attendues (Input)
class CarFeatures(BaseModel):
    model_key: str
    mileage: int
    engine_power: int
    fuel: str
    paint_color: str
    car_type: str
    private_parking_available: bool
    has_gps: bool
    has_air_conditioning: bool
    automatic_car: bool
    has_getaround_connect: bool
    has_leather_seats: bool

@app.get("/", response_class=HTMLResponse)
async def root():
    # Page d'accueil stylisée aux couleurs du Dashboard
    return """
    <html>
        <head>
            <title>GetAround API</title>
            <style>
                body { background-color: #0b0914; color: #ffffff; font-family: sans-serif; text-align: center; padding-top: 100px; }
                h1 { color: #9d4edd; font-size: 2.5rem; }
                p { color: #c3b9d6; font-size: 1.2rem; }
                a { display: inline-block; margin-top: 30px; color: #ffffff; background: #9d4edd; padding: 12px 24px; border-radius: 20px; text-decoration: none; font-weight: bold; transition: 0.2s; }
                a:hover { background: #7b2cbf; }
            </style>
        </head>
        <body>
            <h1>🚀 API GetAround Opérationnelle</h1>
            <p>Le moteur de prédiction du Machine Learning est en ligne et configuré.</p>
            <a href="/docs">Accéder à la Documentation Interactive</a>
        </body>
    </html>
    """

# 2. On reconstruit entièrement la page /docs en HTML/CSS pur pour forcer le Dark/Violet Mode
@app.get("/docs", response_class=HTMLResponse, include_in_schema=False)
async def custom_swagger_ui():
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
    <link rel="stylesheet" type="text/css" href="https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/5.11.0/swagger-ui.css" >
    <title>GetAround API - Documentation</title>
    <style>
        /* Force le fond sombre et violet */
        html, body, #swagger-ui {{ background-color: #0b0914 !important; color: #ffffff !important; margin: 0; padding: 0; }}
        .swagger-ui .topbar {{ display: none !important; }} /* Cache la barre noire */
        .swagger-ui .info .title, .swagger-ui .info h2, .swagger-ui .opblock-tag {{ color: #9d4edd !important; }}
        .swagger-ui .info p, .swagger-ui .opblock .opblock-summary-description, .swagger-ui .tab li button {{ color: #c3b9d6 !important; }}
        .swagger-ui .opblock.opblock-post {{ background: #141126 !important; border-color: #2a2240 !important; }}
        .swagger-ui .opblock.opblock-post .opblock-summary {{ border-color: #2a2240 !important; }}
        .swagger-ui .opblock.opblock-post .opblock-summary-method {{ background: #9d4edd !important; border-radius: 20px !important; }}
        .swagger-ui .btn, .swagger-ui .opblock-description-wrapper p, .swagger-ui .prop-type, .swagger-ui .parameter__name {{ color: #ffffff !important; }}
        .swagger-ui .btn.execute {{ background-color: #9d4edd !important; border-color: #9d4edd !important; color: white !important; }}
        .swagger-ui .btn.execute:hover {{ background-color: #7b2cbf !important; }}
        .swagger-ui select, .swagger-ui input[type=text], .swagger-ui textarea {{ background: #141126 !important; color: #ffffff !important; border: 1px solid #2a2240 !important; }}
        .swagger-ui .responses-table, .swagger-ui .opblock-section-header, .swagger-ui .opblock-body {{ background: #141126 !important; color: #ffffff !important; }}
        .swagger-ui code {{ background: #2a2240 !important; color: #ffffff !important; }}
        .swagger-ui .model-box {{ background: #141126 !important; color: #ffffff !important; }}
    </style>
    </head>
    <body>
    <div id="swagger-ui"></div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/5.11.0/swagger-ui-bundle.js"> </script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/5.11.0/swagger-ui-standalone-preset.js"> </script>
    <script>
    window.onload = function() {{
      const ui = SwaggerUIBundle({{
        url: "{app.openapi_url}",
        dom_id: '#swagger-ui',
        presets: [
          SwaggerUIBundle.presets.apis,
          SwaggerUIStandalonePreset
        ],
        layout: "BaseLayout"
      }});
      window.ui = ui;
    }};
    </script>
    </body>
    </html>
    """

@app.post("/predict")
async def predict(features: CarFeatures):
    # Transformation des données d'entrée en DataFrame pour le modèle
    input_df = pd.DataFrame([features.dict()])
    
    # Prédiction
    prediction = model.predict(input_df)
    
    # Retour du résultat
    return {"prediction": float(prediction[0])}
