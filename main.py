import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List

app = FastAPI(
    title="GetAround Pricing API",
    description="API de prédiction du prix journalier de location de voiture",
    version="1.0.0",
    docs_url="/swagger",
    redoc_url=None
)

# Chargement du modèle entraîné (pipeline sklearn)
model = joblib.load("model.joblib")

# Colonnes attendues par le modèle (13 features)
FEATURE_COLUMNS = [
    "model_key", "mileage", "engine_power", "fuel", "paint_color",
    "car_type", "private_parking_available", "has_gps",
    "has_air_conditioning", "automatic_car", "has_getaround_connect",
    "has_speed_regulator", "winter_tires"
]


class PredictionInput(BaseModel):
    """Schéma d'un véhicule individuel (non utilisé directement)."""

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
    has_speed_regulator: bool
    winter_tires: bool


class PredictionRequest(BaseModel):
    """Corps de la requête : liste de listes (1 sous-liste = 1 véhicule)."""

    input: List[List]


class PredictionResponse(BaseModel):
    """Réponse : liste des prix prédits en EUR/jour."""

    prediction: List[float]


# Style CSS commun entre la page d'accueil et la documentation
COMMON_STYLE = """
<style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
        font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
        background: #0f1117; color: #e0e0e0; line-height: 1.7;
    }
    .header {
        background: linear-gradient(135deg, #1a1229 0%, #241a3d 50%, #3b1f66 100%);
        padding: 48px 24px; text-align: center;
        border-bottom: 3px solid #9d7bf5;
    }
    .header h1 { font-size: 2.2rem; font-weight: 700; color: #fff; margin-bottom: 8px; }
    .header .subtitle { color: #8899aa; font-size: 1.1rem; }
    .header .badge {
        display: inline-block; background: #9d7bf5; color: #0f1117;
        padding: 4px 14px; border-radius: 20px;
        font-size: 0.8rem; font-weight: 600; margin-top: 12px;
    }
    .container { max-width: 960px; margin: 0 auto; padding: 32px 24px; }
    .card {
        background: #1a1d29; border: 1px solid #2a2d3a;
        border-radius: 12px; padding: 28px; margin-bottom: 24px;
        transition: border-color 0.2s;
    }
    .card:hover { border-color: #9d7bf5; }
    .card h2 {
        color: #9d7bf5; font-size: 1.3rem; margin-bottom: 12px;
        display: flex; align-items: center; gap: 10px;
    }
    .card p { color: #aab; }
    a { color: #9d7bf5; text-decoration: none; }
    a:hover { text-decoration: underline; }
    .btn {
        display: inline-block; padding: 12px 28px; border-radius: 8px;
        font-weight: 600; font-size: 1rem; transition: all 0.2s;
        text-decoration: none !important;
    }
    .btn-primary { background: #9d7bf5; color: #0f1117; }
    .btn-primary:hover { background: #8a63f0; transform: translateY(-1px); }
    .btn-outline { border: 1px solid #9d7bf5; color: #9d7bf5; background: transparent; }
    .btn-outline:hover { background: rgba(157,123,245,0.12); }
    .metrics {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 16px; margin: 20px 0;
    }
    .metric {
        background: #12141d; border: 1px solid #2a2d3a;
        border-radius: 10px; padding: 20px; text-align: center;
    }
    .metric .value { font-size: 1.8rem; font-weight: 700; color: #9d7bf5; }
    .metric .label { font-size: 0.85rem; color: #778; margin-top: 4px; }
    code {
        background: #12141d; color: #9d7bf5; padding: 3px 8px;
        border-radius: 4px;
        font-family: 'Fira Code', 'Consolas', monospace; font-size: 0.9em;
    }
    pre {
        background: #12141d; border: 1px solid #2a2d3a;
        border-radius: 8px; padding: 20px; overflow-x: auto;
        font-family: 'Fira Code', 'Consolas', monospace;
        font-size: 0.88rem; color: #c8d0d8; line-height: 1.5;
    }
    table {
        width: 100%; border-collapse: separate; border-spacing: 0;
        border-radius: 8px; overflow: hidden;
        border: 1px solid #2a2d3a; margin: 16px 0;
    }
    th {
        background: #1e2130; color: #9d7bf5; padding: 12px 16px;
        text-align: left; font-weight: 600; font-size: 0.85rem;
        text-transform: uppercase; letter-spacing: 0.5px;
    }
    td { padding: 10px 16px; border-top: 1px solid #2a2d3a; font-size: 0.92rem; }
    tr:hover td { background: rgba(157,123,245,0.05); }
    .method-badge {
        display: inline-block; padding: 4px 12px; border-radius: 4px;
        font-weight: 700; font-size: 0.8rem; font-family: monospace;
    }
    .post { background: rgba(73,204,144,0.15); color: #49cc91; }
    .get { background: rgba(157,123,245,0.15); color: #9d7bf5; }
    .endpoint { font-family: monospace; font-size: 1rem; color: #fff; margin-left: 8px; }
    .nav { display: flex; gap: 12px; justify-content: center; margin-top: 20px; }
    .footer {
        text-align: center; padding: 32px; color: #556;
        font-size: 0.85rem; border-top: 1px solid #1e2130; margin-top: 40px;
    }
    .form-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 16px; margin: 20px 0;
    }
    .field label {
        display: block; font-size: 0.8rem; color: #8899aa;
        text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px;
    }
    .field select, .field input[type="number"] {
        width: 100%; background: #12141d; border: 1px solid #2a2d3a;
        color: #e0e0e0; border-radius: 8px; padding: 10px 12px;
        font-size: 0.95rem; font-family: inherit;
    }
    .field select:focus, .field input[type="number"]:focus {
        outline: none; border-color: #9d7bf5;
    }
    .field .range-value { color: #9d7bf5; font-weight: 600; margin-left: 6px; }
    .checkbox-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 10px; margin: 16px 0;
    }
    .checkbox-item {
        display: flex; align-items: center; gap: 10px;
        background: #12141d; border: 1px solid #2a2d3a;
        border-radius: 8px; padding: 10px 14px; cursor: pointer;
    }
    .checkbox-item:hover { border-color: #9d7bf5; }
    .checkbox-item input { accent-color: #9d7bf5; width: 16px; height: 16px; }
    .checkbox-item span { font-size: 0.9rem; color: #ccd; }
    .btn-submit {
        background: #9d7bf5; color: #0f1117; border: none;
        padding: 14px 32px; border-radius: 8px; font-weight: 700;
        font-size: 1rem; cursor: pointer; transition: all 0.2s;
        font-family: inherit;
    }
    .btn-submit:hover { background: #8a63f0; transform: translateY(-1px); }
    .btn-submit:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
    .result-box {
        background: #12141d; border: 1px solid #2a2d3a; border-radius: 8px;
        padding: 20px; margin-top: 16px; font-family: monospace;
        font-size: 0.95rem; min-height: 24px;
    }
    .result-box.success { border-color: #49cc91; color: #49cc91; }
    .result-box.error { border-color: #ff6b6b; color: #ff6b6b; }
    .result-box.pending { color: #778; }
    details.faq-item {
        background: #12141d; border: 1px solid #2a2d3a; border-radius: 8px;
        margin-bottom: 10px; overflow: hidden;
    }
    details.faq-item summary {
        padding: 16px 20px; cursor: pointer; font-weight: 600;
        color: #fff; list-style: none; display: flex; align-items: center;
        justify-content: space-between; gap: 12px;
    }
    details.faq-item summary::-webkit-details-marker { display: none; }
    details.faq-item summary::after {
        content: '+'; color: #9d7bf5; font-size: 1.3rem; flex-shrink: 0;
    }
    details.faq-item[open] summary::after { content: '−'; }
    details.faq-item[open] summary { border-bottom: 1px solid #2a2d3a; }
    details.faq-item .faq-answer { padding: 16px 20px; color: #aab; font-size: 0.92rem; }
</style>
"""


BRANDS = [
    "Alfa Romeo", "Audi", "BMW", "Citroën", "Ferrari", "Fiat", "Ford", "Honda",
    "KIA Motors", "Lamborghini", "Lexus", "Maserati", "Mazda", "Mercedes",
    "Mini", "Mitsubishi", "Nissan", "Opel", "PGO", "Peugeot", "Porsche",
    "Renault", "SEAT", "Subaru", "Suzuki", "Toyota", "Volkswagen", "Yamaha"
]
CAR_TYPES = ["sedan", "suv", "hatchback", "estate", "convertible", "coupe", "subcompact", "van"]
FUELS = ["diesel", "petrol", "hybrid_petrol", "electro"]
COLORS = ["black", "white", "grey", "silver", "red", "blue", "green", "beige", "brown", "orange"]

BOOL_OPTIONS = [
    ("private_parking_available", "Place de parking privee ?"),
    ("has_gps", "GPS integre ?"),
    ("has_air_conditioning", "Climatisation ?"),
    ("automatic_car", "Boite automatique ?"),
    ("has_getaround_connect", "GetAround Connect ?"),
    ("has_speed_regulator", "Regulateur de vitesse ?"),
    ("winter_tires", "Pneus hiver ?"),
]

FAQ_ITEMS = [
    ("Pourquoi un Gradient Boosting et pas un reseau de neurones ?",
     "Dataset de 4 843 lignes, pas assez pour un reseau profond. Le Gradient Boosting gere tres bien "
     "les donnees tabulaires mixtes (numeriques et categorielles), reste interpretable grace a la "
     "feature importance, s'entraine en quelques secondes et se deploie dans un fichier joblib de "
     "moins de 2 Mo. Regression lineaire testee en baseline : R2 0.69. Gradient Boosting : R2 0.75 "
     "sur le test set."),
    ("Que se passe-t-il si j'envoie une marque inconnue, par exemple \"Tesla\" ?",
     "Le OneHotEncoder est configure avec handle_unknown='ignore' : toutes les variables one-hot de "
     "cette marque tombent a 0, le modele s'appuie alors uniquement sur les autres features "
     "(puissance, kilometrage, options). Pas de crash, pas d'erreur HTTP 500, juste une prediction "
     "legerement moins precise."),
    ("Comment le JSON est-il valide avant d'arriver au modele ?",
     "Pydantic verifie la structure a l'entree de FastAPI : cle input presente, liste de listes. Si "
     "le body est mal forme, FastAPI renvoie un HTTP 422 avec le detail de l'erreur. Ensuite, les "
     "valeurs arrivent dans une pipeline scikit-learn qui applique le preprocessing (StandardScaler, "
     "OneHotEncoder) puis predit."),
    ("Pourquoi le body est une liste de listes et pas un objet nomme ?",
     "Choix delibere : permet d'envoyer plusieurs vehicules en une seule requete (batch). Par exemple "
     "100 vehicules d'un coup, une seule aller-retour reseau, plus performant en production. Le "
     "compromis est la lisibilite : l'ordre des 13 features est critique, c'est pour ca que la page "
     "/docs detaille le tableau ordonne."),
    ("L'API est-elle securisee ?",
     "Cette instance est en mode demo : pas d'authentification, pas de rate limiting, HTTPS active "
     "par defaut via Hugging Face. Pour une mise en production reelle : ajouter une API key via "
     "fastapi.security, deployer derriere un API Gateway (AWS ou Cloudflare) avec quotas, et "
     "versionner les modeles via MLflow Model Registry plutot qu'un fichier joblib fige."),
    ("Comment le modele est-il versionne ?",
     "MLflow tracke chaque entrainement localement (dossier mlruns/ du notebook). Deux runs sont "
     "enregistres : linear_regression et gradient_boosting, avec parametres, metriques (R2, MAE, "
     "RMSE) et pipeline complet logge. Pour cette demo, le modele retenu est exporte via "
     "joblib.dump(pipeline, 'model.joblib') et charge au demarrage de l'API."),
    ("Pourquoi la couleur est dans le modele si son impact est quasi nul ?",
     "Le dataset la fournit, donc elle passe dans le pipeline. Le Gradient Boosting l'utilise comme "
     "signal faible (importance < 1%). La retirer aurait necessite une etape de feature selection "
     "explicite, non priorisee ici. En production : on auditerait le dataset, on retirerait cette "
     "feature et on aurait un modele plus simple, sans perte mesurable de performance."),
    ("Quelle latence typique pour une prediction ?",
     "Environ 150 a 300 ms pour un vehicule (reseau + serialisation + preprocessing + prediction), "
     "essentiellement la latence reseau vers Hugging Face. Le calcul ML pur prend moins de 5 ms. En "
     "batch, le cout marginal par vehicule supplementaire est negligeable."),
    ("Que renvoie chaque code HTTP ?",
     "200 OK : prediction reussie, corps = {\"prediction\": [...]}. 422 Unprocessable Entity : body "
     "JSON mal forme (cle input manquante, nombre de features incorrect, type invalide). 500 Internal "
     "Server Error : defaillance cote serveur (rare). Timeout : le Space Hugging Face dort, premier "
     "appel plus long (5 a 12 secondes) le temps du reveil."),
]


def _options_html(values, selected):
    return "\n".join(
        f'<option value="{v}"{" selected" if v == selected else ""}>{v}</option>'
        for v in values
    )


def _checkbox_grid_html():
    items = []
    for key, label in BOOL_OPTIONS:
        checked = " checked" if key in ("private_parking_available", "has_gps",
                                         "has_getaround_connect", "has_speed_regulator",
                                         "winter_tires") else ""
        items.append(f"""
            <label class="checkbox-item">
                <input type="checkbox" id="{key}"{checked}>
                <span>{label}</span>
            </label>""")
    return "\n".join(items)


def _faq_html():
    items = []
    for question, answer in FAQ_ITEMS:
        items.append(f"""
            <details class="faq-item">
                <summary>{question}</summary>
                <div class="faq-answer">{answer}</div>
            </details>""")
    return "\n".join(items)


@app.get("/", response_class=HTMLResponse)
def root():
    """Page d'accueil de l'API avec constructeur de requete interactif."""
    return f"""
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>GetAround Pricing API</title>
        {COMMON_STYLE}
    </head>
    <body>
        <div class="header">
            <h1>GetAround Pricing API</h1>
            <p class="subtitle">Prediction du prix journalier de location de voiture par Machine Learning</p>
            <span class="badge">v1.0.0 &mdash; Gradient Boosting</span>
            <div class="nav">
                <a href="/docs" class="btn btn-primary">Documentation</a>
                <a href="/swagger" class="btn btn-outline">Swagger UI</a>
            </div>
        </div>
        <div class="container">
            <div class="metrics">
                <div class="metric"><div class="value">0.75</div><div class="label">R2 Score</div></div>
                <div class="metric"><div class="value">10.29</div><div class="label">MAE (EUR)</div></div>
                <div class="metric"><div class="value">16.22</div><div class="label">RMSE (EUR)</div></div>
                <div class="metric"><div class="value">4 843</div><div class="label">Vehicules entraines</div></div>
            </div>

            <div class="card">
                <h2>Constructeur de requete interactif</h2>
                <p>Construisez votre requete avec les menus ci-dessous, observez le JSON genere en temps reel puis envoyez-le a l'endpoint <code>/predict</code>.</p>

                <div class="form-grid">
                    <div class="field">
                        <label for="model_key">Marque</label>
                        <select id="model_key">{_options_html(BRANDS, "Renault")}</select>
                    </div>
                    <div class="field">
                        <label for="car_type">Type de vehicule</label>
                        <select id="car_type">{_options_html(CAR_TYPES, "sedan")}</select>
                    </div>
                    <div class="field">
                        <label for="fuel">Carburant</label>
                        <select id="fuel">{_options_html(FUELS, "diesel")}</select>
                    </div>
                    <div class="field">
                        <label for="paint_color">Couleur</label>
                        <select id="paint_color">{_options_html(COLORS, "black")}</select>
                    </div>
                    <div class="field">
                        <label for="mileage">Kilometrage <span class="range-value" id="mileage_val">140 000 km</span></label>
                        <input type="number" id="mileage" value="140000" min="0" step="1000">
                    </div>
                    <div class="field">
                        <label for="engine_power">Puissance moteur <span class="range-value" id="engine_power_val">135 ch</span></label>
                        <input type="number" id="engine_power" value="135" min="0" step="5">
                    </div>
                </div>

                <p style="margin-top:20px; color:#8899aa; font-size:0.85rem; text-transform:uppercase; letter-spacing:0.5px;">Options du vehicule</p>
                <div class="checkbox-grid">{_checkbox_grid_html()}</div>

                <div style="margin-top:20px;">
                    <button class="btn-submit" id="submitBtn" onclick="sendPrediction()">Envoyer la requete</button>
                </div>

                <p style="margin-top:24px; color:#8899aa; font-size:0.85rem; text-transform:uppercase; letter-spacing:0.5px;">Requete POST /predict</p>
                <pre id="jsonPreview">{{}}</pre>
                <p style="color:#556; font-size:0.85rem; margin-top:4px;">Le JSON ci-dessus est celui qui sera envoye au serveur avec l'en-tete <code>Content-Type: application/json</code>.</p>

                <p style="margin-top:20px; color:#8899aa; font-size:0.85rem; text-transform:uppercase; letter-spacing:0.5px;">Reponse du modele</p>
                <div class="result-box pending" id="resultBox">En attente d'une requete&hellip;</div>
            </div>

            <div class="card">
                <h2>Endpoints disponibles</h2>
                <table>
                    <tr><th>Methode</th><th>Route</th><th>Description</th></tr>
                    <tr>
                        <td><span class="method-badge post">POST</span></td>
                        <td><code>/predict</code></td>
                        <td>Prediction du prix journalier a partir de 13 features. Renvoie <code>{{"prediction": [float]}}</code>.</td>
                    </tr>
                    <tr>
                        <td><span class="method-badge get">GET</span></td>
                        <td><code>/</code></td>
                        <td>Page d'accueil (cette page) avec le constructeur de requete interactif.</td>
                    </tr>
                    <tr>
                        <td><span class="method-badge get">GET</span></td>
                        <td><code>/docs</code></td>
                        <td>Documentation HTML pour un lecteur humain : tableau des features, exemples curl et Python.</td>
                    </tr>
                    <tr>
                        <td><span class="method-badge get">GET</span></td>
                        <td><code>/swagger</code></td>
                        <td>Interface Swagger UI interactive, branchee sur <code>/openapi.json</code>. Permet de tester avec "Try it out".</td>
                    </tr>
                    <tr>
                        <td><span class="method-badge get">GET</span></td>
                        <td><code>/openapi.json</code></td>
                        <td>Descripteur OpenAPI brut (JSON), lisible par Postman, Insomnia ou tout generateur de SDK.</td>
                    </tr>
                </table>
            </div>

            <div class="card">
                <h2>FAQ technique</h2>
                <p style="margin-bottom:16px;">Cliquez sur une question pour afficher la reponse.</p>
                {_faq_html()}
            </div>

            <div class="card">
                <h2>Top 3 features</h2>
                <div class="metrics">
                    <div class="metric"><div class="value">46%</div><div class="label">Puissance moteur</div></div>
                    <div class="metric"><div class="value">27%</div><div class="label">Kilometrage</div></div>
                    <div class="metric"><div class="value">5%</div><div class="label">GPS</div></div>
                </div>
            </div>
        </div>
        <div class="footer">GetAround Pricing API &mdash; Athanor Savouillan &mdash; Jedha Bootcamp</div>

        <script>
            const boolIds = {[key for key, _ in BOOL_OPTIONS]!r};

            function buildPayload() {{
                const row = [
                    document.getElementById('model_key').value,
                    parseInt(document.getElementById('mileage').value || '0', 10),
                    parseInt(document.getElementById('engine_power').value || '0', 10),
                    document.getElementById('fuel').value,
                    document.getElementById('paint_color').value,
                    document.getElementById('car_type').value,
                    ...boolIds.map(id => document.getElementById(id).checked)
                ];
                return {{ input: [row] }};
            }}

            function refreshPreview() {{
                document.getElementById('mileage_val').textContent =
                    parseInt(document.getElementById('mileage').value || '0', 10).toLocaleString('fr-FR') + ' km';
                document.getElementById('engine_power_val').textContent =
                    (document.getElementById('engine_power').value || '0') + ' ch';
                document.getElementById('jsonPreview').textContent = JSON.stringify(buildPayload(), null, 2);
            }}

            document.querySelectorAll('.field select, .field input, .checkbox-item input')
                .forEach(el => el.addEventListener('input', refreshPreview));

            async function sendPrediction() {{
                const btn = document.getElementById('submitBtn');
                const box = document.getElementById('resultBox');
                btn.disabled = true;
                box.className = 'result-box pending';
                box.textContent = 'Envoi en cours...';
                try {{
                    const res = await fetch('/predict', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify(buildPayload())
                    }});
                    const data = await res.json();
                    if (res.ok) {{
                        box.className = 'result-box success';
                        box.textContent = JSON.stringify(data, null, 2);
                    }} else {{
                        box.className = 'result-box error';
                        box.textContent = 'Erreur ' + res.status + ' : ' + JSON.stringify(data, null, 2);
                    }}
                }} catch (err) {{
                    box.className = 'result-box error';
                    box.textContent = 'Erreur reseau : ' + err.message;
                }} finally {{
                    btn.disabled = false;
                }}
            }}

            refreshPreview();
        </script>
    </body>
    </html>
    """


@app.get("/docs", response_class=HTMLResponse)
def documentation():
    """Page de documentation avec exemples curl et Python."""
    return f"""
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>GetAround API - Documentation</title>
        {COMMON_STYLE}
    </head>
    <body>
        <div class="header">
            <h1>Documentation API</h1>
            <p class="subtitle">Tout ce qu'il faut pour utiliser l'endpoint de prediction</p>
            <div class="nav">
                <a href="/" class="btn btn-outline">Accueil</a>
                <a href="/swagger" class="btn btn-primary">Tester avec Swagger</a>
            </div>
        </div>
        <div class="container">

            <div class="card">
                <h2><span class="method-badge post">POST</span> <span class="endpoint">/predict</span></h2>
                <p>Predit le prix journalier de location a partir des caracteristiques du vehicule.</p>
            </div>

            <div class="card">
                <h2>Features attendues</h2>
                <p>JSON avec une cle <code>input</code> contenant une liste de listes. Chaque sous-liste = 1 vehicule (13 features) :</p>
                <table>
                    <tr><th>#</th><th>Feature</th><th>Type</th><th>Exemple</th></tr>
                    <tr><td>0</td><td>model_key</td><td><code>string</code></td><td>Renault</td></tr>
                    <tr><td>1</td><td>mileage</td><td><code>int</code></td><td>140 000</td></tr>
                    <tr><td>2</td><td>engine_power</td><td><code>int</code></td><td>135</td></tr>
                    <tr><td>3</td><td>fuel</td><td><code>string</code></td><td>diesel</td></tr>
                    <tr><td>4</td><td>paint_color</td><td><code>string</code></td><td>black</td></tr>
                    <tr><td>5</td><td>car_type</td><td><code>string</code></td><td>sedan</td></tr>
                    <tr><td>6</td><td>private_parking_available</td><td><code>bool</code></td><td>true</td></tr>
                    <tr><td>7</td><td>has_gps</td><td><code>bool</code></td><td>true</td></tr>
                    <tr><td>8</td><td>has_air_conditioning</td><td><code>bool</code></td><td>false</td></tr>
                    <tr><td>9</td><td>automatic_car</td><td><code>bool</code></td><td>false</td></tr>
                    <tr><td>10</td><td>has_getaround_connect</td><td><code>bool</code></td><td>true</td></tr>
                    <tr><td>11</td><td>has_speed_regulator</td><td><code>bool</code></td><td>true</td></tr>
                    <tr><td>12</td><td>winter_tires</td><td><code>bool</code></td><td>true</td></tr>
                </table>
            </div>

            <div class="card">
                <h2>Exemple curl</h2>
                <pre>curl -X POST "https://athanormark-getaround-pricing-api.hf.space/predict" \\
  -H "Content-Type: application/json" \\
  -d '{{"input": [["Renault", 140000, 135, "diesel", "black", "sedan", true, true, false, false, true, true, true]]}}'</pre>
            </div>

            <div class="card">
                <h2>Exemple Python</h2>
                <pre>import requests

response = requests.post(
    "https://athanormark-getaround-pricing-api.hf.space/predict",
    json={{"input": [["Renault", 140000, 135, "diesel", "black",
                     "sedan", True, True, False, False, True, True, True]]}}
)
print(response.json())  # {{"prediction": [139.12]}}</pre>
            </div>

            <div class="card">
                <h2>Reponse</h2>
                <p>JSON avec une cle <code>prediction</code> contenant la liste des prix predits (EUR/jour) :</p>
                <pre>{{"prediction": [139.12]}}</pre>
            </div>

            <div class="card">
                <h2>Modele</h2>
                <div class="metrics">
                    <div class="metric"><div class="value">0.75</div><div class="label">R2 Score</div></div>
                    <div class="metric"><div class="value">10.29</div><div class="label">MAE (EUR)</div></div>
                    <div class="metric"><div class="value">16.22</div><div class="label">RMSE (EUR)</div></div>
                </div>
                <p style="margin-top:12px">Gradient Boosting Regressor (scikit-learn) &mdash; 200 estimators, depth 5 &mdash; entraine sur 4 843 vehicules.</p>
            </div>
        </div>
        <div class="footer">GetAround Pricing API &mdash; Athanor Savouillan &mdash; Jedha Bootcamp</div>
    </body>
    </html>
    """


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Prédit le prix journalier pour un ou plusieurs véhicules."""
    # Convertir l'input en DataFrame avec les colonnes attendues
    df = pd.DataFrame(request.input, columns=FEATURE_COLUMNS)

    # Conversion des colonnes booleennes en entiers (0/1)
    bool_cols = [
        "private_parking_available", "has_gps", "has_air_conditioning",
        "automatic_car", "has_getaround_connect", "has_speed_regulator",
        "winter_tires"
    ]
    for col in bool_cols:
        df[col] = df[col].astype(int)

    # Prédiction via le pipeline sklearn
    predictions = model.predict(df)
    return {"prediction": [round(p, 2) for p in predictions]}
