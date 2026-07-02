import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Configuration de la page ---
st.set_page_config(
    page_title="GetAround - Delay Analysis",
    page_icon="GA",
    layout="wide"
)

# Thème CSS basé sur le Violet et le Blanc
st.markdown("""
<style>
    .stApp { background-color: #0b0914; }
    header[data-testid="stHeader"] { background-color: #0b0914; }
    section[data-testid="stSidebar"] { background-color: #141126; }
    .hero {
        background: linear-gradient(135deg, #2e1a47 0%, #1a1126 100%);
        border-bottom: 3px solid #9d4edd;
        padding: 2rem 2rem 1.5rem; margin: -1rem -1rem 2rem;
        border-radius: 8px; text-align: center;
    }
    .hero h1 { color: #ffffff; font-size: 2rem; margin-bottom: 0.3rem; }
    .hero p { color: #c3b9d6; font-size: 1.05rem; }
    .badge {
        display: inline-block; background: #ffffff; color: #2e1a47;
        padding: 4px 14px; border-radius: 20px;
        font-size: 0.8rem; font-weight: 600; margin-top: 8px;
    }
    .section-title {
        color: #9d4edd; border-bottom: 1px solid #2a2240;
        padding-bottom: 8px; margin-top: 2rem;
    }
    div[data-testid="stMetric"] {
        background: #141126; border: 1px solid #2a2240;
        border-radius: 10px; padding: 16px;
    }
    div[data-testid="stMetric"] label { color: #a499be; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #ffffff; }
    .footer {
        text-align: center; color: #6b5e85; font-size: 0.85rem;
        margin-top: 3rem; padding-top: 1.5rem;
        border-top: 1px solid #1e1933;
    }
</style>
""", unsafe_allow_html=True)

# Template Plotly assorti (Violet et Blanc/Gris clair)
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0b0914",
    plot_bgcolor="#141126",
    font_color="#ffffff",
    title_font_color="#9d4edd",
    xaxis=dict(gridcolor="#2a2240", zerolinecolor="#2a2240"),
    yaxis=dict(gridcolor="#2a2240", zerolinecolor="#2a2240"),
    colorway=[
        "#9d4edd", "#ffffff", "#c3b9d6",
        "#e06c75", "#7b2cbf", "#e0e0e0",
    ],
    legend=dict(bgcolor="rgba(0,0,0,0)")
)

# --- En-tête ---
st.markdown("""
<div class="hero">
    <h1>GetAround &mdash; Analyse des retards</h1>
    <p>Dashboard d'aide à la décision : quel seuil minimum entre deux locations ?</p>
    <span class="badge">21 310 locations analysées</span>
</div>
""", unsafe_allow_html=True)

# --- Chargement des données ---
DATA_URL = (
    "https://full-stack-assets.s3.eu-west-3.amazonaws.com"
    "/Deployment/get_around_delay_analysis.xlsx"
)


@st.cache_data
def load_data():
    """Charge le fichier Excel depuis S3 (mise en cache Streamlit)."""
    return pd.read_excel(DATA_URL)


df = load_data()
total = len(df)
ended = df[df["state"] == "ended"]
canceled = df[df["state"] == "canceled"]

# --- Barre latérale : paramètres de simulation ---
st.sidebar.markdown("### Paramètres de simulation")
threshold = st.sidebar.slider(
    "Seuil minimum (minutes)", 0, 720, 120, step=15
)
scope = st.sidebar.radio(
    "Scope", ["Toutes les voitures", "Connect uniquement"]
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Recommandation :** 120 min, Connect uniquement")
st.sidebar.markdown("*84% des cas résolus, 36% de locations bloquées*")

# --- KPIs généraux ---
st.markdown(
    '<h3 class="section-title">Vue d\'ensemble</h3>',
    unsafe_allow_html=True
)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total locations", f"{total:,}")
col2.metric(
    "Terminées", f"{len(ended):,}",
    f"{len(ended) / total * 100:.1f}%",
)
col3.metric(
    "Annulées", f"{len(canceled):,}",
    f"{len(canceled) / total * 100:.1f}%",
)

delays = ended["delay_at_checkout_in_minutes"].dropna()
late = delays[delays > 0]
col4.metric("En retard", f"{len(late):,}", f"{len(late)/len(delays)*100:.1f}%")

# --- Distribution des retards ---
st.markdown(
    '<h3 class="section-title">Distribution des retards</h3>',
    unsafe_allow_html=True
)
col_left, col_right = st.columns(2)

with col_left:
    fig = px.pie(
        df, names="checkin_type", title="Type de checkin",
        color_discrete_sequence=["#9d4edd", "#ffffff"]
    )
    fig.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    fig = px.pie(
        df, names="state", title="Statut des locations",
        color_discrete_sequence=["#ffffff", "#e06c75"]
    )
    fig.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

# Histogramme des retards (clippé pour lisibilité)
fig = px.histogram(
    delays.clip(-200, 500), nbins=100,
    title="Distribution des retards au checkout (clippe -200 a 500 min)",
    labels={"value": "Retard (minutes)", "count": "Nombre"}
)
fig.add_vline(
    x=0, line_dash="dash", line_color="#e06c75",
    annotation_text="Heure prevue"
)
fig.update_layout(**PLOTLY_LAYOUT)
st.plotly_chart(fig, use_container_width=True)

# Boxplot par type de checkin
fig = px.box(
    ended, x="checkin_type", y="delay_at_checkout_in_minutes",
    title="Retards par type de checkin", range_y=[-200, 500],
    color="checkin_type",
    color_discrete_sequence=["#9d4edd", "#ffffff"]
)
fig.update_layout(**PLOTLY_LAYOUT, showlegend=False)
st.plotly_chart(fig, use_container_width=True)

# --- Impact sur les locations suivantes ---
st.markdown(
    '<h3 class="section-title">Impact sur les locations suivantes</h3>',
    unsafe_allow_html=True
)

# Jointure : retrouver le retard de la location precedente
prev_delays_map = df.set_index("rental_id")["delay_at_checkout_in_minutes"]
with_prev = df[df["previous_ended_rental_id"].notna()].copy()
with_prev["prev_delay"] = with_prev["previous_ended_rental_id"].map(
    prev_delays_map
)

# Cas problematiques : le retard precedent depasse le buffer disponible
problematic = with_prev[
    with_prev["prev_delay"]
    > with_prev["time_delta_with_previous_rental_in_minutes"]
]

col1, col2, col3 = st.columns(3)
col1.metric("Locations consecutives", f"{len(with_prev):,}")
col2.metric(
    "Cas problematiques", f"{len(problematic):,}",
    f"{len(problematic)/len(with_prev)*100:.1f}%"
)
prob_canceled = problematic[problematic["state"] == "canceled"]
col3.metric("Annulations liees", f"{len(prob_canceled):,}")

fig = px.histogram(
    with_prev["time_delta_with_previous_rental_in_minutes"],
    nbins=50,
    title="Delta entre locations consecutives",
    labels={"value": "Delta (minutes)", "count": "Nombre"}
)
fig.update_layout(**PLOTLY_LAYOUT)
st.plotly_chart(fig, use_container_width=True)

# --- Simulation du seuil selectionne ---
st.markdown(
    f'<h3 class="section-title">'
    f'Simulation : seuil = {threshold} min | {scope.lower()}'
    f'</h3>',
    unsafe_allow_html=True
)

if scope == "Connect uniquement":
    subset = with_prev[with_prev["checkin_type"] == "connect"]
    prob_sub = problematic[problematic["checkin_type"] == "connect"]
else:
    subset = with_prev
    prob_sub = problematic

if threshold == 0:
    affected, solved = 0, 0
else:
    affected = (
        subset["time_delta_with_previous_rental_in_minutes"] < threshold
    ).sum()
    solved = (
        prob_sub["time_delta_with_previous_rental_in_minutes"]
        + threshold >= prob_sub["prev_delay"]
    ).sum()

col1, col2, col3 = st.columns(3)
col1.metric(
    "Locations bloquees", f"{affected}",
    f"{affected/len(subset)*100:.1f}%" if len(subset) > 0 else "0%"
)
col2.metric(
    "Cas resolus", f"{solved}",
    f"{solved/len(prob_sub)*100:.1f}%" if len(prob_sub) > 0 else "0%"
)
pct_revenue = affected / total * 100
col3.metric("Impact revenu (locations)", f"{pct_revenue:.1f}%")

# --- Graphique de compromis selon le seuil ---
st.markdown(
    '<h3 class="section-title">Compromis selon le seuil</h3>',
    unsafe_allow_html=True
)

# Calcul pour tous les seuils de 0 a 720 par pas de 15 min
results = []
for t in range(0, 721, 15):
    for s in ["all", "connect"]:
        sub = (
            with_prev if s == "all"
            else with_prev[with_prev["checkin_type"] == "connect"]
        )
        ps = (
            problematic if s == "all"
            else problematic[problematic["checkin_type"] == "connect"]
        )
        if t == 0:
            aff, sol = 0, 0
        else:
            aff = (
                sub["time_delta_with_previous_rental_in_minutes"] < t
            ).sum()
            sol = (
                ps["time_delta_with_previous_rental_in_minutes"]
                + t >= ps["prev_delay"]
            ).sum()
        results.append({
            "threshold": t,
            "scope": s,
            "pct_bloquees": (
                round(aff / len(sub) * 100, 1) if len(sub) > 0 else 0
            ),
            "pct_resolus": (
                round(sol / len(ps) * 100, 1) if len(ps) > 0 else 0
            ),
        })

df_res = pd.DataFrame(results)

fig = make_subplots(specs=[[{"secondary_y": True}]])
for s, color in [("all", "#9d4edd"), ("connect", "#ffffff")]:
    sub = df_res[df_res["scope"] == s]
    fig.add_trace(
        go.Scatter(
            x=sub["threshold"], y=sub["pct_resolus"],
            name=f"Cas resolus ({s})", mode="lines",
            line=dict(color=color, width=2)
        ),
        secondary_y=False
    )
    fig.add_trace(
        go.Scatter(
            x=sub["threshold"], y=sub["pct_bloquees"],
            name=f"Bloquees ({s})", mode="lines",
            line=dict(color=color, dash="dash", width=2)
        ),
        secondary_y=True
    )

fig.add_vline(
    x=threshold, line_dash="dot", line_color="#e06c75",
    annotation_text=f"Seuil: {threshold}min",
    annotation_font_color="#e06c75"
)
fig.update_layout(
    title="Impact du seuil sur les cas résolus et les locations bloquées",
    **PLOTLY_LAYOUT
)
fig.update_xaxes(title_text="Seuil (minutes)", gridcolor="#2a2240")
fig.update_yaxes(
    title_text="% cas résolus", secondary_y=False, gridcolor="#2a2240"
)
fig.update_yaxes(
    title_text="% locations bloquées", secondary_y=True, gridcolor="#2a2240"
)
st.plotly_chart(fig, use_container_width=True)

# --- Tableau recapitulatif ---
st.markdown(
    '<h3 class="section-title">Tableau recapitulatif</h3>',
    unsafe_allow_html=True
)
summary = df_res[df_res["threshold"].isin([30, 60, 120, 180, 240, 360, 720])]
st.dataframe(
    summary.style.format({
        "pct_bloquees": "{:.1f}%",
        "pct_resolus": "{:.1f}%",
    }),
    use_container_width=True
)

# --- Pied de page ---
st.markdown(
    '<div class="footer">'
    'GetAround Delay Dashboard'
    ' &mdash; Mathieu Le Faou'
    ' &mdash; Jedha Bootcamp'
    '</div>',
    unsafe_allow_html=True
)