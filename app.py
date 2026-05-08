import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Configuration de la page 
st.set_page_config(page_title="Coupe des Diamants — SVM", page_icon="💎", layout="wide")
st.title("💎 Prédiction de la Qualité de Coupe — DOMTCHOUANG")
st.markdown("---")

# Chargement des données et du modèle 
df         = pd.read_csv("diamonds.csv").drop(columns=["Unnamed: 0"])
model      = joblib.load("svm_model.pkl")
scaler     = joblib.load("scaler.pkl")
le_color   = joblib.load("le_color.pkl")
le_clarity = joblib.load("le_clarity.pkl")
le_cut     = joblib.load("le_cut.pkl")

ORDRE    = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
COULEURS = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#3498db']

#  Navigation 
page = st.sidebar.radio("Page", ["📊 Données", "📈 Graphiques", "🔮 Prédiction"])


# PAGE 1 : Données

if page == "📊 Données":
    st.header("📊 Exploration des données")

    # Métriques rapides
    c1, c2, c3 = st.columns(3)
    c1.metric("Nombre de diamants", f"{len(df):,}")
    c2.metric("Prix moyen ($)", f"{df['price'].mean():,.0f}")
    c3.metric("Carat moyen", f"{df['carat'].mean():.2f}")

    st.subheader("Aperçu du dataset")
    st.dataframe(df.head(15), use_container_width=True)

    st.subheader("Statistiques descriptives")
    st.dataframe(df.describe().round(2), use_container_width=True)


# PAGE 2 : Graphiques

elif page == "📈 Graphiques":
    st.header("📈 Visualisations")

    col1, col2 = st.columns(2)

    # Graphique 1 : Nombre de diamants par cut
    with col1:
        fig, ax = plt.subplots()
        counts = df['cut'].value_counts().reindex(ORDRE)
        ax.bar(ORDRE, counts.values, color=COULEURS)
        ax.set_title("Nombre de diamants par Cut")
        ax.set_ylabel("Nombre")
        st.pyplot(fig)
        st.caption("Ce graphique montre combien de diamants appartiennent à chaque qualité de coupe. Ideal est la plus fréquente.")

    # Graphique 2 : Prix moyen par cut
    with col2:
        fig, ax = plt.subplots()
        prix = df.groupby('cut')['price'].mean().reindex(ORDRE)
        ax.bar(ORDRE, prix.values, color=COULEURS)
        ax.set_title("Prix moyen par Cut")
        ax.set_ylabel("Prix ($)")
        st.pyplot(fig)
        st.caption("Paradoxalement, les coupes Fair et Premium ont un prix moyen plus élevé car elles concernent souvent de gros diamants.")

    col3, col4 = st.columns(2)

    # Graphique 3 : Boxplot carat par cut
    with col3:
        fig, ax = plt.subplots()
        df_ord = df.copy()
        df_ord['cut'] = pd.Categorical(df_ord['cut'], categories=ORDRE, ordered=True)
        df_ord.sort_values('cut').boxplot(column='carat', by='cut', ax=ax)
        ax.set_title("Carat par Cut")
        plt.suptitle("")
        st.pyplot(fig)
        st.caption("Les boîtes montrent la distribution du carat selon la coupe. Les coupes Fair contiennent les plus gros diamants en moyenne.")

    # Graphique 4 : Corrélation
    with col4:
        fig, ax = plt.subplots()
        sns.heatmap(df.select_dtypes('number').corr(), annot=True, fmt='.2f',
                    cmap='coolwarm', center=0, ax=ax)
        ax.set_title("Corrélation")
        st.pyplot(fig)
        st.caption("Les cases rouges/bleues indiquent une forte corrélation. x, y, z et carat sont très liés car ils mesurent tous la taille du diamant.")


# PAGE 3 : Prédiction

elif page == "🔮 Prédiction":
    st.header("🔮 Prédire la qualité de coupe")

    col1, col2 = st.columns(2)
    with col1:
        carat   = st.slider("Carat",           0.2,  5.0,  0.8,  step=0.01)
        color   = st.selectbox("Couleur",       ['D','E','F','G','H','I','J'])
        clarity = st.selectbox("Clarté",        ['IF','VVS1','VVS2','VS1','VS2','SI1','SI2','I1'])
        depth   = st.slider("Profondeur (%)",  40.0, 80.0, 61.5, step=0.1)
    with col2:
        table   = st.slider("Table (%)",       40.0, 95.0, 57.0, step=0.5)
        price   = st.slider("Prix ($)",         300, 20000, 2000, step=100)
        x       = st.slider("Longueur x (mm)", 0.0,  11.0,  4.0, step=0.01)
        y       = st.slider("Largeur  y (mm)", 0.0,  11.0,  4.0, step=0.01)
        z       = st.slider("Hauteur  z (mm)", 0.0,   7.0,  2.5, step=0.01)

    if st.button("🔍 Prédire", type="primary"):
        # Encodage + normalisation + prédiction
        data = np.array([[carat,
                          le_color.transform([color])[0],
                          le_clarity.transform([clarity])[0],
                          depth, table, price, x, y, z]])
        pred = le_cut.inverse_transform(model.predict(scaler.transform(data)))[0]

        emoji = {'Fair':'🔴','Good':'🟠','Very Good':'🟡','Premium':'🟢','Ideal':'🔵'}
        st.success(f"{emoji[pred]} **Qualité prédite : {pred}**")
