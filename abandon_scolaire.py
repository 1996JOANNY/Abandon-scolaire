# ============================================================================
# IMPORTS DES BIBLIOTHÈQUES
# ============================================================================

import streamlit as st                    # Interface web interactive pour créer des applications de données
import pandas as pd                       # Manipulation et analyse de données structurées
import numpy as np                        # Calculs numériques et opérations sur les arrays
import seaborn as sns                     # Visualisations statistiques avancées basées sur matplotlib
import matplotlib.pyplot as plt          # Création de graphiques et visualisations de base
import plotly.express as px              # Graphiques interactifs et modernes
from sklearn.cluster import KMeans       # Algorithme de clustering non supervisé
from sklearn.ensemble import RandomForestClassifier  # Classificateur par forêt aléatoire
from sklearn.preprocessing import StandardScaler     # Normalisation des données
from mlxtend.frequent_patterns import apriori, association_rules  # Analyse des règles d'association
import io                                # Gestion des flux d'entrée/sortie en mémoire
from reportlab.pdfgen import canvas      # Génération de documents PDF
from reportlab.lib.pagesizes import letter  # Formats de page pour PDF

# ============================================================================
# CONFIGURATION DE L'INTERFACE STREAMLIT
# ============================================================================

# Configuration de la page principale avec titre et mise en page large
st.set_page_config(page_title="🎓 Prévention de l'abandon scolaire", layout="wide")

# Injection de CSS personnalisé pour styliser l'application
st.markdown("""
    <style>
        .main {background-color: #f9fafa;}          /* Couleur de fond gris clair */
        h1, h2, h3 {color: #004080;}                /* Titres en bleu foncé */
        .stButton>button {background-color: #007acc; color: white; border-radius: 8px;}  /* Boutons bleus arrondis */
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# FONCTION DE CHARGEMENT ET PRÉPARATION DES DONNÉES OULAD
# ============================================================================

@st.cache_data  # Cache pour éviter de recharger les données à chaque interaction
def load_oulad_data():
    """
    Charge et préprocess les données du dataset OULAD (Open University Learning Analytics Dataset)
    Fusionne plusieurs fichiers CSV pour créer un dataset unifié d'analyse
    """
    # Chargement des trois fichiers CSV principaux du dataset OULAD
    assessments = pd.read_csv("assessments.csv")           # Informations sur les évaluations
    student_assessment = pd.read_csv("studentAssessment.csv")  # Résultats des étudiants aux évaluations
    student_info = pd.read_csv("studentInfo.csv")          # Informations démographiques des étudiants
    
    # Fusion des données : d'abord évaluations avec résultats
    merged = student_assessment.merge(assessments, on='id_assessment', how='left')
    # Puis ajout des informations étudiants
    merged = merged.merge(student_info, on='id_student', how='left')
    
    # Agrégation des données par étudiant pour avoir une ligne par étudiant
    df_grouped = merged.groupby('id_student').agg({
        'score': 'mean',                    # Score moyen de l'étudiant
        'date_submitted': 'count',          # Nombre d'évaluations soumises
        'studied_credits': 'mean',          # Crédits moyens étudiés
        'age_band': 'first',               # Tranche d'âge (première valeur)
        'gender': 'first',                 # Genre (première valeur)
        'region': 'first',                 # Région (première valeur)
        'highest_education': 'first',      # Niveau d'éducation le plus élevé
        'disability': 'first',             # Statut de handicap
        'final_result': 'first'            # Résultat final du cours
    }).reset_index()
    
    # Renommage des colonnes en français pour plus de clarté
    df_grouped.rename(columns={
        'score': 'score_moyen',
        'date_submitted': 'nb_evaluations',
        'studied_credits': 'credits_etudies',
        'age_band': 'age',
        'gender': 'sexe',
        'highest_education': 'niveau_parental',
        'disability': 'handicap',
        'final_result': 'abandon'
    }, inplace=True)
    
    # Génération de données synthétiques pour enrichir l'analyse
    df_grouped['temps_moodle'] = np.random.uniform(0, 20, size=len(df_grouped))        # Temps passé sur Moodle (heures)
    df_grouped['participation_forum'] = np.random.randint(0, 50, size=len(df_grouped)) # Nombre de posts sur forum
    df_grouped['satisfaction'] = np.random.uniform(1, 5, size=len(df_grouped))         # Note de satisfaction (1-5)
    
    # Transformation de la variable cible : 1 si abandon ('Withdrawn' ou 'Fail'), 0 sinon
    df_grouped['abandon'] = df_grouped['abandon'].map(lambda x: 1 if x in ['Withdrawn', 'Fail'] else 0)
    
    return df_grouped

# ============================================================================
# FONCTION DE PRÉPROCESSING DES DONNÉES
# ============================================================================

def preprocess(df):
    """
    Préprocess les données pour les préparer à l'analyse machine learning
    - Gère les valeurs manquantes
    - Encode les variables catégorielles
    """
    # Remplacement des valeurs manquantes par la moyenne pour les colonnes numériques
    df = df.fillna(df.mean(numeric_only=True))
    
    # Identification des colonnes catégorielles (type object)
    cat_cols = df.select_dtypes(include='object').columns
    
    # Encodage one-hot des variables catégorielles (drop_first=True évite la multicolinéarité)
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    return df

# ============================================================================
# PIPELINE PRINCIPAL D'ANALYSE
# ============================================================================

def analysis_pipeline(df, label, pdf_key):
    """
    Pipeline complet d'analyse des données incluant :
    - Visualisations exploratoires
    - Clustering
    - Classification
    - Règles d'association
    - Simulation individuelle
    """
    # Préprocessing des données
    df = preprocess(df)

    # ========================================================================
    # SECTION 1 : VISUALISATIONS DE BASE
    # ========================================================================
    
    # Création de deux colonnes pour l'affichage côte à côte
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Histogramme")
        # Histogramme du score moyen coloré par statut d'abandon
        st.plotly_chart(px.histogram(df, x=label, color='abandon', 
                                   color_discrete_sequence=['#1f77b4', '#ff7f0e']))

    with col2:
        st.subheader("🔥 Heatmap des corrélations")
        # Matrice de corrélation entre toutes les variables numériques
        fig, ax = plt.subplots(figsize=(6,4))
        sns.heatmap(df.corr(), cmap='viridis', ax=ax)
        st.pyplot(fig)

    # ========================================================================
    # SECTION 2 : ANALYSE DÉTAILLÉE AVEC BOXPLOTS
    # ========================================================================
    
    with st.expander("📦 Boxplots détaillés"):
        # Identification des colonnes numériques
        num_cols = df.select_dtypes(include=np.number).columns
        
        # Création d'un boxplot pour chaque variable numérique (sauf 'abandon')
        for col in num_cols:
            if col != 'abandon':
                # Boxplot comparant les distributions entre étudiants qui abandonnent et ceux qui réussissent
                fig = px.box(df, y=col, color='abandon', 
                           color_discrete_sequence=['#2ca02c', '#d62728'])
                st.plotly_chart(fig)

    # ========================================================================
    # SECTION 3 : CLUSTERING K-MEANS
    # ========================================================================
    
    st.subheader("🔍 Clustering K-Means")
    
    # Standardisation des données pour le clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.drop(['abandon'], axis=1, errors='ignore'))
    
    # Application de K-Means avec 3 clusters
    df['cluster'] = KMeans(n_clusters=3, random_state=42).fit_predict(X_scaled)
    
    # Visualisation des clusters dans un scatter plot 2D
    st.plotly_chart(px.scatter(df, x=label, y='temps_moodle', color='cluster', 
                             color_continuous_scale='Turbo'))

    # ========================================================================
    # SECTION 4 : CLASSIFICATION RANDOM FOREST
    # ========================================================================
    
    with st.expander("🤖 Classification Random Forest"):
        # Préparation des données pour la classification
        X = df.drop(['abandon', 'cluster'], axis=1, errors='ignore')  # Variables explicatives
        y = df['abandon']                                             # Variable cible
        
        # Entraînement du modèle Random Forest
        clf = RandomForestClassifier().fit(X, y)
        st.success("Le modèle Random Forest a été entraîné avec succès.")

    # ========================================================================
    # SECTION 5 : RÈGLES D'ASSOCIATION
    # ========================================================================
    
    with st.expander("📈 Règles d'association"):
        # Copie du dataframe pour la transformation
        df_assoc = df.copy()
        
        # Discrétisation des variables continues en 3 catégories (Low, Medium, High)
        for col in df_assoc.select_dtypes(include=['float64', 'int64']).columns:
            if df_assoc[col].nunique() >= 3:  # Seulement si au moins 3 valeurs uniques
                df_assoc[col] = pd.qcut(df_assoc[col], q=3, labels=['Low', 'Medium', 'High'], 
                                      duplicates='drop')
        
        # Encodage one-hot pour l'algorithme Apriori
        df_assoc = pd.get_dummies(df_assoc)
        
        # Filtrage pour ne garder que les colonnes binaires (0 et 1)
        df_assoc = df_assoc.loc[:, df_assoc.apply(lambda x: set(x.unique()).issubset({0, 1}))]
        
        if not df_assoc.empty:
            # Application de l'algorithme Apriori pour trouver les itemsets fréquents
            freq_items = apriori(df_assoc, min_support=0.1, use_colnames=True)
            
            # Génération des règles d'association
            rules = association_rules(freq_items, metric='confidence', min_threshold=0.6)
            
            # Affichage des règles les plus intéressantes
            st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence']])
        else:
            st.info("Aucune règle d'association trouvée.")

    # ========================================================================
    # SECTION 6 : SIMULATION INDIVIDUELLE ET GÉNÉRATION DE RAPPORT
    # ========================================================================
    
    with st.expander("🎯 Simulation individuelle et rapport"):
        # Création d'un formulaire interactif pour saisir les données d'un étudiant
        input_data = {}
        
        # Pour chaque variable du modèle, création d'un widget d'entrée approprié
        for col in X.columns:
            if X[col].dtype in ['float64', 'int64']:
                # Slider pour les variables numériques
                input_data[col] = st.slider(f"{col}", 
                                          float(X[col].min()), 
                                          float(X[col].max()), 
                                          float(X[col].mean()))
            else:
                # Selectbox pour les variables catégorielles
                input_data[col] = st.selectbox(f"{col}", X[col].unique())
        
        # Transformation des données d'entrée en DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Alignement avec les colonnes du modèle (ajout de colonnes manquantes avec 0)
        input_df = pd.get_dummies(input_df).reindex(columns=X.columns, fill_value=0)
        
        # Prédiction du risque d'abandon (probabilité de la classe positive)
        risk = clf.predict_proba(input_df)[0][1]
        
        # Affichage du risque sous forme de métrique
        st.metric(label="Risque d'abandon estimé", value=f"{risk:.2%}")

        # ====================================================================
        # GÉNÉRATION DE RAPPORT PDF
        # ====================================================================
        
        if st.button("📥 Générer le rapport PDF", key=pdf_key):
            # Création d'un buffer en mémoire pour le PDF
            buffer = io.BytesIO()
            
            # Création d'un canvas PDF avec format letter
            c = canvas.Canvas(buffer, pagesize=letter)
            
            # Écriture du risque d'abandon en haut de la page
            c.drawString(100, 750, f"Risque d'abandon: {risk:.2%}")
            
            # Position initiale pour les détails
            y = 720
            
            # Écriture de chaque paramètre d'entrée
            for k, v in input_data.items():
                c.drawString(100, y, f"{k}: {v}")
                y -= 20  # Déplacement vers le bas pour la ligne suivante
            
            # Sauvegarde du PDF
            c.save()
            
            # Retour au début du buffer pour la lecture
            buffer.seek(0)
            
            # Bouton de téléchargement du rapport
            st.download_button("Télécharger le rapport PDF", 
                             buffer, 
                             "rapport_abandon.pdf", 
                             "application/pdf")

# ============================================================================
# EXÉCUTION PRINCIPALE DE L'APPLICATION
# ============================================================================

# Titre principal de l'application
st.title("🎓 Tableau de bord : Prévention de l'abandon scolaire (OULAD)")

# Chargement des données OULAD
df_oulad = load_oulad_data()

# Lancement du pipeline d'analyse complet
analysis_pipeline(df_oulad, label='score_moyen', pdf_key="pdf_oulad")