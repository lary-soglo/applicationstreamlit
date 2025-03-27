import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from sklearn.impute import SimpleImputer
import io

# Configuration de la page avec thème sombre
st.set_page_config(page_title="Système de Recommandation de Films", page_icon="🎬", layout="wide")

# Style CSS personnalisé pour thème sombre
st.markdown("""
    <style>
    :div[data-testid="stContainer"] {
    padding: 15px;
    border-radius: 10px;
    background-color: #2a2a2a;
    border-left: 5px solid #1890ff;
    margin: 10px 0;
    } 
    :root {
        --primary-color: #1890ff;
        --background-color: #121212;
        --card-color: #1e1e1e;
        --text-color: #ffffff;
        --border-color: #333;
        --warning-color: #ff4b4b;
        --success-color: #4CAF50;
        --rating-high: #4CAF50;
        --rating-medium: #FFC107;
        --rating-low: #F44336;
        --unavailable-text: #888;
    }
    
    .main {
        background-color: var(--background-color);
        color: var(--text-color);
    }
    
    .stButton>button {
        background-color: var(--primary-color);
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
    }
    
    .stButton>button:hover {
        background-color: #1474d4;
    }
    
    .dataframe {
        background-color: var(--card-color) !important;
        color: var(--text-color) !important;
        border-radius: 5px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    }
    
    .recommendation-card {
        background-color: var(--card-color);
        color: var(--text-color);
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.1);
        border: 1px solid var(--border-color);
    }
    
    .highlight-box {
        background-color: #2a2a2a;
        padding: 15px;
        border-radius: 5px;
        margin: 15px 0;
        border-left: 5px solid var(--primary-color);
    }
    
    .warning-box {
        background-color: #3a1e1e;
        padding: 15px;
        border-radius: 5px;
        margin: 15px 0;
        border-left: 5px solid var(--warning-color);
    }
    
    .rating-high {
        color: var(--rating-high);
        font-weight: bold;
    }
    
    .rating-medium {
        color: var(--rating-medium);
        font-weight: bold;
    }
    
    .rating-low {
        color: var(--rating-low);
        font-weight: bold;
    }
    
    .unavailable-text {
        color: var(--unavailable-text);
    }
    
    /* Styles pour les composants Streamlit */
    .stTextInput>div>div>input,
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>select {
        background-color: #333;
        color: white;
        border-color: #555;
    }
    
    .stTextInput>div>div>input:focus,
    .stNumberInput>div>div>input:focus,
    .stSelectbox>div>div>select:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 0.2rem rgba(24, 144, 255, 0.25);
    }
    
    .stDataFrame {
        background-color: var(--card-color) !important;
    }
    
    /* Style pour les onglets */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: var(--card-color);
        border-radius: 5px 5px 0 0;
        padding: 10px 20px;
        margin-right: 5px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--primary-color);
        color: white;
    }
    
    /* Style pour les tooltips */
    .stTooltip {
        background-color: var(--card-color) !important;
        color: var(--text-color) !important;
        border: 1px solid var(--border-color) !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Titre de l'application avec icône
st.title("🎬 Système de Recommandation de Films")


# Initialisation de l'état de session
if 'ratings_df' not in st.session_state:
    st.session_state.ratings_df = pd.DataFrame()
if 'manual_data' not in st.session_state:
    st.session_state.manual_data = {'Utilisateur': [], 'Film': [], 'Note': []}

# Fonction pour calculer la similarité Pearson
def pearson_similarity(user1_ratings, user2_ratings):
    common_movies = (user1_ratings.notna()) & (user2_ratings.notna())
    if common_movies.sum() < 2:
        return 0
    return pearsonr(user1_ratings[common_movies], user2_ratings[common_movies])[0]

# Fonction principale de calcul des recommandations
def calculate_recommendations(df, n_recommendations=5, similarity_metric='cosine'):
    # Création de la matrice utilisateur-film
    ratings_matrix = df.pivot_table(index='Utilisateur', columns='Film', values='Note')
    
    # Imputation des valeurs manquantes
    imputer = SimpleImputer(strategy='mean')
    imputed_matrix = imputer.fit_transform(ratings_matrix)
    imputed_df = pd.DataFrame(imputed_matrix, index=ratings_matrix.index, columns=ratings_matrix.columns)
    
    # Calcul de la similarité
    if similarity_metric == 'cosine':
        user_similarity = cosine_similarity(imputed_df)
    else:
        n_users = imputed_df.shape[0]
        user_similarity = np.zeros((n_users, n_users))
        for i in range(n_users):
            for j in range(n_users):
                if i == j:
                    user_similarity[i, j] = 1.0
                else:
                    user_similarity[i, j] = pearson_similarity(imputed_df.iloc[i], imputed_df.iloc[j])
    
    user_similarity_df = pd.DataFrame(user_similarity, index=imputed_df.index, columns=imputed_df.index)
    
    # Calcul des prédictions
    user_predicted_ratings = np.dot(user_similarity_df, imputed_df) / np.array([np.abs(user_similarity_df).sum(axis=1)]).T
    predicted_ratings_df = pd.DataFrame(user_predicted_ratings, index=imputed_df.index, columns=imputed_df.columns)
    
    # Génération des recommandations
    recommendations = {}
    for user in predicted_ratings_df.index:
        user_ratings = df[df['Utilisateur'] == user]
        rated_movies = user_ratings['Film'].unique()
        available_movies = [m for m in predicted_ratings_df.columns if m in df['Film'].unique()]
        unrated_movies = [m for m in available_movies if m not in rated_movies]
        
        if unrated_movies:
            user_predictions = predicted_ratings_df.loc[user, unrated_movies]
            top_movies = user_predictions.sort_values(ascending=False).head(n_recommendations)
            if not top_movies.empty:
                recommendations[user] = top_movies
                
    return ratings_matrix, predicted_ratings_df, recommendations

# Sidebar pour les paramètres
with st.sidebar:
    st.header("⚙️ Paramètres")
    top_n = st.number_input("Nombre de recommandations (Top N)", min_value=1, max_value=20, value=5)
    similarity_metric = st.selectbox("Métrique de similarité", ['cosine', 'pearson'], index=0)
    
    st.header("👀 Options d'affichage")
    show_raw_data = st.checkbox("Afficher les données brutes", value=True)
    show_ratings_matrix = st.checkbox("Afficher la matrice des notes", value=True)
    show_predictions = st.checkbox("Afficher les prédictions", value=False)

# Onglets principaux
tab1, tab2, tab3 = st.tabs(["📤 Charger des données", "✏️ Ajouter des notes", "🔍 Recherche"])

with tab1:
    st.header("📤 Chargement des données")
    uploaded_file = st.file_uploader("Télécharger un fichier CSV", type=['csv'])
    
    if uploaded_file is not None:
        try:
            bytes_data = uploaded_file.getvalue()
            data = io.StringIO(bytes_data.decode('utf-8'))
            new_df = pd.read_csv(data)
            
            required_columns = {'Utilisateur', 'Film', 'Note'}
            if not required_columns.issubset(new_df.columns):
                st.error("❌ Le fichier CSV doit contenir les colonnes: 'Utilisateur', 'Film', 'Note'")
            else:
                st.session_state.ratings_df = pd.concat([st.session_state.ratings_df, new_df], ignore_index=True)
                st.success("✅ Données chargées avec succès!")
                
        except Exception as e:
            st.error(f"❌ Erreur lors de la lecture du fichier: {e}")

# ... [le code précédent jusqu'à la section tab2] ...

with tab2:
    st.header("✏️ Ajout manuel de notes")
    
    # Utilisation d'un formulaire avec gestion d'état séparée
    with st.form(key='manual_rating_form'):
        col1, col2, col3 = st.columns(3)
        with col1:
            user_input = st.text_input("Utilisateur", key="user_input_widget")
        with col2:
            movie_input = st.text_input("Film", key="movie_input_widget")
        with col3:
            rating_input = st.number_input("Note (1-5)", min_value=1, max_value=5, step=1, key="rating_input_widget")
        
        submitted = st.form_submit_button("➕ Ajouter la note")
        
        if submitted:
            if user_input and movie_input:
                # Création d'une nouvelle entrée
                new_entry = {
                    'Utilisateur': user_input,
                    'Film': movie_input,
                    'Note': rating_input
                }
                
                # Conversion en DataFrame et concaténation
                new_df = pd.DataFrame([new_entry])
                st.session_state.ratings_df = pd.concat(
                    [st.session_state.ratings_df, new_df], 
                    ignore_index=True
                ).drop_duplicates(subset=['Utilisateur', 'Film'], keep='last')
                
                st.success("✅ Note ajoutée avec succès!")
                
                # Réinitialisation via rerun plutôt que modification directe de l'état
                st.rerun()
            else:
                st.warning("⚠️ Veuillez entrer un utilisateur et un film")

with tab3:
    st.header("🔍 Recherche et recommandations")
    
    if not st.session_state.ratings_df.empty:
        ratings_matrix, predicted_ratings_df, recommendations = calculate_recommendations(
            st.session_state.ratings_df, top_n, similarity_metric)
        
        # Section Recommandation ciblée
        st.subheader("🎯 Recommandation ciblée")
        col1, col2 = st.columns(2)
        with col1:
            target_user = st.selectbox("Sélectionnez un utilisateur", 
                                     sorted(st.session_state.ratings_df['Utilisateur'].unique()),
                                     key="target_user")
        with col2:
            target_movie = st.selectbox("Sélectionnez un film", 
                                      sorted(st.session_state.ratings_df['Film'].unique()),
                                      key="target_movie")
        
        user_rating = st.session_state.ratings_df[
            (st.session_state.ratings_df['Utilisateur'] == target_user) & 
            (st.session_state.ratings_df['Film'] == target_movie)]
        
        has_rated = not user_rating.empty
        predicted_rating = None
        
        if not has_rated:
            try:
                predicted_rating = predicted_ratings_df.loc[target_user, target_movie]
            except KeyError:
                predicted_rating = None
        
        st.markdown(f"""
        <div class="highlight-box">
            <h4>Recommandation pour {target_user}</h4>
            <p>🎬 Film sélectionné: <strong>{target_movie}</strong></p>
            <p>⭐ Note existante: <strong>{
                user_rating['Note'].values[0] if has_rated 
                else '<span class="unavailable-text">Non évalué</span>'
            }</strong></p>
            <p>🔮 Note prédite: <strong>{
                f'<span class="rating-high">{predicted_rating:.2f}</span>' if predicted_rating is not None and predicted_rating >= 4 
                else f'<span class="rating-medium">{predicted_rating:.2f}</span>' if predicted_rating is not None and predicted_rating >= 3 
                else f'<span class="rating-low">{predicted_rating:.2f}</span>' if predicted_rating is not None 
                else '<span class="unavailable-text">Non disponible</span>'
            }</strong></p>
            <p>💡 Recommandation: <strong>{
                "✅ Oui" if predicted_rating is not None and predicted_rating >= 3.5 
                else "❌ Non, nous ne recommandons pas ce film"
            }</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Section Top N recommandations
        st.markdown("---")
        st.subheader(f"🏆 Top {top_n} recommandations pour {target_user}")
        
        if target_user in recommendations and not recommendations[target_user].empty:
            user_recommendations = recommendations[target_user]
            
            # Affichage sous forme de tableau
            reco_df = pd.DataFrame({
                'Film': user_recommendations.index,
                'Note prédite': user_recommendations.values.round(2),
                'Recommandé': ['✅' if x >= 3.5 else '❌' for x in user_recommendations.values]
            })
            
            st.dataframe(reco_df, height=(len(reco_df) + 1) * 35 + 3)
            
            # Affichage complémentaire sous forme de cartes
            st.markdown("---")
            st.subheader("📋 Détails des recommandations")
            
            for movie, rating in user_recommendations.items():
                confidence = 'Élevée' if rating >= 4 else 'Moyenne' if rating >= 3 else 'Faible'
                rating_class = 'rating-high' if rating >= 4 else 'rating-medium' if rating >= 3 else 'rating-low'
                
                st.markdown(f"""
                <div class="recommendation-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h4 style="margin: 0;">🎬 {movie}</h4>
                        <span style="font-size: 1.5em;">{'✅' if rating >= 3.5 else '❌'}</span>
                    </div>
                    <p>⭐ <strong>Note prédite:</strong> <span class="{rating_class}">{rating:.2f}/5</span></p>
                    <p>📊 <strong>Métrique:</strong> {'Cosinus' if similarity_metric == 'cosine' else 'Pearson'}</p>
                    <p>🔍 <strong>Confiance:</strong> {confidence}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="warning-box">
                <h4 style="color: var(--warning-color); margin-top: 0;">⚠️ Aucune recommandation disponible pour {target_user}</h4>
                <p>Raisons possibles :</p>
                <ul>
                    <li>Tous les films ont déjà été notés par cet utilisateur</li>
                    <li>Données insuffisantes pour calculer des similarités</li>
                    <li>Pas assez d'utilisateurs similaires dans la base</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Affichage des données
        if show_raw_data:
            st.markdown("---")
            st.subheader("📊 Données brutes")
            st.dataframe(st.session_state.ratings_df)
        
        if show_ratings_matrix:
            st.markdown("---")
            st.subheader("📈 Matrice des notes")
            st.dataframe(ratings_matrix)
        
        if show_predictions:
            st.markdown("---")
            st.subheader("🔮 Matrice des prédictions")
            st.dataframe(predicted_ratings_df)
    
    else:
        st.markdown("""
        <div class="highlight-box">
            <h4>ℹ️ Aucune donnée disponible</h4>
            <p>Veuillez charger des données ou ajouter des notes manuellement pour commencer.</p>
            <p>Vous pouvez :</p>
            <ul>
                <li>Télécharger un fichier CSV dans l'onglet "📤 Charger des données"</li>
                <li>Ajouter des notes manuellement dans l'onglet "✏️ Ajouter des notes"</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Pied de page
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: var(--unavailable-text); font-size: 0.9em; padding: 15px;">
        Système de recommandation de Films - Réalisé par SOGLO Grace Hillary | IFRI MASTER 2025 © 2025 
    </div>
""", unsafe_allow_html=True)