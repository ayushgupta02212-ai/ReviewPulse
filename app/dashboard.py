import streamlit as st
import pandas as pd
import os
import pickle
import string
import matplotlib.pyplot as plt
import altair as alt

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Set page config
st.set_page_config(
    page_title="ReviewPulse Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom premium styling using CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;700;900&display=swap');

/* Main font override */
html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

/* Custom header */
.header-container {
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    padding: 30px;
    border-radius: 15px;
    margin-bottom: 25px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    color: white;
    text-align: center;
}
.header-title {
    font-size: 38px;
    font-weight: 900;
    margin-bottom: 5px;
    letter-spacing: -0.5px;
}
.header-subtitle {
    font-size: 16px;
    opacity: 0.9;
    font-weight: 300;
}

/* Card metrics */
.metric-card {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    border-top: 5px solid #2a5298;
    text-align: center;
    margin-bottom: 20px;
}
.metric-card-positive {
    border-top: 5px solid #2ecc71;
}
.metric-card-negative {
    border-top: 5px solid #e74c3c;
}
.metric-val {
    font-size: 32px;
    font-weight: 700;
    color: #2c3e50;
    margin-bottom: 2px;
}
.metric-lbl {
    font-size: 14px;
    color: #7f8c8d;
    text-transform: uppercase;
    font-weight: 500;
    letter-spacing: 0.5px;
}

/* Prediction Output */
.prediction-box {
    padding: 20px;
    border-radius: 10px;
    margin-top: 15px;
    text-align: center;
    font-weight: bold;
}
.pred-pos {
    background-color: rgba(46, 204, 113, 0.15);
    border: 2px solid #2ecc71;
    color: #27ae60;
}
.pred-neg {
    background-color: rgba(231, 76, 60, 0.15);
    border: 2px solid #e74c3c;
    color: #c0392b;
}
.pred-neu {
    background-color: rgba(127, 140, 141, 0.15);
    border: 2px solid #7f8c8d;
    color: #7f8c8d;
}
</style>
""", unsafe_allow_html=True)

# Top Header banner
st.markdown("""
<div class="header-container">
    <div class="header-title">ReviewPulse: Product Intelligence Dashboard</div>
    <div class="header-subtitle">Real-time Review Analysis & Aspect-Based Sentiment Insights</div>
</div>
""", unsafe_allow_html=True)

# Automatically find the absolute project root folder path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Check paths
processed_reviews_path = os.path.join(BASE_DIR, "data", "processed_reviews.csv")
aspect_insights_path = os.path.join(BASE_DIR, "data", "aspect_insights.csv")

if not os.path.exists(processed_reviews_path) or not os.path.exists(aspect_insights_path):
    st.error("Missing raw dataset files! Please verify that Phase 1 (preprocess.py) and Phase 4 (aspect_extractor.py) have run successfully.")
    st.stop()

# Load data
df_reviews = pd.read_csv(processed_reviews_path)
df_aspects = pd.read_csv(aspect_insights_path)

# ==========================================
# Serverless NLP Model Loading & Preprocessing
# ==========================================

# Initialize NLTK resources dynamically
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

@st.cache_resource
def setup_nltk():
    return set(stopwords.words('english')), WordNetLemmatizer()

stop_words, lemmatizer = setup_nltk()

def clean_text(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return " ".join(cleaned_tokens)

# Cache loading of machine learning models
@st.cache_resource
def load_nlp_models():
    tfidf_path = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")
    nb_path = os.path.join(BASE_DIR, "models", "naive_bayes_model.pkl")
    
    if not os.path.exists(tfidf_path) or not os.path.exists(nb_path):
        raise FileNotFoundError("Model binary files are missing. Ensure vectorize.py and train_models.py have run.")
        
    with open(tfidf_path, "rb") as f:
        tfidf = pickle.load(f)
    with open(nb_path, "rb") as f:
        nb = pickle.load(f)
    return tfidf, nb

# ==========================================
# Sidebar Model Configuration & Status Check
# ==========================================
st.sidebar.title("Model Configuration")

try:
    tfidf_vectorizer, nb_model = load_nlp_models()
    st.sidebar.success("🟢 Local Inference Models Loaded!")
    st.sidebar.info("Dashboard running in serverless mode on Streamlit Cloud.")
except Exception as e:
    st.sidebar.error(f"🔴 Error loading models: {str(e)}")
    tfidf_vectorizer, nb_model = None, None

# ==========================================
# Main Layout Tabs
# ==========================================
tab_dashboard, tab_prediction = st.tabs(["📈 Sentiment & Product Insights", "🔮 Real-Time Sentiment Predictor"])

with tab_dashboard:
    # 1. Operational Key Metrics
    st.subheader("Key Operational Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-val">15,000</div>
            <div class="metric-lbl">Total Reviews Analyzed</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-val">67.7 words</div>
            <div class="metric-lbl">Avg. Review Word Count (Before)</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-val">34.9 words</div>
            <div class="metric-lbl">Avg. Review Word Count (After)</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-val">48.37%</div>
            <div class="metric-lbl">Vocabulary Compression Rate</div>
        </div>
        """, unsafe_allow_html=True)

    # 2. Main Analytics Charts
    st.write("---")
    col_left, col_right = st.columns([1, 1.3])

    with col_left:
        st.subheader("Dataset Sentiment Balance")
        sentiment_counts = df_reviews['sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Reviews']
        
        # Color mapping for sentiment
        chart = alt.Chart(sentiment_counts).mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8).encode(
            x=alt.X('Sentiment:N', sort=['positive', 'neutral', 'negative']),
            y=alt.Y('Reviews:Q', title="Count"),
            color=alt.Color('Sentiment:N', scale=alt.Scale(
                domain=['positive', 'neutral', 'negative'],
                range=['#2ecc71', '#95a5a6', '#e74c3c']
            ), legend=None)
        ).properties(
            height=300
        )
        st.altair_chart(chart, use_container_width=True)

    with col_right:
        st.subheader("Aspect Sentiment Extraction Analysis")
        # Transform aspects dataframe to long-form for grouped bar charts in Altair
        df_aspects_melted = pd.melt(
            df_aspects, 
            id_vars=['aspect'], 
            value_vars=['positive', 'negative'], 
            var_name='Sentiment', 
            value_name='Mentions'
        )
        
        aspect_chart = alt.Chart(df_aspects_melted).mark_bar().encode(
            x=alt.X('Sentiment:N', title=None, sort=['positive', 'negative']),
            y=alt.Y('Mentions:Q', title="Mentions"),
            color=alt.Color('Sentiment:N', scale=alt.Scale(
                domain=['positive', 'negative'],
                range=['#2ecc71', '#e74c3c']
            )),
            column=alt.Column('aspect:N', title=None, header=alt.Header(labelFontSize=14, labelColor='#2c3e50'))
        ).properties(
            width=80,
            height=300
        ).configure_facet(
            spacing=10
        )
        st.altair_chart(aspect_chart, use_container_width=True)

    # Aspect Ratios table summary
    st.subheader("Granular Aspect Insights Detail Table")
    df_aspects['total_mentions'] = df_aspects['positive'] + df_aspects['negative']
    df_aspects['pos_ratio_pct'] = (df_aspects['positive'] / df_aspects['total_mentions'] * 100).round(1)
    df_aspects_display = df_aspects.sort_values(by='total_mentions', ascending=False)
    
    st.dataframe(
        df_aspects_display.rename(columns={
            'aspect': 'Product Feature',
            'positive': 'Positive Mentions',
            'negative': 'Negative Mentions',
            'total_mentions': 'Total Mentions',
            'pos_ratio_pct': 'Approval Ratio (%)'
        }),
        use_container_width=True,
        hide_index=True
    )

with tab_prediction:
    st.subheader("Real-Time Review Analysis Terminal")
    st.write("Write a review below to test the active NLP baseline classifier.")

    review_input = st.text_area(
        "Review Text",
        value="This cell phone has a beautiful, bright screen and the battery life lasts for days! Highly recommended.",
        height=150
    )

    if st.button("Predict Sentiment 🔮", use_container_width=True):
        if not review_input.strip():
            st.warning("Please enter some text to test.")
        elif tfidf_vectorizer is None or nb_model is None:
            st.error("Model inference is currently unavailable due to loading errors.")
        else:
            with st.spinner("Processing review text and running local inference..."):
                try:
                    # Clean text
                    cleaned = clean_text(review_input)
                    
                    # Vectorize
                    features = tfidf_vectorizer.transform([cleaned])
                    
                    # Predict using local Naive Bayes
                    pred_label_idx = nb_model.predict(features)[0]
                    pred_probs = nb_model.predict_proba(features)[0]
                    
                    sentiment_labels = ['negative', 'neutral', 'positive']
                    sentiment = sentiment_labels[pred_label_idx]
                    confidence = float(pred_probs[pred_label_idx]) * 100
                    
                    probabilities = {
                        'negative': float(pred_probs[0]),
                        'neutral': float(pred_probs[1]),
                        'positive': float(pred_probs[2])
                    }
                    
                    # Apply CSS styling class based on predicted sentiment
                    class_name = "pred-neu"
                    if sentiment == "positive":
                        class_name = "pred-pos"
                    elif sentiment == "negative":
                        class_name = "pred-neg"
                        
                    # Show prediction card
                    st.markdown(f"""
                    <div class="prediction-box {class_name}">
                        <div style="font-size: 24px; text-transform: uppercase;">{sentiment}</div>
                        <div style="font-size: 14px; font-weight: 300;">Confidence: {confidence:.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Render confidence table details
                    st.write("---")
                    st.write("Confidence Breakdown per Sentiment Class:")
                    df_probs = pd.DataFrame({
                        'Sentiment Class': ['Positive', 'Neutral', 'Negative'],
                        'Confidence Probability (%)': [
                            f"{probabilities.get('positive', 0.0)*100:.2f}%",
                            f"{probabilities.get('neutral', 0.0)*100:.2f}%",
                            f"{probabilities.get('negative', 0.0)*100:.2f}%"
                        ]
                    })
                    st.table(df_probs)
                except Exception as e:
                    st.error(f"Inference error during processing: {str(e)}")
