import streamlit as st
import pandas as pd
import requests
import os
import matplotlib.pyplot as plt
import altair as alt

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

# Check paths
processed_reviews_path = os.path.join("data", "processed_reviews.csv")
aspect_insights_path = os.path.join("data", "aspect_insights.csv")

if not os.path.exists(processed_reviews_path) or not os.path.exists(aspect_insights_path):
    st.error("Missing raw dataset files! Please verify that Phase 1 (preprocess.py) and Phase 4 (aspect_extractor.py) have run successfully.")
    st.stop()

# Load data
df_reviews = pd.read_csv(processed_reviews_path)
df_aspects = pd.read_csv(aspect_insights_path)

# ==========================================
# Sidebar Settings / Status Check
# ==========================================
st.sidebar.title("Configuration & Services")
api_url = st.sidebar.text_input("FastAPI Engine URL", value="http://127.0.0.1:8000")

# Check API health
try:
    health_resp = requests.get(f"{api_url}/health", timeout=2)
    if health_resp.status_code == 200 and health_resp.json().get("status") == "healthy":
        st.sidebar.success("🟢 FastAPI Backend Engine is online!")
    else:
        st.sidebar.warning("🟡 FastAPI Backend returned unhealthy status.")
except Exception:
    st.sidebar.error("🔴 FastAPI Backend Engine is offline.")

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
        else:
            with st.spinner("Processing text and contacting FastAPI Inference Server..."):
                try:
                    payload = {"review": review_input}
                    resp = requests.post(f"{api_url}/predict", json=payload)
                    
                    if resp.status_code == 200:
                        pred_data = resp.json()
                        sentiment = pred_data.get("sentiment")
                        confidence = pred_data.get("confidence", 0.0) * 100
                        probabilities = pred_data.get("probabilities", {})
                        
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
                    else:
                        st.error(f"Prediction server error: Status Code {resp.status_code}. Response: {resp.text}")
                except Exception as e:
                    st.error(f"Unable to connect to active FastAPI engine. Details: {str(e)}")
