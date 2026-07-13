import os
import pickle
import string
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Create FastAPI app
app = FastAPI(title="ReviewPulse API", description="Sentiment Prediction Engine for ReviewPulse")

# Global variables for models
tfidf_vectorizer = None
nb_model = None
stop_words = None
lemmatizer = None

# Initialize resources at startup
@app.on_event("startup")
def startup_event():
    global tfidf_vectorizer, nb_model, stop_words, lemmatizer
    
    # Paths to the model files
    tfidf_path = os.path.join("models", "tfidf_vectorizer.pkl")
    nb_path = os.path.join("models", "naive_bayes_model.pkl")
    
    print("Loading models and vectorizer...")
    if not os.path.exists(tfidf_path) or not os.path.exists(nb_path):
        raise RuntimeError(f"Required models not found. Please ensure they are generated in models/ directory.")
        
    with open(tfidf_path, "rb") as f:
        tfidf_vectorizer = pickle.load(f)
        
    with open(nb_path, "rb") as f:
        nb_model = pickle.load(f)
        
    # Setup NLTK resources
    print("Initializing NLTK resources...")
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
        
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    print("API startup completed successfully.")

def clean_text(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    # Lowercasing
    text = text.lower()
    # Punctuation removal
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenization
    tokens = word_tokenize(text)
    # Stopword removal & WordNet Lemmatization
    cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return " ".join(cleaned_tokens)

# Request schema
class PredictionRequest(BaseModel):
    review: str

# Response schema
class PredictionResponse(BaseModel):
    review: str
    cleaned_review: str
    sentiment: str
    confidence: float
    probabilities: Dict[str, float]

@app.post("/predict", response_model=PredictionResponse)
def predict_sentiment(payload: PredictionRequest):
    if not payload.review.strip():
        raise HTTPException(status_code=400, detail="Review text cannot be empty.")
        
    try:
        # Preprocess text
        cleaned = clean_text(payload.review)
        
        # Vectorize
        features = tfidf_vectorizer.transform([cleaned])
        
        # Predict using Naive Bayes
        pred_label_idx = nb_model.predict(features)[0]
        pred_probs = nb_model.predict_proba(features)[0]
        
        # Mapping index to label string
        sentiment_labels = ['negative', 'neutral', 'positive']
        predicted_sentiment = sentiment_labels[pred_label_idx]
        
        # Build probabilities dictionary
        probs_dict = {
            'negative': float(pred_probs[0]),
            'neutral': float(pred_probs[1]),
            'positive': float(pred_probs[2])
        }
        
        confidence = float(pred_probs[pred_label_idx])
        
        return PredictionResponse(
            review=payload.review,
            cleaned_review=cleaned,
            sentiment=predicted_sentiment,
            confidence=confidence,
            probabilities=probs_dict
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": nb_model is not None}
