import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

def main():
    processed_path = os.path.join("data", "processed_reviews.csv")
    tfidf_vectorizer_path = os.path.join("models", "tfidf_vectorizer.pkl")
    nb_model_path = os.path.join("models", "naive_bayes_model.pkl")
    lstm_model_path = os.path.join("models", "lstm_model.keras")
    lstm_tokenizer_path = os.path.join("models", "lstm_tokenizer.pkl")
    confusion_img_path = os.path.join("data", "confusion_matrices.png")

    print("Loading dataset for evaluation...")
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"Processed dataset not found at {processed_path}")

    df = pd.read_csv(processed_path)
    df['cleaned_body'] = df['cleaned_body'].fillna("").astype(str)

    # Map sentiments to integers
    sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    df['label'] = df['sentiment'].map(sentiment_map)

    X = df['cleaned_body'].astype(str).tolist()
    y = df['label'].astype(int).tolist()

    # Recreate the exact same train/validation split (stratified, seed=42)
    _, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Convert labels to numpy arrays for Keras compatibility
    y_val = np.array(y_val)
    
    # ---------------------------------------------
    # 1. Evaluate Naive Bayes Baseline
    # ---------------------------------------------
    print("\nEvaluating Naive Bayes Model...")
    with open(tfidf_vectorizer_path, "rb") as f:
        tfidf_vectorizer = pickle.load(f)
    
    with open(nb_model_path, "rb") as f:
        nb_classifier = pickle.load(f)

    X_val_tfidf = tfidf_vectorizer.transform(X_val)
    y_pred_nb = nb_classifier.predict(X_val_tfidf)

    nb_report = classification_report(
        y_val, 
        y_pred_nb, 
        target_names=['negative', 'neutral', 'positive'], 
        output_dict=True
    )
    nb_report_str = classification_report(
        y_val, 
        y_pred_nb, 
        target_names=['negative', 'neutral', 'positive']
    )

    # ---------------------------------------------
    # 2. Evaluate LSTM Model
    # ---------------------------------------------
    print("\nEvaluating LSTM Model...")
    with open(lstm_tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    # Load Keras LSTM model
    lstm_model = tf.keras.models.load_model(lstm_model_path)

    # Convert text to sequences & pad
    X_val_seq = tokenizer.texts_to_sequences(X_val)
    X_val_padded = pad_sequences(X_val_seq, maxlen=100, padding='post', truncating='post')

    # Get predictions
    y_pred_lstm_prob = lstm_model.predict(X_val_padded)
    y_pred_lstm = np.argmax(y_pred_lstm_prob, axis=-1)

    lstm_report = classification_report(
        y_val, 
        y_pred_lstm, 
        target_names=['negative', 'neutral', 'positive'], 
        output_dict=True
    )
    lstm_report_str = classification_report(
        y_val, 
        y_pred_lstm, 
        target_names=['negative', 'neutral', 'positive']
    )

    # ---------------------------------------------
    # Print Metrics Tables to Console
    # ---------------------------------------------
    print("\n" + "="*50)
    print("METRICS PERFORMANCE COMPARISON")
    print("="*50)
    print("\n--- 1. NAIVE BAYES BASELINE ---")
    print(nb_report_str)
    
    print("\n--- 2. LSTM NEURAL NETWORK ---")
    print(lstm_report_str)
    print("="*50 + "\n")

    # ---------------------------------------------
    # Generate and Save Confusion Matrix Plot
    # ---------------------------------------------
    print(f"Generating confusion matrices plot at: {confusion_img_path}")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ConfusionMatrixDisplay.from_predictions(
        y_val, 
        y_pred_nb, 
        display_labels=['negative', 'neutral', 'positive'], 
        ax=axes[0], 
        cmap='Blues',
        values_format='d'
    )
    axes[0].set_title('Naive Bayes Confusion Matrix', fontsize=14, pad=10)

    ConfusionMatrixDisplay.from_predictions(
        y_val, 
        y_pred_lstm, 
        display_labels=['negative', 'neutral', 'positive'], 
        ax=axes[1], 
        cmap='Oranges',
        values_format='d'
    )
    axes[1].set_title('LSTM Confusion Matrix', fontsize=14, pad=10)

    plt.tight_layout()
    plt.savefig(confusion_img_path, dpi=150)
    plt.close()
    print("Performance evaluation complete.")

if __name__ == "__main__":
    main()
