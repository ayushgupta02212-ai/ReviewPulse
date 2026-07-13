import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from gensim.models import Word2Vec

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

def main():
    processed_path = os.path.join("data", "processed_reviews.csv")
    tfidf_vectorizer_path = os.path.join("models", "tfidf_vectorizer.pkl")
    w2v_model_path = os.path.join("models", "word2vec.model")
    
    nb_model_path = os.path.join("models", "naive_bayes_model.pkl")
    lstm_model_path = os.path.join("models", "lstm_model.keras")
    lstm_tokenizer_path = os.path.join("models", "lstm_tokenizer.pkl")

    print("Loading processed reviews...")
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"Processed dataset not found at {processed_path}")
    
    df = pd.read_csv(processed_path)
    df['cleaned_body'] = df['cleaned_body'].fillna("").astype(str)

    # Map sentiments to integers: negative=0, neutral=1, positive=2
    sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    df['label'] = df['sentiment'].map(sentiment_map)

    X = df['cleaned_body'].astype(str).tolist()
    y = df['label'].astype(int).tolist()

    # Train/Validation Split (80% train, 20% val)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Convert labels to numpy arrays for Keras compatibility
    y_train = np.array(y_train)
    y_val = np.array(y_val)
    print(f"Train set size: {len(X_train)}, Validation set size: {len(X_val)}")

    # ==========================================
    # 1. Naive Bayes (Baseline)
    # ==========================================
    print("\n--- Training Naive Bayes Baseline ---")
    print(f"Loading TF-IDF vectorizer from: {tfidf_vectorizer_path}")
    if not os.path.exists(tfidf_vectorizer_path):
        raise FileNotFoundError(f"TF-IDF Vectorizer not found at {tfidf_vectorizer_path}")
        
    with open(tfidf_vectorizer_path, "rb") as f:
        tfidf_vectorizer = pickle.load(f)

    # Transform texts
    X_train_tfidf = tfidf_vectorizer.transform(X_train)
    X_val_tfidf = tfidf_vectorizer.transform(X_val)

    # Fit MultinomialNB
    nb_classifier = MultinomialNB()
    nb_classifier.fit(X_train_tfidf, y_train)

    # Evaluate Naive Bayes on validation set
    y_pred_nb = nb_classifier.predict(X_val_tfidf)
    print("Naive Bayes Classification Report (Validation):")
    print(classification_report(y_val, y_pred_nb, target_names=['negative', 'neutral', 'positive']))

    # Save Naive Bayes Model
    print(f"Saving Naive Bayes baseline model to: {nb_model_path}")
    with open(nb_model_path, "wb") as f:
        pickle.dump(nb_classifier, f)

    # ==========================================
    # 2. LSTM Classifier (Keras/TensorFlow)
    # ==========================================
    print("\n--- Training LSTM Neural Network ---")
    
    # Tokenize text for LSTM
    max_vocab_size = 15000
    max_sequence_len = 100
    
    tokenizer = Tokenizer(num_words=max_vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)
    
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_val_seq = tokenizer.texts_to_sequences(X_val)

    X_train_padded = pad_sequences(X_train_seq, maxlen=max_sequence_len, padding='post', truncating='post')
    X_val_padded = pad_sequences(X_val_seq, maxlen=max_sequence_len, padding='post', truncating='post')

    # Save tokenizer
    print(f"Saving LSTM Tokenizer to: {lstm_tokenizer_path}")
    with open(lstm_tokenizer_path, "wb") as f:
        pickle.dump(tokenizer, f)

    # Load Word2Vec model to initialize embedding layer weights
    print(f"Loading Word2Vec model from: {w2v_model_path}")
    embedding_dim = 100
    word_index = tokenizer.word_index
    vocab_size = min(max_vocab_size, len(word_index) + 1)
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    if os.path.exists(w2v_model_path):
        w2v_model = Word2Vec.load(w2v_model_path)
        for word, i in word_index.items():
            if i >= max_vocab_size:
                continue
            if word in w2v_model.wv:
                embedding_matrix[i] = w2v_model.wv[word]
        print("Loaded Word2Vec embeddings into embedding matrix.")
    else:
        print("Word2Vec model not found. Initializing random embeddings.")
        embedding_matrix = None

    # Build LSTM Model Architecture
    model = Sequential([
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            weights=[embedding_matrix] if embedding_matrix is not None else None,
            input_length=max_sequence_len,
            trainable=True  # Fine-tune the embeddings during training
        ),
        Dropout(0.2),
        LSTM(64, dropout=0.2, recurrent_dropout=0.2),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(3, activation='softmax')  # 3 output units: negative, neutral, positive
    ])

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    model.summary()

    # Train model
    print("Training LSTM network...")
    model.fit(
        X_train_padded,
        y_train,
        validation_data=(X_val_padded, y_val),
        epochs=5,
        batch_size=64,
        verbose=1
    )

    # Save LSTM model
    print(f"Saving finalized LSTM model to: {lstm_model_path}")
    model.save(lstm_model_path)
    print("Model training pipeline finished successfully.")

if __name__ == "__main__":
    main()
