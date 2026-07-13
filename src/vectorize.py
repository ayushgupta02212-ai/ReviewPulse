import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec

def main():
    processed_path = os.path.join("data", "processed_reviews.csv")
    tfidf_pkl_path = os.path.join("models", "tfidf_vectorizer.pkl")
    w2v_model_path = os.path.join("models", "word2vec.model")

    # Make sure output directory models/ exists
    os.makedirs("models", exist_ok=True)

    print(f"Loading processed dataset from: {processed_path}")
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"Processed file not found at {processed_path}")

    df = pd.read_csv(processed_path)
    
    # Handle any potential NaNs in cleaned_body
    texts = df['cleaned_body'].fillna("").astype(str)

    print("Fitting CountVectorizer (Bag-of-Words)...")
    bow_vectorizer = CountVectorizer(max_features=5000)
    bow_matrix = bow_vectorizer.fit_transform(texts)

    print("Fitting TfidfVectorizer (TF-IDF)...")
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

    # Save the TF-IDF vectorizer
    print(f"Saving TF-IDF Vectorizer to: {tfidf_pkl_path}")
    with open(tfidf_pkl_path, "wb") as f:
        pickle.dump(tfidf_vectorizer, f)

    # Implement and train Word2Vec
    print("Training Word2Vec model...")
    # Tokenize by splitting cleaned_body text on spaces
    tokenized_sentences = [text.split() for text in texts]
    
    # Train Word2Vec model: vector_size=100
    w2v_model = Word2Vec(
        sentences=tokenized_sentences,
        vector_size=100,
        window=5,
        min_count=1,
        workers=4,
        epochs=10
    )

    # Save Word2Vec model
    print(f"Saving Word2Vec model to: {w2v_model_path}")
    w2v_model.save(w2v_model_path)

    # Print structural shapes
    print("\n" + "="*40)
    print("VECTORIZATION SHAPES")
    print("="*40)
    print(f"BoW Matrix Shape:    {bow_matrix.shape}")
    print(f"TF-IDF Matrix Shape: {tfidf_matrix.shape}")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()
