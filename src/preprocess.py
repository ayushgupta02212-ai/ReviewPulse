import os
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def main():
    raw_path = os.path.join("data", "20191226-reviews.csv")
    processed_path = os.path.join("data", "processed_reviews.csv")

    print(f"Loading raw dataset from: {raw_path}")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw reviews file not found at {raw_path}")

    # Load data
    df = pd.read_csv(raw_path)
    print(f"Initial shape: {df.shape}")

    # Drop rows with missing values in the essential columns: 'body' and 'rating'
    df = df.dropna(subset=['body', 'rating'])
    print(f"Shape after dropping missing 'body' or 'rating': {df.shape}")

    # Slice the first 15,000 reviews
    df_subset = df.iloc[:15000].copy()
    print(f"Processing first {len(df_subset)} reviews...")

    # Set up NLTK components
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def clean_text(text):
        if not isinstance(text, str):
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

    # Word count before cleaning
    df_subset['word_count_before'] = df_subset['body'].apply(lambda x: len(str(x).split()))

    # Apply text cleaning
    df_subset['cleaned_body'] = df_subset['body'].apply(clean_text)

    # Word count after cleaning
    df_subset['word_count_after'] = df_subset['cleaned_body'].apply(lambda x: len(str(x).split()))

    # Derive target sentiment label based on ratings
    # Rating >= 4 -> 'positive'
    # Rating == 3 -> 'neutral'
    # Rating <= 2 -> 'negative'
    def get_sentiment(rating):
        if rating >= 4:
            return 'positive'
        elif rating == 3:
            return 'neutral'
        else:
            return 'negative'

    df_subset['sentiment'] = df_subset['rating'].apply(get_sentiment)

    # Save the processed dataset
    df_subset.to_csv(processed_path, index=False)
    print(f"Processed dataset successfully saved to: {processed_path}")

if __name__ == "__main__":
    main()
