import os
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def main():
    processed_path = os.path.join("data", "processed_reviews.csv")
    pos_wc_path = os.path.join("data", "wordcloud_positive.png")
    neg_wc_path = os.path.join("data", "wordcloud_negative.png")

    print(f"Loading processed dataset from: {processed_path}")
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"Processed file not found at {processed_path}")

    df = pd.read_csv(processed_path)
    
    # Handle NaNs in cleaned_body
    df['cleaned_body'] = df['cleaned_body'].fillna("").astype(str)

    # Filter positive and negative reviews
    pos_reviews = df[df['sentiment'] == 'positive']['cleaned_body']
    neg_reviews = df[df['sentiment'] == 'negative']['cleaned_body']

    print(f"Number of Positive Reviews: {len(pos_reviews)}")
    print(f"Number of Negative Reviews: {len(neg_reviews)}")

    # Combine text for word clouds
    pos_text = " ".join(pos_reviews)
    neg_text = " ".join(neg_reviews)

    # Generate and save Positive Word Cloud
    print("Generating Word Cloud for Positive Reviews...")
    if pos_text.strip():
        pos_wc = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='viridis',
            max_words=250
        ).generate(pos_text)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(pos_wc, interpolation='bilinear')
        plt.axis('off')
        plt.title("Word Cloud - Positive Reviews", fontsize=16, pad=10)
        plt.tight_layout(pad=0)
        plt.savefig(pos_wc_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Saved Positive Word Cloud to: {pos_wc_path}")
    else:
        print("Warning: Positive text is empty. Skipping word cloud.")

    # Generate and save Negative Word Cloud
    print("Generating Word Cloud for Negative Reviews...")
    if neg_text.strip():
        neg_wc = WordCloud(
            width=800,
            height=400,
            background_color='black',
            colormap='inferno',
            max_words=250
        ).generate(neg_text)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(neg_wc, interpolation='bilinear')
        plt.axis('off')
        plt.title("Word Cloud - Negative Reviews", fontsize=16, pad=10, color='white')
        # We need a dark background figure for a black-bg word cloud
        fig = plt.gcf()
        fig.patch.set_facecolor('black')
        plt.tight_layout(pad=0)
        plt.savefig(neg_wc_path, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Saved Negative Word Cloud to: {neg_wc_path}")
    else:
        print("Warning: Negative text is empty. Skipping word cloud.")

    print("Word cloud visualization completed.")

if __name__ == "__main__":
    main()
