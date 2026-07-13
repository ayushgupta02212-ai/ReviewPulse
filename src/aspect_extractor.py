import os
import pandas as pd
import spacy

def main():
    processed_path = os.path.join("data", "processed_reviews.csv")
    output_csv = os.path.join("data", "aspect_insights.csv")

    print("Loading processed reviews dataset...")
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"Processed dataset not found at {processed_path}")

    df = pd.read_csv(processed_path)
    df['body'] = df['body'].fillna("").astype(str)
    
    # Target aspects list
    target_aspects = {'battery', 'screen', 'camera', 'price', 'software', 'sound'}

    # Initialize sentiment counts
    aspect_counts = {aspect: {'positive': 0, 'negative': 0} for aspect in target_aspects}

    # Custom sentiment lexicon for adjectives
    pos_words = {
        'good', 'great', 'excellent', 'amazing', 'nice', 'wonderful', 'perfect', 
        'fantastic', 'awesome', 'decent', 'beautiful', 'clear', 'bright', 'fast', 
        'easy', 'satisfied', 'love', 'happy', 'outstanding', 'crisp', 'smooth'
    }
    neg_words = {
        'bad', 'poor', 'terrible', 'worst', 'horrible', 'slow', 'cheap', 'broken', 
        'waste', 'disappointed', 'disappointing', 'difficult', 'useless', 'faulty', 
        'scratched', 'dead', 'defective', 'expensive', 'blurry', 'weak', 'laggy'
    }

    def get_adj_sentiment(adj_text, overall_sentiment):
        adj_lower = adj_text.lower()
        if adj_lower in pos_words:
            return 'positive'
        elif adj_lower in neg_words:
            return 'negative'
        else:
            # Fallback to overall review sentiment (positive / negative / neutral)
            return overall_sentiment

    # Load spaCy pipeline, disable NER and textcat for performance
    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])

    print("Running aspect extraction pipeline...")
    bodies = df['body'].tolist()
    sentiments = df['sentiment'].tolist()

    # Process documents in batches for maximum efficiency
    doc_count = 0
    for doc, overall_sent in zip(nlp.pipe(bodies, batch_size=512), sentiments):
        doc_count += 1
        if doc_count % 3000 == 0:
            print(f"Processed {doc_count} reviews...")

        for token in doc:
            lemma = token.lemma_.lower()
            if lemma in target_aspects:
                # 1. Direct adjectival modifier (amod)
                for child in token.children:
                    if child.pos_ == "ADJ" and child.dep_ == "amod":
                        adj_sent = get_adj_sentiment(child.text, overall_sent)
                        if adj_sent in ['positive', 'negative']:
                            aspect_counts[lemma][adj_sent] += 1
                
                # 2. Copular structures (e.g., "the screen is amazing")
                if token.dep_ == "nsubj" and token.head.lemma_ == "be":
                    for sibling in token.head.children:
                        if sibling.pos_ == "ADJ" and sibling.dep_ in ["acomp", "attr"]:
                            adj_sent = get_adj_sentiment(sibling.text, overall_sent)
                            if adj_sent in ['positive', 'negative']:
                                aspect_counts[lemma][adj_sent] += 1

    # Format findings as a DataFrame
    results = []
    for aspect, counts in aspect_counts.items():
        results.append({
            'aspect': aspect,
            'positive': counts['positive'],
            'negative': counts['negative']
        })
    results_df = pd.DataFrame(results)

    # Save to CSV
    print(f"Saving aspect insights to: {output_csv}")
    results_df.to_csv(output_csv, index=False)

    # Print summary table to console
    print("\n" + "="*40)
    print("AGGREGATED ASPECT SENTIMENT COUNTS")
    print("="*40)
    print(results_df.to_string(index=False))
    print("="*40 + "\n")

if __name__ == "__main__":
    main()
