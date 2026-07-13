import os
import pandas as pd

def main():
    processed_path = os.path.join("data", "processed_reviews.csv")
    report_path = os.path.join("data", "eda_summary.md")

    print(f"Loading processed dataset from: {processed_path}")
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"Processed reviews file not found at {processed_path}")

    df = pd.read_csv(processed_path)

    # Class balance (distribution of sentiment)
    sentiment_counts = df['sentiment'].value_counts()
    sentiment_pcts = df['sentiment'].value_counts(normalize=True) * 100

    # Average word count
    avg_words_before = df['word_count_before'].mean()
    avg_words_after = df['word_count_after'].mean()

    # Generate Markdown content
    md_content = f"""# ReviewPulse: Preprocessing & EDA Report

## Sentiment Distribution (Class Balance)
| Sentiment | Count | Percentage |
|-----------|-------|------------|
| Positive  | {sentiment_counts.get('positive', 0)} | {sentiment_pcts.get('positive', 0.0):.2f}% |
| Neutral   | {sentiment_counts.get('neutral', 0)} | {sentiment_pcts.get('neutral', 0.0):.2f}% |
| Negative  | {sentiment_counts.get('negative', 0)} | {sentiment_pcts.get('negative', 0.0):.2f}% |

## Word Count Analysis
- **Average Word Count (Before Cleaning)**: {avg_words_before:.2f} words
- **Average Word Count (After Cleaning)**: {avg_words_after:.2f} words
- **Reduction Rate**: {(1 - (avg_words_after / avg_words_before)) * 100:.2f}%
"""

    # Print to console
    print("\n" + "="*40)
    print("EDA REPORT SUMMARY")
    print("="*40)
    print(md_content)
    print("="*40 + "\n")

    # Save as markdown note
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"EDA report saved as markdown note to: {report_path}")

if __name__ == "__main__":
    main()
