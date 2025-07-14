"""
Statistical testing of various NLP metrics across LLMs using Friedman and Wilcoxon tests.

- Loads augmented summary analysis results.
- Runs Friedman and post-hoc Wilcoxon tests (with Bonferroni correction) for each metric.
- Outputs results as a DataFrame.

Usage:
    python statistical_testing_metrics.py

Requires:
    - data/raw/summary_analysis_results_augmented.csv
    - pandas, scipy
"""
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from itertools import combinations

# Load the data
df = pd.read_csv("data/raw/llm_metrics_augmented.csv")

# Normalize whitespace in strings
df["Model"] = df["Model"].str.strip()
df["Paper_Title"] = df["Paper_Title"].str.strip()

# Metrics to test
metrics = [
    "Cosine_Similarity", "Sentence_BLEU_Score", "Levenshtein_Distance",
    "ROUGE-1", "ROUGE-2", "ROUGE-L", "METEOR_Score", "BERTScore_(F1)",
    "Generated_Flesch_Reading_Ease", "Generated_Flesch-Kincaid_Grade"
]

stats_df = df.groupby("Model")[metrics].describe().transpose()
stats_df.to_csv("metrics_descriptive_statistics.csv")

print("Descriptive statistics saved to 'metrics_descriptive_statistics.csv'.")

# Filter valid models
valid_models = ["ChatGPT o3 mini", "Gemini (2.0) Flash Experimental", "DeepSeek R1"]
models = valid_models

# Prepare output
results = []

for metric in metrics:
    # Filter for papers that have all models for this metric
    valid_papers = (
        df[df["Model"].isin(models)]
        .groupby("Paper_Title")["Model"]
        .nunique()
    )
    valid_papers = valid_papers[valid_papers == len(models)].index

    df_filtered = df[df["Paper_Title"].isin(valid_papers) & df["Model"].isin(models)]

    # Build the score matrix
    score_matrix = df_filtered.pivot(index="Paper_Title", columns="Model", values=metric)
    score_matrix = score_matrix.dropna()

    print(f"\nMetric: {metric}, Score matrix shape: {score_matrix.shape}")
    print(score_matrix.head())

    if score_matrix.shape[0] >= 2:
        try:
            # Run Friedman test
            stat, p = friedmanchisquare(*[pd.to_numeric(score_matrix[model], errors="coerce") for model in models])
            results.append({
                "Metric": metric,
                "Test": "Friedman",
                "P-Value": p,
                "Significant": p < 0.05
            })

            # Post-hoc pairwise Wilcoxon tests
            if p < 0.05:
                for m1, m2 in combinations(models, 2):
                    x = pd.to_numeric(score_matrix[m1], errors='coerce')
                    y = pd.to_numeric(score_matrix[m2], errors='coerce')

                    valid_pairs = (~x.isna()) & (~y.isna())
                    x_valid = x[valid_pairs]
                    y_valid = y[valid_pairs]

                    if len(x_valid) >= 2:
                        stat_w, p_w = wilcoxon(x_valid, y_valid)
                        p_w_adj = p_w * 3  # Bonferroni correction
                        results.append({
                            "Metric": metric,
                            "Test": f"Wilcoxon {m1} vs {m2}",
                            "P-Value": p_w_adj,
                            "Significant": p_w_adj < 0.05
                        })
                    else:
                        print(f"Not enough valid data for Wilcoxon {m1} vs {m2} on {metric}.")
        except Exception as e:
            print(f"Error running Friedman or Wilcoxon on {metric}: {e}")

# Final output
results_df = pd.DataFrame(results)
print("\nFinal Statistical Results:\n")
print(results_df)

# Optional: Save to CSV
results_df.to_csv("model_statistical_comparison_results.csv", index=False)
