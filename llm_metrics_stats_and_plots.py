"""
Script for descriptive statistics, statistical significance testing, and visualization of LLM evaluation metrics.

- Computes descriptive statistics for selected models and metrics.
- Performs Friedman and post-hoc Wilcoxon tests (with Bonferroni correction).
- Saves results to CSV files.
- Generates box and swarm plots for each metric.

Usage:
    python stats3.py

Requires:
    - summary_analysis_results_augmented.csv in data/raw/ (update path if needed)
    - pandas, seaborn, matplotlib, scipy
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, wilcoxon
from itertools import combinations

# Load the analysis results
results_df = pd.read_csv("data/raw/llm_metrics_augmented.csv")

# Filter to only use the three desired models
models_to_keep = ["ChatGPT o3 mini", "DeepSeek R1", "Gemini (2.0) Flash Experimental"]
results_df = results_df[results_df["Model"].isin(models_to_keep)]

# Metrics to analyze
metrics = [
    "Cosine_Similarity", "Sentence_BLEU_Score", "Levenshtein_Distance",
    "ROUGE-1", "ROUGE-2", "ROUGE-L", "METEOR_Score", "BERTScore_(F1)",
    "Generated_Flesch_Reading_Ease", "Generated_Flesch-Kincaid_Grade"
]

# Compute descriptive statistics
stats_df = results_df.groupby("Model")[metrics].describe().transpose()
stats_df.to_csv("metrics_descriptive_statistics.csv")
print("Descriptive statistics saved to 'metrics_descriptive_statistics.csv'.")

# Statistical significance testing
significance_results = []

for metric in metrics:
    # Create pivot for Friedman test
    pivot = results_df.pivot_table(index="Paper_Title", columns="Model", values=metric)
    pivot = pivot.dropna(subset=models_to_keep)
    
    if pivot.shape[0] >= 2:
        # Friedman test
        stat, p = friedmanchisquare(*[pivot[model] for model in models_to_keep])
        
        # Only run pairwise if Friedman is significant
        if p < 0.05:
            for m1, m2 in combinations(models_to_keep, 2):
                stat_w, p_w = wilcoxon(pivot[m1], pivot[m2])
                p_w_adj = min(p_w * 3, 1.0)  # Bonferroni correction
                significance_results.append({
                    'Metric': metric,
                    'Comparison': f"{m1} vs. {m2}",
                    'Wilcoxon p-value': p_w_adj,
                    'Significant': p_w_adj < 0.05
                })

significance_df = pd.DataFrame(significance_results)
significance_df.to_csv("pairwise_significance_results.csv", index=False)
print("Pairwise significance results saved to 'pairwise_significance_results.csv'.")

# Generate box plots with swarm plots for each metric
for metric in metrics:
    plt.figure(figsize=(12, 8))
    sns.boxplot(x="Model", y=metric, data=results_df, showcaps=True, boxprops={'facecolor': 'None'})
    sns.swarmplot(x="Model", y=metric, data=results_df, color="0.25", alpha=0.7)
    plt.title(f"{metric} Distribution by Model")
    plt.xlabel('Model')
    plt.ylabel(metric)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{metric}_box_swarm.png")
    plt.close()

print("Box plots with swarm overlay for each metric have been saved as PNG files.")
