"""
Statistical analysis of LLM judging scores across domains (Quality, Accuracy, Bias, Relevance).

- Runs Friedman and post-hoc Wilcoxon signed-rank tests (with Bonferroni correction).
- Generates and saves boxplots for each domain.

Usage:
    python statistical_analysis_llm_judging.py

Requires:
    - data/raw/PAPERS.csv
    - pandas, scipy, matplotlib, seaborn
"""
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('data/raw/llm_judging_scores.csv')

# Define the domains and corresponding model column names
domains = {
    'Quality': ['OpenAI Quality', 'Gemini Quality', 'DeepSeek Quality'],
    'Accuracy': ['OpenAI Accuracy', 'Gemini Accuracy', 'DeepSeek Accuracy'],
    'Bias': ['OpenAI Bias', 'Gemini Bias', 'DeepSeek Bias'],
    'Relevance': ['OpenAI relevance', 'Gemini relevance', 'DeepSeek relevance']
}

# Store results
friedman_results = {}
wilcoxon_results = {}

# Run Friedman test for each domain
for domain, cols in domains.items():
    data = [df[col].dropna() for col in cols]
    stat, p = friedmanchisquare(*data)
    friedman_results[domain] = {'Statistic': stat, 'p-value': p}
    print(f"Friedman test for {domain}: statistic = {stat:.4f}, p = {p:.4g}")

    # Post hoc pairwise Wilcoxon signed-rank tests with Bonferroni correction
    if p < 0.05:
        from itertools import combinations
        n_comparisons = len(cols) * (len(cols) - 1) // 2
        for col1, col2 in combinations(cols, 2):
            # Drop NA pairs
            paired = df[[col1, col2]].dropna()
            if len(paired) > 0:
                stat_w, p_w = wilcoxon(paired[col1], paired[col2])
                p_w_adj = min(p_w * n_comparisons, 1.0)  # Bonferroni correction
                print(f"  Wilcoxon {col1} vs {col2}: statistic = {stat_w:.4f}, p (adj) = {p_w_adj:.4g}")
                wilcoxon_results[(domain, col1, col2)] = {'Statistic': stat_w, 'p-value': p_w, 'p-value-adj': p_w_adj}


# Plot boxplots for each domain
for domain, cols in domains.items():
    plt.figure(figsize=(8, 6))
    subset = df[cols].melt(var_name='Model', value_name=domain)
    sns.boxplot(x='Model', y=domain, data=subset)
    sns.swarmplot(x='Model', y=domain, data=subset, color=".25", alpha=0.6)
    plt.title(f'{domain} Scores by Model')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{domain}_model_boxplot.png')
    plt.close()

print("Friedman tests completed and boxplots saved.")
