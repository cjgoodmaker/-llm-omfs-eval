"""
Script for generating a single bar or boxplot visualization of LLM judge metrics.

Usage:
    python llm_judge_single_chart.py

Requires:
    - pandas, matplotlib, seaborn
    - Input data file (update path as needed)
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('data/raw/llm_judging_scores.csv')

# Create a long-form dataframe
melted = pd.melt(
    df,
    value_vars=[
        'OpenAI Quality', 'Gemini Quality', 'DeepSeek Quality',
        'OpenAI Accuracy', 'Gemini Accuracy', 'DeepSeek Accuracy',
        'OpenAI Bias', 'Gemini Bias', 'DeepSeek Bias',
        'OpenAI relevance', 'Gemini relevance', 'DeepSeek relevance'
    ],
    var_name='Model_Metric',
    value_name='Score'
)

# Split into Model and Metric
melted[['Model', 'Metric']] = melted['Model_Metric'].str.extract(r'(OpenAI|Gemini|DeepSeek)\s+(Quality|Accuracy|Bias|relevance)')
melted['Metric'] = melted['Metric'].str.capitalize()

# Calculate mean and std
summary = melted.groupby(['Metric', 'Model'])['Score'].agg(['mean', 'std']).reset_index()

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(data=summary, x='Metric', y='mean', hue='Model', ci=None, capsize=0.1)

# Add error bars manually
for i in range(len(summary)):
    row = summary.iloc[i]
    plt.errorbar(
        x=i//3 + (-0.25 + 0.25*(i%3)),  # x-position offset per bar
        y=row['mean'],
        yerr=row['std'],
        fmt='none',
        c='black',
        capsize=5
    )

plt.ylabel('Mean Score')
plt.title('LLM-as-a-Judge Evaluation: Mean Scores by Metric and Model')
plt.legend(title='Model')
plt.tight_layout()
plt.savefig('LLM_Judge_Metrics_BarChart.png')
plt.show()
