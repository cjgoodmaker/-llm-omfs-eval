# LLM-OMFS Evaluation

This repository contains the code and data used in the study:  
**"Evaluating the Quality of Large Language Models in Summarising Oral and Maxillofacial Surgery Research Papers"**

## Directory Structure

- `scripts/`: Metric and stats scripts
- `data/raw/` — Raw input data (e.g., human and model scores)
- `data/processed/` — Processed data and statistical results
- `results/`: Summary statistics and figures
- `plots/` — Generated plots and visualizations
- `prompts/`: Rubric used for GPT-4o evaluation
- `statistical_analysis_llm_judging.py` — Statistical analysis of LLM judging (Friedman, Wilcoxon tests)
- `statistical_testing_metrics.py` — Statistical testing of various NLP metrics
- `llm_judge_single_chart.py` — Visualization script for LLM judge metrics

## Setup

1. Clone the repository.
2. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

- Run statistical analysis on LLM judging:
  ```bash
  python statistical_analysis_llm_judging.py
  ```
- Run metric-based statistical testing:
  ```bash
  python statistical_testing_metrics.py
  ```
- Generate visualizations:
  ```bash
  python llm_judge_single_chart.py
  ```

## Citation

If you use this code or data in your research, please cite this repository.

## License

MIT License
