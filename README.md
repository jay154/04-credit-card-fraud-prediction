# Credit Card Fraud Prediction

An end-to-end machine learning project that takes a raw JSON dataset of credit
card transactions and builds a model that flags likely fraud. The pipeline runs
across three Jupyter notebooks: exploratory data analysis, data wrangling
(finding reversed and multi-swipe duplicate transactions), and a baseline
classification model for the `isFraud` label. A small Streamlit app loads the
trained model and shows the predicted fraud probability for a sample or
hand-entered transaction. This repository is built MVP-first: the goal is a
correct, reproducible pipeline from download to deployment before any tuning or
polish.

## Status

Work in progress. Built in phases; see `fraud_mvp_claude_code_brief.md` for the
plan. A fuller README (setup, results, screenshots) lands in the final phase.

## Quick start

```
py -3 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

Run the notebooks in order: `notebooks/01_eda.ipynb`,
`notebooks/02_data_wrangling.ipynb`, `notebooks/03_modeling.ipynb`. The first
notebook downloads and extracts the dataset into `data/raw/`, which is
gitignored, so the data is reproduced from code rather than committed.
