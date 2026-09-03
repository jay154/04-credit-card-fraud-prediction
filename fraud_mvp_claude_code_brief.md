# Fraud Detection MVP — Build Brief for Claude Code

Paste this whole document into Claude Code as the opening message for the project.
It replaces the four standalone prompts — same intent, but scoped for an agent
working across multiple turns in one repo, with the ambiguous bits pinned down.

## Ground rules (apply to every phase below)

- **Working directory:** do all of this inside the existing folder
  `C:\Users\User\Downloads\ErdosBootcamp\git\pyquantnews\GettingStartedWithPythonForQuantFinance\projects\04-credit-card-fraud-prediction`
  — `cd` there first and treat it as the project root (everything under "Repo
  layout" below lives directly inside it, not in a new folder). That path sits
  inside what looks like an existing cloned repo (the pyquantnews bootcamp
  course repo) — **before running `git init`, check `git status` /
  `git rev-parse --show-toplevel` from that folder and tell me what you find.**
  If it's already tracked by the course repo's `.git`, confirm with me whether
  to (a) `git init` a fresh, independent repo right in this subfolder anyway
  (so it gets its own history and its own `jay154` remote, separate from the
  course repo), or (b) add these files into the existing repo instead — don't
  assume either way.
- **Goal:** a Data Scientist II portfolio project on GitHub — a Jupyter notebook
  pipeline on a JSON credit-card transaction dataset, ending in fraud
  classification and a deployed Streamlit demo.
- **MVP-first:** get a working end-to-end pipeline before a polished one.
  Default hyperparameters, simple imputation, no tuning yet — mediocre metrics
  are fine as long as the concepts (EDA → wrangling → modeling → evaluation →
  deploy) are all present and correct. Leave `# TODO(refine):` comments where a
  later pass should improve something, instead of stopping to perfect it now.
- **Git discipline:** one commit per phase, made only after the phase's code
  runs end-to-end without errors (execute the notebook top-to-bottom, e.g. via
  `jupyter nbconvert --execute` or `papermill`, before staging). Use
  [Conventional Commits](https://www.conventionalcommits.org/) messages (see
  suggested ones per phase below — adjust if the actual diff differs).
- **Pause between phases:** after each phase, summarize what was built and
  committed, then stop and wait for my go-ahead before starting the next phase.
- **Repo layout** (directly inside the working directory above, no extra
  nesting folder):
  ```
  ├── data/
  │   ├── raw/              # original JSON dataset (gitignored — see Phase 0)
  │   └── processed/        # deduped/cleaned output from Phase 2
  ├── notebooks/
  │   ├── 01_eda.ipynb
  │   ├── 02_data_wrangling.ipynb
  │   └── 03_modeling.ipynb
  ├── models/
  │   └── fraud_model_v1.pkl
  ├── reports/figures/       # exported charts from Phase 4
  ├── app.py                 # Streamlit MVP
  ├── requirements.txt
  ├── .gitignore
  └── README.md
  ```

---

## Phase 0 — Environment & repo setup

1. Confirm the git situation per the "Working directory" ground rule above
   first, and only proceed once that's resolved.
2. `git init` (if that's what we decided), create `.gitignore` (Python +
   Jupyter checkpoints + `data/raw/*` + venv folder), create a venv, and a
   `requirements.txt` pinned to: `pandas`, `numpy`, `seaborn`, `matplotlib`,
   `scikit-learn`, `imbalanced-learn`, `xgboost`, `streamlit`, `jupyter`.
3. Stub `README.md` with a one-paragraph project description (fill it in
   more at the end, in Phase 5).
4. Commit: `chore: initialize repo, environment, and project scaffolding`

## Phase 1 — Dataset acquisition & EDA

**Data source — do this programmatically, not by hand-downloading:**

1. Download `https://github.com/CapitalOneRecruiting/DS/raw/master/transactions.zip`
   (Python `requests`/`urllib`, not `wget` in a notebook cell) into `data/raw/`.
2. Unzip it there with `zipfile` and list what's inside before assuming a
   filename — the repo doesn't document the archive's contents, so print the
   extracted file name(s) first.
3. Load the extracted file into a DataFrame defensively: try `json.load` (a
   single JSON array) first, and if that fails fall back to
   `pd.read_json(path, lines=True)` (newline-delimited JSON), since public
   write-ups of this exact dataset disagree on which it is. Whichever works,
   note it in a Markdown cell so it's documented for anyone re-running this.
4. `data/raw/` is already gitignored (Phase 0) — the download/extract code is
   what's committed, not the data itself, so the notebook must be able to
   reproduce the dataset from nothing but that code.

Then, in `notebooks/01_eda.ipynb`, with `pandas` and `seaborn`:

1. Show shape, record count, dtypes, and a sample of rows.
2. Summary stats: null counts per column, min/max for numeric fields, unique
   value counts for categorical fields (flag any that look like IDs vs. true
   categories).
3. A detailed histogram of `transactionAmount` (consider a log-scale variant
   too, since transaction amounts are typically right-skewed).
4. Markdown cells explaining each finding for a non-technical stakeholder —
   plain language, no jargon.

Commit: `docs(eda): add exploratory data analysis notebook`

## Phase 2 — Data wrangling: duplicate transaction detection

In `notebooks/02_data_wrangling.ipynb`, detect two duplicate patterns. Use
these working definitions (adjust if the data doesn't support them, and note
the change):

- **Reversed transactions:** a `PURCHASE` matched by a later `REVERSAL` on the
  same account, merchant, and amount — these should net out of revenue.
- **Multi-swipe transactions:** 2+ transactions with identical account,
  merchant, and amount within a short time window (start with 5 minutes) —
  typically a POS retry error, not a second legitimate purchase.

For each type, compute count and total dollar amount, comment the matching
logic clearly, explain the business impact in Markdown, and write the deduped
dataset to `data/processed/`.

Commit: `feat(pipeline): add duplicate transaction detection logic`

## Phase 3 — Baseline classification model

In `notebooks/03_modeling.ipynb`, build a classifier for `isFraud`:

1. Stratified train/test split.
2. Baseline model: Random Forest or XGBoost with default params (this is the
   MVP — no grid search yet).
3. Handle class imbalance (fraud is rare) with `class_weight='balanced'` or
   SMOTE from `imbalanced-learn` — pick one, note the other as a
   `# TODO(refine):` for later comparison.
4. Evaluate with precision, recall, F1, and PR-AUC — **not** accuracy, which
   is misleading on imbalanced data.
5. Markdown explaining the algorithm choice and which features mattered most,
   in plain terms.
6. Save the trained model to `models/fraud_model_v1.pkl` (needed by the
   Streamlit app in Phase 4).

Commit: `feat(model): train baseline fraud classification model`

## Phase 4 — MVP Streamlit deployment

Build `app.py`: load `models/fraud_model_v1.pkl`, let the user pick a sample
transaction (or fill a small input form) and show the predicted fraud
probability. Keep it minimal — this is the MVP deploy step, not the polished
one. Run `streamlit run app.py` locally to confirm it works before committing.
(Pushing to Streamlit Community Cloud is a stretch goal for a later
refinement pass, not required now.)

Commit: `feat(app): add MVP Streamlit fraud-prediction demo`

## Phase 5 — Visual reporting, business hand-off, final push

1. Impactful visualizations of model performance: feature importance chart,
   confusion matrix, PR curve — save to `reports/figures/`.
2. Markdown business summary translating results for a non-technical reader.
3. "Methods considered" and "Future Work" sections (this is where SMOTE vs.
   class-weighting, hyperparameter tuning, and cloud deployment can be
   flagged as next steps rather than done now).
4. Finish `README.md` properly.
5. Final commit, then push:
   ```
   git remote add origin https://github.com/jay154/<repo-name>.git
   git branch -M main
   git push -u origin main
   ```

Commit: `docs(report): add visual reporting, business summary, and final push`

---

## Fill in before sending to Claude Code

- **Repo name** — the last path segment (`04-credit-card-fraud-prediction`) is
  a reasonable default for `<repo-name>` below, but say so explicitly if you
  want a different GitHub repo name. Also confirm whether
  `https://github.com/jay154/<repo-name>` already exists on GitHub or needs
  `gh repo create <repo-name> --public` first.

## After the MVP lands

Once Phases 0–5 are committed and pushed, come back for a refinement pass:
tune hyperparameters, compare SMOTE vs. class-weighting properly, add
cross-validation, tidy notebook narration, and (optionally) deploy the
Streamlit app to the cloud.
