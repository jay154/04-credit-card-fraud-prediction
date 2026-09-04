"""MVP Streamlit demo for the credit card fraud model.

Run with:

    streamlit run app.py

Loads models/fraud_model_v1.pkl (built by notebooks/03_modeling.ipynb), lets you
either pick a real transaction from the dataset or type one in by hand, and shows
the model's predicted fraud probability and whether it clears the saved decision
threshold.
"""
from __future__ import annotations

from datetime import date, datetime, time
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from features import build_features

REPO_ROOT = Path(__file__).parent
MODEL_PATH = REPO_ROOT / "models" / "fraud_model_v1.pkl"
PROCESSED_PATH = REPO_ROOT / "data" / "processed" / "transactions_deduped.parquet"

# Category options, so the manual form works even without the dataset on disk.
CATEGORY_OPTIONS = {
    "transactionType": ["PURCHASE", "REVERSAL", "ADDRESS_VERIFICATION"],
    "merchantCategoryCode": [
        "airline", "auto", "cable/phone", "entertainment", "fastfood", "food",
        "food_delivery", "fuel", "furniture", "gym", "health", "hotels",
        "mobileapps", "online_gifts", "online_retail", "online_subscriptions",
        "personal care", "rideshare", "subscriptions",
    ],
    "posEntryMode": ["02", "05", "09", "80", "90"],
    "posConditionCode": ["01", "08", "99"],
    "acqCountry": ["US", "CAN", "MEX", "PR"],
    "merchantCountryCode": ["US", "CAN", "MEX", "PR"],
}
CREDIT_LIMITS = [250, 500, 1000, 2500, 5000, 7500, 10000, 15000, 20000, 50000]


@st.cache_resource
def load_model() -> dict:
    if not MODEL_PATH.exists():
        st.error(
            f"Model file not found at {MODEL_PATH.relative_to(REPO_ROOT)}. "
            "Run notebooks/03_modeling.ipynb first."
        )
        st.stop()
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_samples(n: int = 300) -> pd.DataFrame | None:
    if not PROCESSED_PATH.exists():
        return None
    df = pd.read_parquet(PROCESSED_PATH)
    # A small mix that is mostly legit with some fraud, so the picker shows both.
    fraud = df[df["isFraud"]].sample(min(n // 3, int(df["isFraud"].sum())), random_state=0)
    legit = df[~df["isFraud"]].sample(n - len(fraud), random_state=0)
    return pd.concat([fraud, legit]).sample(frac=1, random_state=0).reset_index(drop=True)


def predict(bundle: dict, raw_row: dict) -> tuple[float, bool]:
    features = build_features(pd.DataFrame([raw_row]))
    proba = float(bundle["pipeline"].predict_proba(features)[0, 1])
    return proba, proba >= bundle["threshold"]


def show_result(proba: float, flagged: bool, threshold: float) -> None:
    left, right = st.columns(2)
    left.metric("Predicted fraud probability", f"{proba:.1%}")
    right.metric("Decision threshold", f"{threshold:.1%}")
    st.progress(min(proba, 1.0))
    if flagged:
        st.error("Flagged for review: probability is at or above the threshold.")
    else:
        st.success("Not flagged: probability is below the threshold.")
    st.caption(
        "This is an MVP baseline. At this threshold the model catches roughly a "
        "fifth of actual fraud and produces many false alarms, so a flag means "
        "'worth a look', not 'confirmed fraud'."
    )


st.set_page_config(page_title="Credit Card Fraud Demo", page_icon="💳")
st.title("Credit Card Fraud Prediction — MVP demo")
st.write(
    "Scores a single transaction with the baseline XGBoost model from "
    "`notebooks/03_modeling.ipynb`."
)

bundle = load_model()
threshold = bundle["threshold"]

with st.expander("About this model"):
    m = bundle.get("metrics_on_test", {})
    st.write(bundle.get("model", "XGBoost baseline"))
    st.write(
        f"Test-set PR-AUC **{m.get('pr_auc', float('nan')):.3f}** "
        f"(no-skill {m.get('no_skill_pr_auc', float('nan')):.3f}), "
        f"ROC-AUC **{m.get('roc_auc', float('nan')):.3f}**. "
        f"At the saved threshold ({threshold:.1%}): precision "
        f"**{m.get('precision', float('nan')):.2f}**, recall "
        f"**{m.get('recall', float('nan')):.2f}**."
    )
    st.write(f"Trained on {bundle.get('trained_rows', 'n/a'):,} transactions.")

mode = st.radio("Input", ["Pick a sample transaction", "Enter values manually"], horizontal=True)
samples = load_samples()

if mode == "Pick a sample transaction":
    if samples is None:
        st.info(
            "Sample data not found. Run notebooks 01 and 02 to build "
            "`data/processed/transactions_deduped.parquet`, or use manual entry."
        )
    else:
        labels = [
            f"#{i} — ${r.transactionAmount:,.2f} at {r.merchantName} "
            f"({r.merchantCategoryCode}, {'card present' if r.cardPresent else 'card not present'})"
            for i, r in samples.iterrows()
        ]
        choice = st.selectbox("Transaction", range(len(samples)), format_func=lambda i: labels[i])
        row = samples.loc[choice]

        c1, c2, c3 = st.columns(3)
        c1.write(f"**Amount** ${row.transactionAmount:,.2f}")
        c1.write(f"**Type** {row.transactionType}")
        c2.write(f"**Merchant** {row.merchantName}")
        c2.write(f"**Category** {row.merchantCategoryCode}")
        c3.write(f"**POS entry** {row.posEntryMode or 'blank'}")
        c3.write(f"**Card present** {bool(row.cardPresent)}")

        raw_row = row.to_dict()
        proba, flagged = predict(bundle, raw_row)
        show_result(proba, flagged, threshold)
        actual = "fraud" if bool(row.isFraud) else "legitimate"
        st.caption(f"Actual label recorded in the data: **{actual}**.")

else:
    with st.form("manual"):
        c1, c2 = st.columns(2)
        with c1:
            amount = st.number_input("Transaction amount ($)", 0.0, 10000.0, 100.0, step=10.0)
            credit_limit = st.select_slider("Credit limit ($)", CREDIT_LIMITS, value=5000)
            available_money = st.number_input("Available money ($)", -2000.0, 50000.0, 3000.0, step=100.0)
            current_balance = st.number_input("Current balance ($)", 0.0, 50000.0, 500.0, step=100.0)
            txn_date = st.date_input("Transaction date", date(2016, 6, 15),
                                     min_value=date(2016, 1, 1), max_value=date(2016, 12, 31))
            txn_hour = st.slider("Transaction hour", 0, 23, 14)
        with c2:
            category = st.selectbox("Merchant category", CATEGORY_OPTIONS["merchantCategoryCode"],
                                    index=CATEGORY_OPTIONS["merchantCategoryCode"].index("online_retail"))
            txn_type = st.selectbox("Transaction type", CATEGORY_OPTIONS["transactionType"])
            pos_entry = st.selectbox("POS entry mode", CATEGORY_OPTIONS["posEntryMode"], index=1)
            pos_condition = st.selectbox("POS condition code", CATEGORY_OPTIONS["posConditionCode"])
            acq_country = st.selectbox("Acquirer country", CATEGORY_OPTIONS["acqCountry"])
            merch_country = st.selectbox("Merchant country", CATEGORY_OPTIONS["merchantCountryCode"])
            account_open = st.date_input("Account opened", date(2015, 1, 1),
                                         min_value=date(1989, 1, 1), max_value=date(2016, 12, 31))
            last_addr_change = st.date_input("Last address change", date(2015, 1, 1),
                                             min_value=date(1989, 1, 1), max_value=date(2016, 12, 31))

        card_present = st.checkbox("Card present (chip or swipe)", value=False)
        cvv_match = st.checkbox("Entered CVV matches the card", value=True)
        exp_key_match = st.checkbox("Expiration date key-in match", value=False)
        submitted = st.form_submit_button("Score transaction")

    if submitted:
        raw_row = {
            "transactionDateTime": datetime.combine(txn_date, time(hour=txn_hour)).isoformat(),
            "accountOpenDate": account_open.isoformat(),
            "dateOfLastAddressChange": last_addr_change.isoformat(),
            "transactionAmount": amount,
            "creditLimit": credit_limit,
            "availableMoney": available_money,
            "currentBalance": current_balance,
            "cardCVV": 123,
            "enteredCVV": 123 if cvv_match else 999,
            "transactionType": txn_type,
            "merchantCategoryCode": category,
            "posEntryMode": pos_entry,
            "posConditionCode": pos_condition,
            "acqCountry": acq_country,
            "merchantCountryCode": merch_country,
            "cardPresent": card_present,
            "expirationDateKeyInMatch": exp_key_match,
        }
        proba, flagged = predict(bundle, raw_row)
        show_result(proba, flagged, threshold)
