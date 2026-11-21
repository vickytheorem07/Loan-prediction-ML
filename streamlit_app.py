from pathlib import Path
import pickle

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Train_model" / "model_bundle.pkl"

st.set_page_config(page_title="Loan Payback Prediction", page_icon="💳", layout="wide")
st.title("Loan Payback Prediction")
st.write("Enter the applicant and loan details to estimate the probability of repayment.")
st.caption("Demo only. This model should not be used by itself to make real lending decisions.")

if not MODEL_PATH.exists():
    st.error("Model bundle not found. Run the notebook through its final model export cell first.")
    st.stop()

with MODEL_PATH.open("rb") as file:
    model_bundle = pickle.load(file)

feature_columns = model_bundle["feature_columns"]
categorical_columns = model_bundle["categorical_columns"]
categorical_set = set(categorical_columns)
feature_defaults = model_bundle.get("feature_defaults", {})

st.subheader("Applicant and loan details")
inputs = {}
with st.form("loan_prediction_form"):
    left, right = st.columns(2)
    for index, column in enumerate(feature_columns):
        container = left if index % 2 == 0 else right
        label = column.replace("_", " ").title()
        with container:
            if column in categorical_set:
                category_index = categorical_columns.index(column)
                choices = [str(value) for value in model_bundle["encoder"].categories_[category_index]]
                default_value = str(feature_defaults.get(column, choices[0]))
                default_index = choices.index(default_value) if default_value in choices else 0
                inputs[column] = st.selectbox(label, choices, index=default_index, key=column)
            else:
                default_value = float(feature_defaults.get(column, 0.0))
                step = 1.0 if column in {
                    "age", "credit_score", "loan_term", "num_of_open_accounts",
                    "delinquency_history", "public_records", "num_of_delinquencies",
                } else 0.01
                inputs[column] = st.number_input(
                    label,
                    value=default_value,
                    step=step,
                    format="%.4f",
                    key=column,
                )
    submitted = st.form_submit_button("Predict repayment")

if submitted:
    input_frame = pd.DataFrame([inputs]).reindex(columns=feature_columns)
    encoded_categories = pd.DataFrame(
        model_bundle["encoder"].transform(input_frame[categorical_columns]),
        columns=categorical_columns,
        index=input_frame.index,
    )
    processed = input_frame.drop(columns=categorical_columns).join(encoded_categories)
    processed = processed.reindex(columns=feature_columns)
    scaled = model_bundle["scaler"].transform(processed)

    model = model_bundle["model"]
    probabilities = model.predict_proba(scaled)[0]
    class_one_index = list(model.classes_).index(1)
    probability_repaid = float(probabilities[class_one_index])
    threshold = float(model_bundle["threshold"])
    prediction = "Repaid" if probability_repaid >= threshold else "Not repaid"

    st.subheader("Prediction")
    st.metric("Estimated probability of repayment", f"{probability_repaid:.1%}")
    st.write(f"**Predicted outcome:** {prediction}")
    st.write(f"**Classification threshold:** {threshold:.2f}")