
import streamlit as st
import joblib
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "sentiment_pipeline.joblib")

st.set_page_config(page_title="Sentiment Analyzer", layout="centered")
st.title("Customer Sentiment Analyzer")
st.write("Enter customer review text and get predicted sentiment (positive/negative/neutral).")

if not os.path.exists(MODEL_PATH):
    st.warning("Model not found. Run `python train_model.py` first to train and save the model.")

user_input = st.text_area("Customer review", value="I love this product!", height=150)
if st.button("Analyze"):
    if not user_input.strip():
        st.error("Please enter some text to analyze.")
    else:
        if os.path.exists(MODEL_PATH):
            pipeline = joblib.load(MODEL_PATH)
            pred = pipeline.predict([user_input])[0]
            probs = pipeline.predict_proba([user_input])[0]
            labels = pipeline.classes_
            st.success(f"Predicted sentiment: **{pred}**")
            st.write("Prediction probabilities:")
            prob_df = {labels[i]: float(probs[i]) for i in range(len(labels))}
            st.json(prob_df)
        else:
            st.error("Model not found. Train the model using `python train_model.py`.")
