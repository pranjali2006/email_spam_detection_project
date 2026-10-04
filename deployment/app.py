import streamlit as st
import pickle
from pathlib import Path

from preprocessing import preprocess_message

MODEL_PATH = Path(__file__).resolve().parent / "spam_model.pkl"

# Load saved model and threshold
with MODEL_PATH.open("rb") as f:
    model_data = pickle.load(f)

model = model_data["pipeline"]
threshold = model_data["threshold"]


st.title("Email Spam Detection")

message = st.text_area("Enter your email message:")

if st.button("Predict"):

    processed_message = preprocess_message(message)

    # Get spam probability
    spam_probability = model.predict_proba([processed_message])[0, 1]

    # Apply final threshold
    if spam_probability >= threshold:
        st.error("Spam email")
    else:
        st.success("Safe email")

    st.write(f"Safe probability: {(1 - spam_probability) * 100:.2f}%")
    st.write(f"Spam probability: {spam_probability * 100:.2f}%")