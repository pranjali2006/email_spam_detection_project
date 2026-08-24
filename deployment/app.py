import streamlit as st
import pickle
from pathlib import Path

from preprocessing import preprocess_message

MODEL_PATH = Path(__file__).resolve().parent / "spam_model.pkl"

with MODEL_PATH.open("rb") as f:
    model= pickle.load(f)

st.title("email spam detection")
message=st.text_area("enter your email message : ")

if st.button("Predict"):
    processed_message = preprocess_message(message)
    prediction=model.predict([processed_message])[0]
    probability=model.predict_proba([processed_message])[0]

    if prediction == 1:
        st.error("spam email")
    else:
        st.success("safe email")

    st.write(f"safe probability : {probability[0]*100:.2f}%")
    st.write(f"spam probability : {probability[1]*100:.2f}%")



