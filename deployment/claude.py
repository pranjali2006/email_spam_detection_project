import streamlit as st
import pickle
from pathlib import Path

from preprocessing import preprocess_message

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧",
    layout="centered"
)

# ----------------------------
# Load Model
# ----------------------------
MODEL_PATH = Path(__file__).resolve().parent / "spam_model.pkl"

with MODEL_PATH.open("rb") as f:
    model = pickle.load(f)

# ----------------------------
# Header Section
# ----------------------------
st.title("📧 Email Spam Detection")
st.write(
    "This project uses **TF-IDF vectorization** and a **Logistic Regression** "
    "classifier to detect whether an email message is **Spam** or **Safe (Ham)**."
)

st.divider()

# ----------------------------
# Input Section
# ----------------------------
st.subheader("Enter Email Message")
message = st.text_area(
    label="Paste or type the email content below:",
    height=180,
    placeholder="e.g. Congratulations! You've won a free prize, click here to claim now..."
)

predict_clicked = st.button("🔍 Predict", use_container_width=True)

# ----------------------------
# Prediction Section
# ----------------------------
if predict_clicked:
    if not message.strip():
        st.warning("⚠️ Please enter a message before clicking Predict.")
    else:
        processed_message = preprocess_message(message)
        prediction = model.predict([processed_message])[0]
        probability = model.predict_proba([processed_message])[0]

        safe_probability = probability[0] * 100
        spam_probability = probability[1] * 100

        st.divider()
        st.subheader("Result")

        if prediction == 1:
            st.error("🚫 This message is classified as **SPAM**.")
        else:
            st.success("✅ This message is classified as **SAFE / HAM**.")

        st.subheader("Prediction Probabilities")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Safe probability", value=f"{safe_probability:.2f}%")
        with col2:
            st.metric(label="Spam probability", value=f"{spam_probability:.2f}%")

        st.progress(int(spam_probability))

# ----------------------------
# Footer
# ----------------------------
st.divider()
st.caption("Built with Streamlit · TF-IDF + Logistic Regression · ML/NLP Portfolio Project")