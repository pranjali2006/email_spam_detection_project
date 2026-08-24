# 📧 Email Spam Detection using Logistic Regression

A Machine Learning project that classifies email messages as **Spam** or **Safe (Ham)** using **TF-IDF Vectorization** and **Logistic Regression**.

This project was developed as part of my learning journey while studying **Machine Learning algorithms**, particularly **Logistic Regression**. After learning the concepts and working of Logistic Regression, I applied them to a practical text-classification problem: detecting spam emails.

## 🎯 Project Objective

The objective of this project is to build a simple machine learning application that can analyze an email message and predict whether it is:

* 🚫 **Spam**
* ✅ **Safe / Ham**

The trained model is integrated into a **Streamlit web application** so that users can enter an email message and receive a prediction along with the model's confidence probabilities.

## 🧠 Machine Learning Approach

The project uses:

* **Text Preprocessing**
* **TF-IDF (Term Frequency–Inverse Document Frequency)**
* **Logistic Regression**
* **Binary Classification**

The trained model is saved as a Scikit-learn Pipeline containing:

```text
TF-IDF Vectorizer
        ↓
Logistic Regression
        ↓
Spam / Safe Prediction
```

Using a Pipeline allows the TF-IDF transformation and Logistic Regression model to be saved and loaded together for deployment.

## 🔄 Project Workflow

```text
Email Message
      ↓
Text Preprocessing
      ↓
Lowercase + Stopword Removal + Punctuation Removal
      ↓
TF-IDF Vectorization
      ↓
Logistic Regression
      ↓
Prediction
      ↓
Spam / Safe
      ↓
Prediction Probability
```

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* NLTK
* Joblib / Pickle
* Streamlit
* Plotly

## 📁 Project Structure

```text
email_spam_detection_project/
│
├── dataset/
│   ├── code.ipynb
│   ├── email.csv
│   ├── new_code.ipynb
│   └── new_synthetic_spam (1).csv
│
├── deployment/
│   ├── app.py
│   ├── claude.py
│   ├── preprocessing.py
│   ├── requirements.txt
│   └── spam_model.pkl
│
├── .gitignore
└── README.md
```

## 🌐 Streamlit Application

The trained model is deployed as an interactive Streamlit application.

The application allows users to:

1. Enter or paste an email message.
2. Preprocess the message.
3. Predict whether the message is Spam or Safe.
4. View the prediction probabilities.

## 🎨 UI Development

For improving and refining the user interface of the Streamlit application, I used **Claude** as an AI-assisted development tool.

The focus was on improving the presentation and usability of the application while keeping the underlying machine learning workflow my own.

## 📚 What I Learned

Through this project, I practiced and strengthened my understanding of:

* Logistic Regression for binary classification
* Text preprocessing
* TF-IDF feature extraction
* Scikit-learn Pipelines
* Model serialization and loading
* Using NLTK for text preprocessing
* Integrating an ML model with Streamlit
* Deploying a machine learning application
* Managing an ML project using Git and GitHub

Most importantly, this project helped me move from **learning the Logistic Regression algorithm theoretically to applying it in a real-world machine learning problem**.

## 🚀 Future Improvements

Some possible improvements for the project include:

* Experimenting with other classification algorithms
* Comparing model performance using different evaluation metrics
* Improving text preprocessing
* Adding a larger and more diverse dataset
* Adding model performance visualizations
* Improving the user interface further

## 👩‍💻 Author

**Pranjali Pradeep Yewale**

AI & Data Science Student
Interested in Machine Learning, Data Science and ML Research.

---

⭐ This project is part of my ongoing journey of learning and implementing Machine Learning algorithms through practical projects.
