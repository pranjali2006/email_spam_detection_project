# 📧 Email Spam Detection

An end-to-end **Natural Language Processing (NLP) and Machine Learning project** that classifies email messages as **Spam** or **Ham (Safe)**.

The project focuses not only on building a classifier, but also on understanding **text preprocessing, TF-IDF feature extraction, Logistic Regression, evaluation metrics, probability analysis, threshold tuning, and deployment**.

---

## 🎯 Project Objective

The objective of this project is to automatically classify an email message as:

* **Spam** — unwanted or potentially malicious message
* **Ham** — legitimate/safe message

The project also analyzes the model's prediction probabilities and selects an appropriate classification threshold based on the trade-off between **spam recall and false positives**.

---

## 📊 Dataset

The project uses a real-world **Kaggle email spam dataset** containing email messages labelled as `ham` or `spam`.

### Dataset characteristics

* **Original records:** 5,573
* **Valid records after removing one invalid category:** 5,572
* **Ham:** 4,825
* **Spam:** 747
* **Features:** Email message text
* **Target:** Email category (`ham` / `spam`)

The dataset is imbalanced, with legitimate emails being much more common than spam emails.

---

## 🔄 Project Workflow

```text
Raw Email Dataset
       ↓
Data Validation
       ↓
Remove Invalid Records
       ↓
Text Preprocessing
       ├── Lowercasing
       ├── Stopword Removal
       └── Punctuation Removal
       ↓
Exploratory Data Analysis
       ├── Class Distribution
       ├── Message Length
       ├── Word Count
       └── Common Words
       ↓
Train-Test Split
       └── Stratified Split
       ↓
TF-IDF Vectorization
       ↓
Logistic Regression
       ↓
Baseline Evaluation
       ↓
Probability & False-Negative Analysis
       ↓
Threshold Tuning
       ↓
Final Threshold = 0.3
       ↓
Streamlit Deployment
```

---

## 🧹 Text Preprocessing

The email messages are cleaned before being passed to the machine learning model.

The preprocessing steps include:

1. Converting text to lowercase
2. Removing English stopwords
3. Removing punctuation

These steps reduce unnecessary variation in the text and provide cleaner input for feature extraction.

---

## 🔤 TF-IDF Feature Extraction

Since machine learning algorithms cannot directly work with raw text, **TF-IDF (Term Frequency-Inverse Document Frequency)** is used to convert email messages into numerical feature vectors.

TF-IDF gives higher importance to words that are useful for distinguishing between different documents while reducing the importance of very common words.

The TF-IDF vectorizer is included inside a Scikit-learn `Pipeline` together with the classifier.

---

## 🤖 Machine Learning Model

### Logistic Regression

Logistic Regression was selected as the classification algorithm because:

* The problem is binary classification.
* TF-IDF produces high-dimensional sparse text features.
* Logistic Regression performs well on many text-classification problems.
* It is relatively simple and interpretable.
* It provides class probabilities that can be used for threshold tuning.

The model pipeline is:

```text
Text
 ↓
TF-IDF
 ↓
Logistic Regression
 ↓
Spam Probability
 ↓
Classification Decision
```

---

## 📈 Model Evaluation

The initial model uses the default **0.5 classification threshold**.

The evaluation includes:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* Classification Report

### Baseline Results — Threshold 0.5

| Metric    |  Ham | Spam |
| --------- | ---: | ---: |
| Precision | 0.96 | 1.00 |
| Recall    | 1.00 | 0.72 |
| F1-score  | 0.98 | 0.84 |

**Overall accuracy:** approximately **96%**

The baseline model produced **42 false negatives**, meaning 42 spam emails were classified as legitimate emails.

This showed that accuracy alone was not sufficient for evaluating the spam-classification problem.

---

## 🎚️ Threshold Analysis

Instead of relying only on the default 0.5 threshold, the model's predicted spam probabilities were analyzed.

Several thresholds were tested:

| Threshold | Spam Precision | Spam Recall |  Spam F1 | False Positives |
| --------: | -------------: | ----------: | -------: | --------------: |
|       0.5 |           1.00 |        0.72 |     0.84 |               0 |
|       0.4 |           0.99 |        0.80 |     0.88 |               1 |
|   **0.3** |       **0.96** |    **0.85** | **0.90** |           **5** |
|       0.2 |           0.89 |        0.92 |     0.90 |              17 |

### Final Threshold: 0.3

A threshold of **0.3** was selected because it provides a better balance between:

* Detecting more spam emails
* Maintaining high spam precision
* Limiting false-positive predictions

At 0.3, spam recall increased from **72% to 85%**, while only 5 legitimate emails were incorrectly classified as spam.

The threshold was therefore selected based on the **precision-recall trade-off**, rather than simply choosing the default 0.5 threshold.

---

## 🧪 Custom Prediction

The trained model can also classify a custom email message.

Example:

```text
"call to win a mobile click the link to claim the reward"
```

The application calculates the spam probability and compares it against the selected threshold.

```text
Spam Probability >= 0.3
        ↓
     Spam

Spam Probability < 0.3
        ↓
      Safe
```

---

## 🚀 Deployment

The model is deployed using **Streamlit**.

The saved model contains:

* TF-IDF + Logistic Regression pipeline
* Final classification threshold

This allows the deployment application to use the same trained pipeline and decision threshold used during evaluation.

### Run the application

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Then run:

```bash
streamlit run app.py
```

The application provides a text box where the user can enter an email message and receive:

* Spam/Safe prediction
* Spam probability
* Safe probability

---


## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* NLTK
* Scikit-learn
* TF-IDF
* Logistic Regression
* Streamlit
* Pickle
* Git & GitHub

---

## 💡 Key Learnings

Through this project, I worked on:

* Text preprocessing for NLP
* Exploratory data analysis
* Handling class imbalance during evaluation
* TF-IDF feature extraction
* Building Scikit-learn pipelines
* Logistic Regression for text classification
* Precision, recall and F1-score interpretation
* Confusion matrix analysis
* False-negative analysis
* Probability-based predictions
* Classification threshold tuning
* Model serialization
* Streamlit deployment

A major learning from this project was that **a high accuracy score does not necessarily mean that a classification model is performing well for the business objective**. For spam detection, missing actual spam emails is important, so recall and the precision-recall trade-off need to be considered when selecting the classification threshold.

---

## 👩‍💻 Author

**Pranjali**

B.Tech — Artificial Intelligence & Data Science

Interested in **Data Science, Machine Learning, Deep Learning, NLP and AI Engineering**.
