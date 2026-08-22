# Credit Card Fraud Detection System

An end-to-end Machine Learning system engineered to identify fraudulent credit card transactions in highly imbalanced financial datasets. Built for It Simplera Solutions, this project combines Synthetic Minority Over-sampling Technique (SMOTE) with Cost-Sensitive XGBoost classification, an operational decision threshold (P >= 0.15), and a real-time Streamlit web interface.

---

## Overview & Architecture

Financial transaction streams feature extreme class imbalance, where fraudulent activities represent less than 0.2% of overall volume. Optimizing models purely for raw accuracy leads to deceptive performance while leaving critical fraud instances undetected.

Key components of this pipeline include:
* **SMOTE Resampling**: Applied strictly on training folds to address minority class representation without causing data leakage.
* **Cost-Sensitive XGBoost**: Trained on resampled features to heavily penalize misclassified fraud cases during gradient boosting.
* **Threshold Optimization**: Calibrated decision boundaries at P >= 0.15 to prioritize Recall (>91%) over default thresholding (P = 0.50).
* **Streamlit Web Portal**: Serialized model artifacts deployed for sub-10ms real-time inference.
* **IEEE Publication**: Complete 2-column LaTeX research paper documentation.

---

## Dataset Characteristics

Experiments utilize the Kaggle European Credit Card Fraud dataset:

* **Total Records**: 284,807 transactions (2-day window).
* **Legitimate Class (0)**: 284,315 transactions (99.828%).
* **Fraudulent Class (1)**: 492 transactions (0.172%).
* **Features**: `V1` to `V28` (PCA transformed attributes), alongside raw `Time` and `Amount` fields standardized via Z-score scaling.

---

## Empirical Performance Results

Model evaluation was conducted on a 20% holdout test set:

| Model Architecture | Precision | Recall | F1-Score | AUPRC |
|---|---|---|---|---|
| Baseline Logistic Regression | 0.85 | 0.62 | 0.72 | 0.71 |
| Standard Random Forest | 0.88 | 0.76 | 0.81 | 0.82 |
| **SMOTE + XGBoost (Proposed)** | **0.87** | **0.85** | **0.86** | **0.88** |

### Decision Threshold Analysis (P >= 0.15)
Shifting the decision boundary from default 0.50 to 0.15 elevates operational Recall to 91.83%, catching the vast majority of fraudulent transactions in production scenarios.

---

## Repository Structure

* `app.py`: Streamlit frontend code for real-time risk scoring.
* `models/`: Folder containing serialized `fraud_model.pkl` and `scaler.pkl` artifacts.
* `notebooks/`: Jupyter Notebook detailing data preprocessing, SMOTE training, and model evaluation.
* `paper/`: Complete IEEE LaTeX paper (`main.tex`) and compiled PDF (`main.pdf`).
* `requirements.txt`: Environment dependencies required to execute the pipeline.

---

## How to Run

1. **Clone the repository**:
   git clone https://github.com/your-username/Credit-Card-Fraud-Detection.git
   cd Credit-Card-Fraud-Detection

2. **Install dependencies**:
   pip install -r requirements.txt

3. **Launch the Streamlit app**:
   streamlit run app.py

---

## Author

* **Author**: Sharmeen Fatima
* **Intern ID**: AIMLB01-1994
* **Organization**: It Simplera Solutions


 ## Screeenshots 
<img width="947" height="416" alt="final project ss" src="https://github.com/user-attachments/assets/beb43bac-14c3-4f21-ac19-781341f81c66" />


<img width="748" height="421" alt="final project ss 2" src="https://github.com/user-attachments/assets/1f78dec6-5304-4ad9-aa46-51599a99acf5" />


  

