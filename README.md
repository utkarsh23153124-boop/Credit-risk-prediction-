# 💳 Credit Card Risk Prediction

## 📌 Overview

This project aims to predict whether a customer is **risky (likely to default)** or **safe** based on their financial and demographic data.
It uses machine learning techniques to help financial institutions make better credit decisions.

---

## 🚀 Key Features

* Data preprocessing and cleaning
* Missing value handling (median & mode)
* One-hot encoding for categorical features
* Feature scaling using StandardScaler
* Handling imbalanced dataset using SMOTE
* Hyperparameter tuning using GridSearchCV
* Model comparison and selection

---

## 📊 Dataset

The dataset contains customer-related information such as income, demographics, and financial behavior.

**Target Variable:**

* `Status = 1` → Risky Customer
* `Status = 0` → Safe Customer

---

## 🛠️ Technologies Used

* Python
* Pandas, NumPy
* Matplotlib
* Scikit-learn
* Imbalanced-learn (SMOTE)

---

## 🔍 Data Preprocessing

* Missing values handled:

  * Numerical → Median
  * Categorical → Mode
* Categorical variables encoded using One-Hot Encoding
* Features scaled using StandardScaler
* Data split into training and testing sets (80:20 with stratification)

---

## ⚖️ Handling Imbalanced Data

* Applied **SMOTE (Synthetic Minority Oversampling Technique)**
* Balanced the dataset to improve prediction of risky customers

---

## 🤖 Models Implemented

### 1. Logistic Regression

* Tuned using GridSearchCV
* Optimized using F1-score

### 2. Random Forest Classifier

* Handles complex relationships
* Provides feature importance

---

## 📊 Evaluation Metrics

Models are evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC Score
* Confusion Matrix

---

## 📈 Feature Importance

* Extracted using Random Forest
* Identifies top factors affecting credit risk

---

## 🏆 Model Selection

* Compared Logistic Regression and Random Forest
* Selected best model based on **F1-score**

---

## 📂 Project Structure

```
├── dataset.csv
├── credit_risk.ipynb / main.py
├── README.md
```

---

## ▶️ How to Run

1. Clone the repository:

```
git clone <your-repo-link>
```

2. Install dependencies:

```
pip install pandas numpy matplotlib scikit-learn imbalanced-learn
```

3. Run the notebook or script:

```
python main.py
```

---

## 🎯 Future Improvements

* Implement XGBoost / LightGBM
* Perform feature engineering
* Deploy model using Streamlit or Flask
* Add cross-validation for better performance

---

## 👨‍💻 Author

**Utkarsh Srivastav**
B.Tech CSE (AI & ML)
Noida, India

---

## ⭐ Acknowledgment

This project was built for learning and practical implementation of machine learning techniques in financial risk prediction.
