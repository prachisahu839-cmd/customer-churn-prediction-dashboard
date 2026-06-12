# Customer Churn Prediction Dashboard

## Overview

Customer churn is one of the most important business problems faced by telecom companies and subscription-based services. Predicting customer churn helps organizations identify customers who are likely to leave and take proactive actions to retain them.

This project uses Machine Learning and Data Analytics techniques to predict customer churn based on customer demographic information, service usage, and billing details. An interactive Streamlit dashboard has been developed to visualize insights and provide real-time churn predictions.

---

## Dataset

**Dataset:** Telco Customer Churn Dataset

The dataset contains information about telecom customers, including:

* Customer demographics
* Subscription details
* Contract information
* Payment methods
* Monthly charges
* Total charges
* Churn status

Target Variable:

* **Churn**

  * 0 = Customer Stays
  * 1 = Customer Leaves

---

## Project Workflow

### 1. Data Preprocessing

* Removed unnecessary columns
* Handled missing values
* Converted data types
* Encoded categorical features
* Saved cleaned dataset

### 2. Exploratory Data Analysis (EDA)

* Churn Distribution Analysis
* Customer Behavior Analysis
* Service Usage Analysis
* Correlation Analysis

### 3. Model Building

Machine Learning Algorithm Used:

* Random Forest Classifier

### 4. Model Evaluation

Evaluation Metrics:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

### 5. Dashboard Development

Interactive dashboard built using Streamlit for:

* Customer input
* Churn prediction
* Visual analytics
* Model insights

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Seaborn
* Streamlit
* Joblib
* Git & GitHub

---

## Project Structure

```text
customer-churn-prediction
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── churn_model.pkl
│
├── notebooks/
│   └── churn_analysis.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── eda.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── predict.py
│   └── utils.py
│
├── visuals/
│
├── README.md
└── requirements.txt
```

---

## Model Performance

* Algorithm: Random Forest Classifier
* Accuracy: **79.56%**

### Classification Results

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 79.56% |
| Precision | 66%    |
| Recall    | 47%    |
| F1-Score  | 55%    |

---

## Features

* Data Cleaning and Preprocessing
* Exploratory Data Analysis
* Machine Learning Model Training
* Churn Prediction
* Feature Importance Analysis
* Confusion Matrix Visualization
* Interactive Dashboard

---

## How to Run

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Dashboard

```bash
streamlit run dashboard/app.py
```

---

## Future Enhancements

* XGBoost Model Implementation
* Hyperparameter Tuning
* Deployment on Streamlit Cloud
* Advanced Customer Segmentation
* Real-Time Database Integration

---

## Author

Prachi Sahu

B.Tech (Data Analytics)

Customer Churn Prediction Dashboard using Machine Learning and Streamlit.
