
#  Colorectal Cancer Survival Prediction using Machine Learning

##  Project Overview

This project aims to build machine learning models to predict **5-year survival** among patients with **colorectal cancer** based on clinical, lifestyle, and demographic factors.

It demonstrates the application of:
- Classification algorithms
- Ensemble learning
- Model evaluation techniques
- Model deployment via Streamlit

> Early prediction can help doctors personalize treatment strategies, reduce mortality, and improve patient outcomes.

---

##  Dataset

- **Rows:** 167,497 patients  
- **Features:** 28 (demographics, clinical data, lifestyle factors, economic indicators)  
- **Target:** `Survival_5_years` (Yes/No)

 [Dataset on Kaggle](https://www.kaggle.com/datasets/ankushpanday2/colorectal-cancer-global-dataset-and-predictions)

---

## 🔍 Data Processing Steps

### 1. **Exploration**
- Summary statistics
- Class distribution

### 2. **Preprocessing**
- Handled missing values
- Removed outliers using IQR
- Removed duplicates
- Label and Ordinal Encoding
- Whitespace stripping

### 3. **Feature Selection & Engineering**
- Applied `SelectKBest` with `f_classif` and `chi2`
- **Top Features:**
  - Family_History, Alcohol_Consumption, Mortality, Insurance_Status
  - Urban_or_Rural, Country, Incidence_Rate_per_100K
  - Mortality_Rate_per_100K, Healthcare_Access, Obesity_BMI
  - Screening_History, Treatment_Type, Tumor_Size_mm
- Engineered Feature: `Tumor_Size_Category`
- Cleaned data saved to `cleaned_selected_data.csv`

---

##  Visualization

- Heatmaps (correlation matrix)
- Distributions of selected features
- Class balance charts
- ROC Curves
- Confusion Matrices

---

##  Modeling Pipeline

- Train/Test Split
- Feature Scaling
- Class Imbalance Handling with **SMOTE**
- Model Training & Cross Validation

###  Models Used

Logistic Regression (baseline) 

K-Nearest Neighbors (KNN) (with different k values) 

Random Forest Classifier 

STACKING classifier (Using LR, KNN, NB) 

Bagging (with Logistic Regression) 

Decision Tree Classifier (with different max depths) 

XGBoost Classifier 

AdaBoost Classifier 

Voting Classifier (ensemble) 

Ensemble Model Training and Evaluation 

Cross Validation for each model 

---

## Deployment

A **Streamlit app** was deployed using:

- `AdaBoostClassifier` with `DecisionTreeClassifier` base
- Trained on selected features
- Encoded with `LabelEncoder` for country and other categories

🔗 [Live App on Streamlit](https://colorectal-cancer-survival-prediction-rybrucaoafe8jpmxpozqxn.streamlit.app/)

---

##  Repository Structure

```
├── app.py                     # Streamlit app
├── adaboost_colorectal_model.pkl  # Trained AdaBoost model
├── country_encoder.pkl        # LabelEncoder for Country
├── requirements.txt           # Python dependencies
└── README.md                  # Project overview
```


