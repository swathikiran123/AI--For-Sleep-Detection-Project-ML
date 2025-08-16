# 💤 AI for Sleep Detection: Predicting Sleep States from Accelerometer Data  

## 📌 Problem Statement  
Sleep plays a vital role in human health, development, and cognitive functioning. However, large-scale sleep research is challenging due to limited naturalistic data and accurate annotations.  

This project aims to **build a machine learning model that detects sleep states (onset and wakeup) from wrist-worn accelerometer data**. By leveraging data science, researchers and healthcare professionals can better analyze sleep patterns and improve sleep-related studies.  

---

## 📖 Project Description  
This project focuses on **predicting sleep onset and wakeup times** using accelerometer data from the [Child Mind Institute Kaggle Competition](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/data).  

The dataset includes:  
- **500+ multi-day accelerometer recordings**  
- Sleep events:  
  - **Onset** → Falling asleep  
  - **Wakeup** → Waking up  

---

## ⚙️ Project Workflow  

### 🔹 Data Collection  
- Source: Kaggle (Child Mind Institute Dataset)  
- Files:  
  - `train_series.parquet` → Sensor data (timestamp, anglez, enmo)  
  - `train_events.csv` → Sleep events (onset/wakeup)  

### 🔹 Data Preprocessing & Cleaning  
- Handle missing values, duplicates, and outliers  
- Feature scaling (StandardScaler, MinMaxScaler)  
- Encode categorical variables (One-Hot, Label Encoding)  
- Address data imbalance (oversampling techniques)  

### 🔹 Exploratory Data Analysis (EDA)  
- Correlation analysis  
- Visualization of sleep/wake patterns  
- Outlier detection (Box Plot, IQR)  

### 🔹 Feature Selection  
- Variance Threshold  
- SelectKBest (Chi-Square, Mutual Information)  
- Recursive Feature Elimination (RFE)  
- Feature importance (Decision Trees, Random Forests)  

### 🔹 Model Building (Classification Task)  
Implemented and compared multiple ML algorithms:  
- K-Nearest Neighbors (KNN)  
- Naïve Bayes  
- Decision Tree  
- Logistic Regression  
- Support Vector Machine (SVM)  
- Random Forest  
- XGBoost  

### 🔹 Model Validation  
- Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC  
- Confusion Matrix  
- Cross-validation  
- Hyperparameter tuning (Optuna, GridSearchCV)  

### 🔹 Deployment  
- Deployed best-performing model using **Streamlit**  
- Hosted on **Hugging Face Spaces**  

---

## 🚀 Skills & Tools Used  
- **Languages:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, Seaborn, Scipy, Statsmodels, XGBoost  
- **Techniques:** Data Cleaning, Feature Engineering, Hyperparameter Tuning, Model Validation  
- **Deployment:** Streamlit, Hugging Face Spaces  

---

## ✅ Outcome  
The project successfully demonstrates how **machine learning can classify sleep states from accelerometer data**, helping researchers conduct large-scale studies on sleep monitoring and its importance.  

---

## 📌 How to Run  
1. Clone the repository  
   ```bash
   git clone https://github.com/yourusername/AI-Sleep-Detection.git
   cd AI-Sleep-Detection
