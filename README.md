# credit-card-fraud-detection-resampling
Code and experiment files for thesis on class imbalance and resampling techniques in credit card fraud detection using XGBoost.

# Credit Card Fraud Detection with XGBoost and Resampling Techniques
This repository contains the source code used for the master's thesis titled  
Evaluating the Impact of Resampling Techniques on XGBoost Performance in Credit Card Fraud Detection.

## 📝 Description

The code implements a machine learning pipeline using the [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) dataset from Kaggle.  
It explores the effects of class imbalance and applies four different sampling methods:

- NearMiss (undersampling)
- Random Undersampling with Tomek Links (hybrid)
- SMOTE (oversampling)
- SMOTETomek (hybrid)

The core model used is **XGBoost**, optimized for imbalanced classification problems.

## 📂 File Structure

- `Thesis-credit-card-fraud-detection-resampling.ipynb`  
   Jupyter Notebook used to preprocess data, apply feature engineering, resample the dataset, train models, and evaluate performance.

> 📌 This notebook was originally run on Kaggle using a GPU environment (P100) with `imbalanced-learn==0.10.1`.

## ⚙️ Requirements

You can also run this notebook directly in Kaggle Notebooks, where the dataset and GPU support are pre-installed.

## 📈 Results:
The experiment evaluates models using:

Accuracy

Precision

Recall

F1 Score

AUC

ROC Curve

Precision-Recall Curve

📖 See the thesis for detailed comparisons and business implications of different sampling strategies.

## 📄 License
This project is licensed under the MIT License.

## 👤Author:
Li Zhu
Master’s student in Management Information Systems
ISEG
Date: June 2025

```bash
pip install xgboost imbalanced-learn pandas numpy matplotlib seaborn scikit-learn
