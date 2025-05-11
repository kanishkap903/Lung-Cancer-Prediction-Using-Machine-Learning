# Lung Cancer Prediction Model

This repository contains a machine-learning model for predicting lung cancer risk based on various patient features. The project includes data preprocessing, model training, evaluation, and saving the trained model for future use.

## Table of Contents

- [Project Overview](#project-overview)
- [Data](#data)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
- [Model Evaluation](#model-evaluation)

## Project Overview

This project aims to develop a machine learning model to predict the likelihood of lung cancer in patients using several health-related features. The workflow involves:

1. Data preprocessing and cleaning
2. Feature standardization
3. Splitting data into training and test sets
4. Training and evaluating different machine learning models
5. Saving the trained model for future predictions

## Data

The dataset used in this project is `Lung Cancer Data.csv`, which includes the following features:

- `AGE`: Age of the patient
- `SMOKING`: Smoking status (0: No, 1: Yes)
- `YELLOW_FINGERS`: Presence of yellow fingers (0: No, 1: Yes)
- `ANXIETY`: Presence of anxiety (0: No, 1: Yes)
- `PEER_PRESSURE`: Peer pressure (0: No, 1: Yes)
- `CHRONIC DISEASE`: Chronic disease (0: No, 1: Yes)
- `FATIGUE`: Presence of fatigue (0: No, 1: Yes)
- `ALLERGY`: Presence of allergy (0: No, 1: Yes)
- `WHEEZING`: Presence of wheezing (0: No, 1: Yes)
- `ALCOHOL CONSUMING`: Alcohol consumption (0: No, 1: Yes)
- `COUGHING`: Presence of coughing (0: No, 1: Yes)
- `SHORTNESS OF BREATH`: Shortness of breath (0: No, 1: Yes)
- `SWALLOWING DIFFICULTY`: Difficulty swallowing (0: No, 1: Yes)
- `CHEST PAIN`: Chest pain (0: No, 1: Yes)
- `LUNG_CANCER`: Target variable (0: No, 1: Yes)

## Prerequisites

Make sure you have the following Python libraries installed:

- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `joblib`

You can install the required libraries using pip:

```bash
pip install pandas numpy matplotlib scikit-learn joblib
```
## Usage

1. **Data Preprocessing**: Load and preprocess the data.
2. **Model Training**: Train the model using `LogisticRegression`. You can also uncomment the `K-Nearest Neighbors` or `SVM` model for comparison.
3. **Model Evaluation**: Evaluate the model using precision, recall, confusion matrix, and log loss.
4. **Model Saving**: Save the trained model using `joblib`.

Run the following command to execute the script:

```bash
python lung_cancer_prediction.py
```
## Model Evaluation

After training the model, the evaluation metrics include:

- **Precision**: Measures the accuracy of positive predictions.
- **Recall**: Measures the ability of the model to find all positive samples.
- **Confusion Matrix**: Displays the number of true positives, true negatives, false positives, and false negatives.
- **Log Loss**: Measures the performance of the classification model where predictions are probabilities.

## Disclaimer:

This project is for educational purposes only and should not be used for real-world medical diagnosis or treatment. The machine learning model developed here is based on publicly available data from Kaggle and has been modified and trained to demonstrate classification techniques using Logistic Regression, KNN, and SVM. It has not been validated for clinical use, and its predictions may not be reliable for actual patient care. Always consult healthcare professionals for medical advice and decisions.
