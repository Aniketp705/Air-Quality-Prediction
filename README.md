# AQI Bucket Prediction App

## Overview

This project is a web application that predicts the Air Quality Index (AQI) bucket based on various pollutant concentrations and the date of measurement. The application uses a pre-trained machine learning model to make predictions and features a user-friendly interface built with Streamlit, including a custom dark theme.

The accompanying Jupyter Notebook (`AQI.ipynb`) details the data preprocessing, exploratory data analysis, and model training process (demonstrated with a Logistic Regression model). The Streamlit application (`main.py`) utilizes a pre-trained Random Forest model for predictions.

## Features

*   **User-friendly Web Interface:** Built with Streamlit for easy interaction.
*   **Pollutant Input:** Allows users to input concentrations for:
    *   PM2.5 (μg/m³)
    *   PM10 (μg/m³)
    *   NO (μg/m³)
    *   NO2 (μg/m³)
    *   NOx (μg/m³)
    *   NH3 (μg/m³)
    *   CO (mg/m³)
    *   SO2 (μg/m³)
    *   O3 (μg/m³)
    *   Benzene (μg/m³)
    *   Toluene (μg/m³)
    *   Xylene (μg/m³)
*   **Date Input:** Select the date of measurement.
*   **AQI Bucket Prediction:** Predicts one of the following AQI buckets:
    *   Good ✨
    *   Satisfactory 👍
    *   Moderate ⚠️
    *   Poor 😷
    *   Very Poor 🚨
    *   Severe 🛑
*   **Color-Coded Results:** Displays the prediction with a color corresponding to the AQI bucket's severity.
*   **Custom Styling:** Features a custom dark theme for improved aesthetics.

## Model Details

### Model Used in the App

The Streamlit application (`main.py`) uses a pre-trained **Random Forest classifier** (`aqi_random_forest_best_model.pkl`) for predicting the AQI bucket. The input features are scaled using a pre-trained **StandardScaler** (`aqi_scaler.pkl`).

### Data Preprocessing (as shown in `AQI.ipynb`)

*   **Missing Value Imputation:**
    *   Numerical features: Imputed with the mean.
    *   Categorical features (AQI_Bucket): Imputed with the mode.
*   **Duplicate Removal:** Duplicate rows were dropped.
*   **Feature Scaling:** Numerical pollutant features were standardized using `StandardScaler` from Scikit-learn.

### Feature Engineering

The following features are used for prediction by the model in `main.py`:

*   **Pollutant Concentrations (12 features):** PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene.
*   **Date Features (4 features):**
    *   Year
    *   Month
    *   Day
    *   Day of the week

## Dataset

The model development process (shown in `AQI.ipynb`) used the `city_day.csv` dataset, which typically contains daily air quality data for various cities, including pollutant concentrations and AQI values.

## Model Training and Results (from `AQI.ipynb` - Random Forest Classifier)

The `AQI.ipynb` notebook demonstrates the model training process using a Random Forest Classifier model for multiclass classification.

*   **Model:** `RandomForestClassifier` with hyperparameter tuning using `RandomizedSearchCV`.
    *   Best Hyperparameters:
        *   `n_estimators`: (Value from notebook, e.g., 170)
        *   `max_depth`: (Value from notebook, e.g., 27)
        *   `min_samples_split`: (Value from notebook, e.g., 4)
        *   `min_samples_leaf`: (Value from notebook, e.g., 1)
        *   `class_weight`: 'balanced'
*   **Best Cross-Validation Accuracy (during tuning):** (Value from `random_search.best_score_`, e.g., 0.9678)
*   **Test Set Accuracy:** (Value from `test_acc`, e.g., 0.9751)

*   **Classification Report (Random Forest Classifier on Test Set):**
    ```
                  precision    recall  f1-score   support

            Good       0.98      0.85      0.91       268
        Moderate       0.98      0.99      0.99      2611
            Poor       0.98      0.95      0.96       593
    Satisfactory       0.96      0.99      0.97      1595
          Severe       1.00      0.99      0.99       278
       Very Poor       0.97      0.98      0.97       467

        accuracy                           0.98      5812
       macro avg       0.98      0.96      0.97      5812
    weighted avg       0.98      0.98      0.97      5812
    ```
*   **Note:** The `AQI.ipynb` notebook details the hyperparameter tuning process using `RandomizedSearchCV` and evaluates the best `RandomForestClassifier` found. The classification report above is based on the test set evaluation of this tuned model.

**Note:** The Streamlit application (`main.py`) uses this pre-trained Random Forest model (`aqi_random_forest_best_model.pkl`).