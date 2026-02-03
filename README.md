# ✈️ Flight Fare Prediction — End-to-End Machine Learning System (AWS-Ready)

## 📌 Overview

This project implements a **production-style machine learning pipeline** to predict airline ticket prices using historical flight data. It demonstrates the complete ML lifecycle — from data preprocessing and feature engineering to model training, hyperparameter tuning, and cloud-ready deployment.

The codebase is modular, scalable, and structured to align with **real-world ML engineering and MLOps practices**.

---

## 💡 Business Problem

Airline ticket prices fluctuate based on several dynamic factors, making manual price estimation inefficient and unreliable.

**Objective:**  
Build a machine learning model that accurately predicts flight fares using historical data to support pricing analysis and demand forecasting.

---

## 🧠 Machine Learning Solution

- **Problem Type:** Supervised Learning (Regression)
- **Inputs:**
  - Airline
  - Source & Destination
  - Journey Date
  - Departure & Arrival Time
  - Duration
  - Number of Stops
- **Output:** Predicted flight ticket price

### ML Workflow
1. Data ingestion and validation  
2. Data cleaning and preprocessing  
3. Feature engineering (date, time, categorical, numerical)  
4. Model training and comparison  
5. Hyperparameter tuning  
6. Model evaluation and persistence  

---

## 🛠️ Tech Stack

- **Programming Language:** Python  
- **Data & ML:** Pandas, NumPy, Scikit-learn  
- **Visualization:** Matplotlib, Seaborn  
- **Cloud:** AWS (deployment-ready architecture)  

---

## 📂 Project Architecture

```bash
flight-fare-prediction/
│
├── data/                # Raw flight fare dataset
├── notebooks/           # EDA & experimentation
├── src/                 # Core ML pipeline
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── models.py
│   └── train.py
│
├── models/              # Saved trained models
├── deploy/              # AWS deployment components
├── main.py              # End-to-end pipeline runner
├── tune.py              # Hyperparameter tuning
└── requirements.txt

```
---
### Models Used

The following regression models were implemented, trained, and evaluated:

- **LightGBM Regressor**
  - Gradient boosting framework optimized for speed and performance
  - Handles large datasets efficiently
  - Well-suited for structured/tabular data

- **XGBoost Regressor**
  - Regularized gradient boosting model
  - Strong performance on non-linear regression problems
  - Robust to overfitting with built-in regularization

Both models were trained using a unified preprocessing + modeling pipeline, evaluated on a held-out test set, and compared using standard regression metrics. The best-performing model was persisted for deployment.

---

## 📊 Model Training & Evaluation

- Implemented and compared multiple regression models to identify the best-performing approach  
- Applied categorical encoding and numerical feature scaling to improve model learning  
- Performed hyperparameter tuning to optimize model performance  
- Evaluated models using standard regression metrics  
- Persisted trained models for reuse and deployment  

The modular design enables easy experimentation, model comparison, and future upgrades without impacting the overall pipeline.

---
### Model Comparison Summary

| Model               | RMSE | MAE | R² Score | Notes |
|--------------------|------|-----|----------|-------|
| LightGBM Regressor | ✓    | ✓   | ✓        | Fast training, strong performance on tabular data |
| XGBoost Regressor  | ✓    | ✓   | ✓        | Robust to overfitting, strong non-linear modeling |

Models were evaluated on a held-out test dataset. The best-performing model was selected based on a balance of error metrics and generalization performance, then persisted for deployment.

---
### Evaluation Metrics

- **RMSE (Root Mean Squared Error):** Measures average prediction error magnitude with higher penalty on large errors  
- **MAE (Mean Absolute Error):** Measures average absolute prediction error  
- **R² Score:** Indicates how well the model explains variance in flight prices  

These metrics together ensure both accuracy and robustness of the regression models.

---
## 🧑‍💻 ML Engineering & MLOps Perspective

This project was designed with production and MLOps principles in mind:

- Modular ML pipeline separating data ingestion, preprocessing, training, and evaluation
- Reproducible experiments with consistent preprocessing and model interfaces
- Model artifacts persisted for versioning and deployment
- Clear separation of training and inference logic
- Cloud-ready structure compatible with AWS EC2, Lambda, and SageMaker

The architecture enables easy retraining, model replacement, CI/CD integration, and scalable deployment — aligning closely with real-world ML engineering workflows.

---


## ☁️ Deployment Readiness (AWS)

The project is structured to support **cloud deployment with minimal changes**, following best practices for scalable machine learning systems.

### Supported Deployment Scenarios

- AWS EC2–based model hosting  
- AWS Lambda for lightweight inference  
- Migration to AWS SageMaker pipelines  

### Why This Matters

- Clear separation of training and inference logic  
- Easy model versioning and retraining  
- Scalable and production-friendly architecture  

---

## 📈 Key Highlights

✔ End-to-end machine learning pipeline  
✔ Production-style project organization  
✔ Cloud-ready and scalable design  
✔ Demonstrates real-world ML engineering practices  
✔ Suitable for ML Engineer / Data Engineer roles  

---

## 🔮 Future Enhancements

- REST API using FastAPI for real-time predictions  
- CI/CD integration for automated ML pipelines  
- Real-time flight fare data ingestion  
- Model monitoring and drift detection  
- Full AWS SageMaker deployment  

---

## 👤 Author

**Karthik Adari**  
Applied Machine Learning / Data Engineering  
Focused on building scalable, production-ready ML systems

