import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib

from src.evaluate import evaluate_model


def train_models(X, y, preprocessor, models):
    results = []

    print("\n[STEP 5] Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("[STEP 6] Training models...\n")

    for name, model in models.items():
        print("-" * 50)
        print(f"Training model: {name}")

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        print("Fitting model...")
        pipeline.fit(X_train, y_train)

        print("Evaluating model...")
        preds = pipeline.predict(X_test)
        metrics = evaluate_model(y_test, preds)

        print(
            f"{name} | "
            f"RMSE: {metrics['RMSE']:.2f} | "
            f"MAE: {metrics['MAE']:.2f} | "
            f"R2: {metrics['R2']:.3f}"
        )

        joblib.dump(pipeline, f"models/{name}.joblib")
        print(f"Model saved to models/{name}.joblib")

        metrics["Model"] = name
        results.append(metrics)

    print("-" * 50)
    print("\n[STEP 7] Model training completed.\n")

    return pd.DataFrame(results)
