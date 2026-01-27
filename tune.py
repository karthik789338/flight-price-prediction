import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import randint, uniform

from src.data_loader import load_data
from src.preprocessing import clean_data, engineer_features, build_preprocessor, TARGET

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor


DATA_PATH = "data/raw/Flight data.csv"
RANDOM_STATE = 42

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def evaluate(name, model, X_test, y_test):
    pred = model.predict(X_test)
    return {
        "Model": name,
        "RMSE": rmse(y_test, pred),
        "MAE": float(mean_absolute_error(y_test, pred)),
        "R2": float(r2_score(y_test, pred))
    }

def main():
    print("[1] Load data...")
    df = load_data(DATA_PATH)
    print("Raw shape:", df.shape)

    print("[2] Clean + engineer features...")
    df = clean_data(df)
    df = engineer_features(df)
    print("After preprocessing shape:", df.shape)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    print("[3] Train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessor()

    # ----------------------------
    # Tune LightGBM
    # ----------------------------
    print("\n[4] Tuning LightGBM...")
    lgbm = LGBMRegressor(
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    lgbm_pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", lgbm)
    ])

    lgbm_params = {
        "model__n_estimators": randint(800, 3500),
        "model__learning_rate": uniform(0.01, 0.08),   # 0.01 to 0.09
        "model__num_leaves": randint(31, 256),
        "model__max_depth": randint(3, 16),
        "model__min_child_samples": randint(10, 200),
        "model__subsample": uniform(0.6, 0.4),         # 0.6 to 1.0
        "model__colsample_bytree": uniform(0.6, 0.4),  # 0.6 to 1.0
        "model__reg_alpha": uniform(0.0, 1.0),
        "model__reg_lambda": uniform(0.0, 2.0),
    }

    lgbm_search = RandomizedSearchCV(
        estimator=lgbm_pipe,
        param_distributions=lgbm_params,
        n_iter=30,
        scoring="neg_mean_squared_error",
        cv=3,
        verbose=2,
        random_state=RANDOM_STATE,
        n_jobs=1   # keep stable on EC2
    )

    lgbm_search.fit(X_train, y_train)

    best_lgbm = lgbm_search.best_estimator_
    joblib.dump(best_lgbm, "models/TuneLightGBM.joblib")
    print("Saved -> models/TuneLightGBM.joblib")
    print("Best LightGBM params:", lgbm_search.best_params_)

    # ----------------------------
    # Tune XGBoost
    # ----------------------------
    print("\n[5] Tuning XGBoost...")
    xgb = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    xgb_pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", xgb)
    ])

    xgb_params = {
        "model__n_estimators": randint(600, 2500),
        "model__learning_rate": uniform(0.01, 0.09),   # 0.01 to 0.10
        "model__max_depth": randint(3, 12),
        "model__min_child_weight": randint(1, 20),
        "model__subsample": uniform(0.6, 0.4),
        "model__colsample_bytree": uniform(0.6, 0.4),
        "model__reg_alpha": uniform(0.0, 1.0),
        "model__reg_lambda": uniform(0.5, 3.0),
        "model__gamma": uniform(0.0, 2.0),
    }

    xgb_search = RandomizedSearchCV(
        estimator=xgb_pipe,
        param_distributions=xgb_params,
        n_iter=30,
        scoring="neg_mean_squared_error",
        cv=3,
        verbose=2,
        random_state=RANDOM_STATE,
        n_jobs=1
    )

    xgb_search.fit(X_train, y_train)

    best_xgb = xgb_search.best_estimator_
    joblib.dump(best_xgb, "models/TuneXGBoost.joblib")
    print("Saved -> models/TuneXGBoost.joblib")
    print("Best XGBoost params:", xgb_search.best_params_)

    # ----------------------------
    # Final evaluation on the same test set
    # ----------------------------
    print("\n[6] Evaluate tuned models on test set...")
    results = []
    results.append(evaluate("TuneLightGBM", best_lgbm, X_test, y_test))
    results.append(evaluate("TuneXGBoost", best_xgb, X_test, y_test))

    df_res = pd.DataFrame(results).sort_values("RMSE")
    print("\n[TUNED RESULTS]")
    print(df_res.to_string(index=False))

    df_res.to_csv("results/tuned_model_results.csv", index=False)
    print("\nSaved -> results/tuned_model_results.csv")


if __name__ == "__main__":
    main()
