
from src.data_loader import load_data
from src.preprocessing import (
    clean_data,
    engineer_features,
    build_preprocessor,
    TARGET,
    NUM_FEATURES,
    CAT_FEATURES
)
from src.models import get_models
from src.train import train_models


DATA_PATH = "data/raw/Flight data.csv"
def main():
    print("\n[STEP 1] Loading data...")
    df = load_data(DATA_PATH)
    print(f"Data shape: {df.shape}")

    print("\n[STEP 2] Cleaning data...")
    df = clean_data(df)

    print("\n[STEP 3] Feature engineering...")
    df = engineer_features(df)

    print("\n[STEP 4] Building preprocessing pipeline...")
    X = df[NUM_FEATURES + CAT_FEATURES]
    y = df[TARGET]

    preprocessor = build_preprocessor()
    models = get_models()

    print("\n[STEP 5] Starting training pipeline...")
    results = train_models(X, y, preprocessor, models)

    results.to_csv("results/model_comparison.csv", index=False)

    print("\n[FINAL RESULTS]")
    print(results.sort_values("RMSE"))



if __name__ == "__main__":
    main()


