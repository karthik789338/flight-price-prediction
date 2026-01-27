from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

def get_models():
    return {
        "LightGBM": LGBMRegressor(
            n_estimators=2500,
            learning_rate=0.03,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        ),

        "XGBoost": XGBRegressor(
            n_estimators=1800,
            learning_rate=0.03,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            objective="reg:squarederror",
            tree_method="hist"
        )
    }
