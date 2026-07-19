import os
import warnings
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# =====================================================
# Configuration
# =====================================================

DATA = "india_tech_jobs.csv"
TARGET = "salary_lpa"


# =====================================================
# Load Dataset
# =====================================================

def load_data(path=DATA):
    df = pd.read_csv(path)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    return X, y


# =====================================================
# Preprocessing Pipeline
# =====================================================

def build_preprocessor():

    categorical_features = [
        "job_title",
        "education",
        "primary_skill",
        "city",
        "state",
        "company",
        "company_size",
        "work_mode",
    ]

    numerical_features = [
        "experience_years",
        "skills_count",
    ]

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_features),
            ("num", numerical_transformer, numerical_features),
        ]
    )

    return preprocessor


# =====================================================
# Evaluation Function
# =====================================================

def evaluate(model, X_test, y_test):

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    return mae, rmse, r2


# =====================================================
# Main
# =====================================================

if __name__ == "__main__":

    # Load data
    X, y = load_data()

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    # Build preprocessing pipeline
    preprocessor = build_preprocessor()

    # Models
    models = {
        "Linear Regression": LinearRegression(),

        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        ),

        "XGBoost": XGBRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        ),
    }

    

    best_model = None
    best_r2 = -1

    print("=" * 60)
    print("Training Models...")
    print("=" * 60)

    for name, algorithm in models.items():

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", algorithm),
            ]
        )

        pipeline.fit(X_train, y_train)

        mae, rmse, r2 = evaluate(
            pipeline,
            X_test,
            y_test,
        )

        print(f"\n{name}")
        print(f"MAE  : {mae:.3f}")
        print(f"RMSE : {rmse:.3f}")
        print(f"R²   : {r2:.4f}")

        if r2 > best_r2:
            best_r2 = r2
            best_model = pipeline

    # Save Best Model
    joblib.dump(best_model, "salary_model.pkl")

    print("\n" + "=" * 60)
    print("Best model saved successfully!")
    print("Location : salary_model.pkl")
    print("=" * 60)