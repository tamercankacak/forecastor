import os
import joblib
import pandas as pd
from typing import Dict, Any

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor

# Ensure we can import from the src directory
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import DataPreprocessor

class ModelTrainer:
    """
    Trains, tunes, and saves baseline and challenger forecasting models.
    """
    def __init__(self, X_train, y_train, models_dir: str = "models"):
        """
        Initializes the ModelTrainer.

        Args:
            X_train: Training feature data.
            y_train: Training target data.
            models_dir (str): Directory to save trained models.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.models_dir = models_dir
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

    def train_linear_regression(self) -> LinearRegression:
        """Trains and saves a Linear Regression model."""
        print("Training Linear Regression model...")
        model = LinearRegression()
        model.fit(self.X_train, self.y_train)
        
        model_path = os.path.join(self.models_dir, "linear_regression.joblib")
        joblib.dump(model, model_path)
        print(f"Linear Regression model saved to {model_path}")
        return model

    def train_random_forest(self) -> RandomForestRegressor:
        """Trains, tunes, and saves a RandomForest Regressor model."""
        print("Training RandomForest Regressor model (light configuration)...")
        model = RandomForestRegressor(random_state=42)
        
        # Greatly reduced parameter grid for fast tuning
        param_dist = {
            'n_estimators': [10, 20],
            'max_depth': [10]
        }
        
        # Use minimal settings for n_iter and cv to ensure completion
        random_search = RandomizedSearchCV(
            model, param_distributions=param_dist, n_iter=2, cv=2, 
            scoring='neg_root_mean_squared_error', n_jobs=-1, random_state=42, verbose=1
        )
        random_search.fit(self.X_train, self.y_train)
        
        best_model = random_search.best_estimator_
        model_path = os.path.join(self.models_dir, "random_forest.joblib")
        joblib.dump(best_model, model_path)
        print(f"Best RandomForest model saved to {model_path}")
        return best_model

    def train_xgboost(self) -> XGBRegressor:
        """Trains, tunes, and saves an XGBoost Regressor model."""
        print("Training XGBoost Regressor model (light configuration)...")
        model = XGBRegressor(random_state=42)

        # Greatly reduced parameter grid for fast tuning
        param_dist = {
            'n_estimators': [50],
            'learning_rate': [0.1],
            'max_depth': [3, 5]
        }
        
        # Use minimal settings for n_iter and cv to ensure completion
        random_search = RandomizedSearchCV(
            model, param_distributions=param_dist, n_iter=2, cv=2, 
            scoring='neg_root_mean_squared_error', n_jobs=-1, random_state=42, verbose=1
        )
        random_search.fit(self.X_train, self.y_train)
        
        best_model = random_search.best_estimator_
        model_path = os.path.join(self.models_dir, "xgboost.joblib")
        joblib.dump(best_model, model_path)
        print(f"Best XGBoost model saved to {model_path}")
        return best_model

    def train_all_models(self) -> Dict[str, Any]:
        """Trains all defined models."""
        models = {
            "LinearRegression": self.train_linear_regression(),
            "RandomForest": self.train_random_forest(),
            "XGBoost": self.train_xgboost()
        }
        print("\nAll models have been trained and saved.")
        return models

def main():
    """
    Main function to run the full preprocessing and model training pipeline.
    """
    print("--- Starting Phase 3: Model Development & Benchmarking ---")
    
    # --- Step 1: Data Preprocessing ---
    print("\nRunning data preprocessor...")
    DATA_FILE = 'supply_chain_data.csv'
    preprocessor = DataPreprocessor(file_path=DATA_FILE)
    X_train, X_test, y_train, y_test = preprocessor.preprocess()

    # --- Step 2: Model Training ---
    print("\nInitializing model trainer...")
    trainer = ModelTrainer(X_train=X_train, y_train=y_train)
    trainer.train_all_models()
    
    print("\n--- Phase 3 Complete ---")

if __name__ == '__main__':
    main()
