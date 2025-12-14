
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# Ensure we can import from the src directory
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from preprocessing import DataPreprocessor

# --- Configuration ---
DATA_FILE = 'supply_chain_data.csv'
MODELS_DIR = 'models'
REPORTS_DIR = 'reports'
FIGURES_DIR = os.path.join(REPORTS_DIR, 'figures')
SCATTER_PLOT_PATH = os.path.join(FIGURES_DIR, 'actual_vs_predicted.png')
TIME_SERIES_PLOT_PATH = os.path.join(FIGURES_DIR, 'time_series_forecast.png')

# --- Metric Functions ---
def calculate_rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    """Calculates the Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_mape(y_true: pd.Series, y_pred: np.ndarray) -> float:
    """
    Calculates the Mean Absolute Percentage Error.
    Handles cases where y_true might be zero by adding a small epsilon.
    """
    y_true_np, y_pred_np = np.array(y_true), np.array(y_pred)
    epsilon = np.finfo(np.float64).eps
    return np.mean(np.abs((y_true_np - y_pred_np) / (y_true_np + epsilon))) * 100

def calculate_r2(y_true: pd.Series, y_pred: np.ndarray) -> float:
    """Calculates the R-squared score."""
    return r2_score(y_true, y_pred)

def generate_scatter_plot(y_test, models, X_test):
    """Generates and saves the Actual vs. Predicted scatter plot."""
    print("\nGenerating Actual vs. Predicted scatter plot...")
    plt.figure(figsize=(12, 8))
    
    min_val, max_val = y_test.min(), y_test.max()
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
    
    for i, (model_key, model) in enumerate(models.items()):
        predictions = np.maximum(0, model.predict(X_test))
        min_val, max_val = min(min_val, predictions.min()), max(max_val, predictions.max())
        
        plt.scatter(y_test, predictions, alpha=0.3, s=10, 
                    color=colors[i % len(colors)], label=f'{model_key.replace("_", " ").title()} Predictions')

    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Prediction')
    plt.xlabel("Actual Sales Quantity")
    plt.ylabel("Predicted Sales Quantity")
    plt.title("Actual vs. Predicted Sales Quantity Comparison (Test Set)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(SCATTER_PLOT_PATH)
    print(f"Scatter plot saved to {SCATTER_PLOT_PATH}")

def generate_time_series_plot(y_test, test_dates, models, X_test):
    """Generates and saves the monthly time-series forecast plot."""
    print("\nGenerating monthly time-series forecast plot...")
    
    # Create a dataframe for time series analysis
    ts_df = pd.DataFrame({'Date': test_dates, 'Actual': y_test})
    for model_key, model in models.items():
        predictions = np.maximum(0, model.predict(X_test))
        ts_df[f'{model_key.replace("_", " ").title()}'] = predictions
        
    # Aggregate data by month (using mean of daily sales)
    monthly_df = ts_df.set_index('Date').resample('ME').mean()
    
    plt.figure(figsize=(15, 7))
    plt.plot(monthly_df.index, monthly_df['Actual'], label='Actual Sales', color='black', linewidth=2)
    
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
    for i, col in enumerate(monthly_df.columns.drop('Actual')):
        plt.plot(monthly_df.index, monthly_df[col], label=f'{col} Forecast', 
                 color=colors[i % len(colors)], linestyle='--')
        
    plt.xlabel("Month")
    plt.ylabel("Average Daily Sales Quantity")
    plt.title("Monthly Forecast vs. Actual Sales")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(TIME_SERIES_PLOT_PATH)
    print(f"Time-series plot saved to {TIME_SERIES_PLOT_PATH}")

def main():
    print("--- Starting Evaluation Module ---")
    
    # 1. Load data
    print("Loading and preprocessing data...")
    preprocessor = DataPreprocessor(file_path=DATA_FILE)
    _, X_test, _, y_test, test_dates = preprocessor.preprocess() 

    # 2. Load models
    print("Loading trained models...")
    models = {}
    for model_file in sorted(os.listdir(MODELS_DIR)):
        if model_file.endswith(".joblib"):
            model_name_key = model_file.replace(".joblib", "")
            models[model_name_key] = joblib.load(os.path.join(MODELS_DIR, model_file))

    # 3. Evaluate each model and print metrics
    print("Evaluating models and calculating metrics...")
    results = []
    for model_key, model in models.items():
        predictions = np.maximum(0, model.predict(X_test))
        results.append({
            "Model": model_key.replace("_", " ").title(),
            "RMSE": calculate_rmse(y_test, predictions),
            "MAPE (%)": calculate_mape(y_test, predictions),
            "R2 Score": calculate_r2(y_test, predictions)
        })
    results_df = pd.DataFrame(results).set_index("Model")
    print("\n--- Model Performance Comparison ---")
    print(results_df.to_string())

    # 4. Generate and save plots
    if not os.path.exists(FIGURES_DIR):
        os.makedirs(FIGURES_DIR)
    
    generate_scatter_plot(y_test, models, X_test)
    generate_time_series_plot(y_test, test_dates, models, X_test)

    print("\n--- Evaluation Complete ---")

if __name__ == "__main__":
    main()

