# Supply Chain AI Forecasting System

This project is a research prototype developed to demonstrate the integration of AI-driven forecasting into supply chain management workflows. It compares the performance of traditional statistical methods (Linear Regression) against modern machine learning models (Random Forest, XGBoost) for sales forecasting.

## Project Structure

```
.
├── models/
│   ├── linear_regression.joblib  # Saved baseline model
│   ├── random_forest.joblib      # Saved challenger model
│   └── xgboost.joblib            # Saved challenger model
├── reports/
│   └── figures/
│       └── actual_vs_predicted.png # Evaluation plot
├── src/
│   ├── preprocessing.py          # DataPreprocessor class
│   └── models.py                 # ModelTrainer class
├── tests/
│   ├── sample_data.csv           # Small dataset for testing
│   └── test_preprocessing.py     # Unit tests for the preprocessor
├── .gitignore
├── evaluate.py                   # Script to evaluate trained models
├── project.md                    # Original project brief
└── README.md                     # This file
└── requirements.txt              # Python dependencies
```

## Setup and Installation

This project uses a Python virtual environment to manage dependencies.

### 1. Provide the Data

This project requires a data file named `supply_chain_data.csv` to be placed in the root of the project directory. The file must contain the following columns: `Date`, `Product_ID`, `Region`, `Sales_Quantity`, `Inventory_Level`.

### 2. Create a Virtual Environment

First, ensure you have the necessary package for creating virtual environments. On Debian/Ubuntu, you can install it with:
```bash
sudo apt-get install python3-venv
```

Then, create the virtual environment in the project root:
```bash
python3 -m venv .venv
```

### 3. Install Dependencies

Install all required packages from the `requirements.txt` file:
```bash
.venv/bin/pip install -r requirements.txt
```

## How to Run the Workflow

Follow these steps in order to train the models and evaluate the results.

### Step 1: Train the Forecasting Models

Run the `src/models.py` script. This script will first preprocess the data from `supply_chain_data.csv` and then train three models: Linear Regression, Random Forest, and XGBoost. The trained models will be saved to the `models/` directory.

```bash
.venv/bin/python src/models.py
```
*(Note: The model training, especially the hyperparameter tuning for RandomForest and XGBoost, can be time-consuming.)*

### Step 2: Evaluate Model Performance

Run the `evaluate.py` script to see how the models performed. This script loads the test data and the saved models, calculates performance metrics, and generates a visualization.

```bash
.venv/bin/python evaluate.py
```

## Expected Results

When you run the evaluation script, you will see two outputs:

1.  **A Performance Comparison Table** printed to the console. This table shows the Root Mean Squared Error (RMSE), Mean Absolute Percentage Error (MAPE), and R-squared (R2) score for each model.

    Example Output:
    ```
    --- Model Performance Comparison ---
                            RMSE      MAPE (%)  R2 Score
    Model
    Linear Regression  48.620533  6.819961e+15  0.447474
    Random Forest      48.610102  6.871273e+15  0.447711
    Xgboost            48.531922  6.825558e+15  0.449486
    ```
    *(Note: The high MAPE value is likely due to division by very small actual values in the test set and can be ignored in favor of RMSE and R2 for this analysis.)*

2.  **An "Actual vs. Predicted" Plot** saved to `reports/figures/actual_vs_predicted.png`. This scatter plot visually compares the model's predicted sales quantities against the actual values for the test set, providing a quick qualitative assessment of its performance.

## Running Tests

Unit tests have been set up using `pytest`. To run the tests, execute the following command:
```bash
.venv/bin/pytest tests/
```
