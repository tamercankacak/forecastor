Master Instruction: Supply Chain AI Forecasting System (Research Prototype)
Role: Act as a Senior Machine Learning Engineer and Data Architect specializing in Supply Chain AI and ERP integrations.

Context: I am conducting Master's level research titled "Supply Chain Efficiency Enhancement by integration of Artificial Intelligence." The goal is to prove that AI-driven forecasting (XGBoost/Random Forest) outperforms traditional statistical methods (Linear Regression) and can be integrated into ERP workflows to optimize inventory (Safety Stock/Reorder Points).

Source Material Reference:

Objective: Develop a Python-based forecasting module.

Data: Historical sales/movement data (Time-series).

Methods: Quantitative analysis, Feature Engineering (lags, seasonality), Model Benchmarking.

Metrics: RMSE, MAPE, R².

Technical Constraints & Stack:

Language: Python 3.9+

Libraries: pandas, numpy, scikit-learn (LinearRegression, RandomForest), xgboost, matplotlib/seaborn (for viz), joblib (saving models).

Style: Modular object-oriented code (Classes), Type Hinting, PEP8 compliance, and comprehensive Docstrings.

Environment: Assume a local environment (CLI execution).

Mission: Implementation Roadmap
You will execute this project in 5 distinct phases. Do not write all the code at once. Wait for my confirmation after each phase to proceed to the next.

Phase 1: Project Structure & Synthetic Data Generation
Goal: Create the directory structure and a script to generate a synthetic dataset that mimics the "US Government Data" described in the research (approx. 300k records).

Data Schema:

Date: Daily frequency (2017–2025).

Product_ID: Categorical (e.g., 50 unique products).

Region: Categorical (e.g., 5 regions).

Sales_Quantity: Numerical (Target variable). Incorporate seasonality and trend noise to make it realistic.

Inventory_Level: Numerical (Current stock).

Output: A Python script generate_data.py that outputs supply_chain_data.csv.

Phase 2: Data Preprocessing Pipeline
Goal: Implement the "Data Preprocessing" methodology outlined in the proposal.

Tasks:

Load data using pandas.

Cleaning: Handle missing values (Median imputation) and duplicates.

Feature Engineering:

Extract temporal features: Month, Week_of_Year, Day_of_Week.

Create Lag features: Sales_Lag_7, Sales_Lag_30.

Rolling averages: Rolling_Mean_7.

Transformation:

OneHotEncoder for Categorical variables (Region, Product_ID).

MinMaxScaler for continuous features.

Output: A class DataPreprocessor in src/preprocessing.py that returns X_train, X_test, y_train, y_test.

Phase 3: Model Development & Benchmarking
Goal: Train the Baseline and Challenger models as defined in the research methodology.

Tasks:

Baseline Model: LinearRegression.

Challenger Models: XGBoostRegressor and RandomForestRegressor.

Training: Implement a training loop. Use GridSearchCV (or RandomizedSearchCV) for the Challenger models to optimize hyperparameters (prevent overfitting).

Output: A class ModelTrainer in src/models.py.

Phase 4: Evaluation & Reporting Module
Goal: Evaluate performance using the specific metrics from the research proposal.

Metrics:

RMSE (Root Mean Square Error).

MAPE (Mean Absolute Percentage Error).

R² Score.

Visualization: Generate a plot comparing Actual vs. Predicted values (like Figure 3 in the research proposal).

Output: A script evaluate.py that prints a comparison table and saves a .png chart.

Phase 5: ERP Integration Prototype (The "Deliverable")
Goal: Simulate the "Operationalization" of the forecast into an ERP object.

Logic:

Take the Forecasted Demand for the next period.

Calculate Safety Stock and Reorder Point based on the forecast error (standard deviation of residuals) calculated in Phase 4.

Generate an "MRP Exception Report" (CSV) flagging products that need reordering.

Output: erp_integration.py.

Immediate Next Step: Please acknowledge you understand the roadmap and the research context. Then, generate the code for Phase 1: Project Structure & Synthetic Data Generation.