
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from typing import Tuple

class DataPreprocessor:
    """
    A class for preprocessing supply chain data for forecasting models.
    """
    def __init__(self, file_path: str):
        """
        Initializes the DataPreprocessor.

        Args:
            file_path (str): The path to the data file.
        """
        self.file_path = file_path
        self.transformer = None

    def preprocess(self) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, pd.Series]:
        """
        Executes the full preprocessing pipeline.

        Returns:
            A tuple containing:
            - X_train_transformed (np.ndarray): Transformed training feature data.
            - X_test_transformed (np.ndarray): Transformed testing feature data.
            - y_train (pd.Series): Training target data.
            - y_test (pd.Series): Testing target data.
            - test_dates (pd.Series): The dates corresponding to the test set.
        """
        df = self._load_and_clean_data()
        df = self._feature_engineer(df)

        # Drop rows with NaNs created by lags/rolling features
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Separate date column for later use
        dates = df['Date']

        # Define features (X) and target (y)
        y = df['Sales_Quantity']
        X = df.drop(columns=['Sales_Quantity', 'Date'])

        # Split data chronologically
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False, random_state=42
        )

        # Get dates for the test set
        test_dates = dates.loc[y_test.index]

        # Define column types for transformation
        categorical_features = ['Product_ID', 'Region']
        numerical_features = X.select_dtypes(include=np.number).columns.tolist()

        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', MinMaxScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ],
            remainder='passthrough'
        )

        self.transformer = preprocessor.fit(X_train)
        X_train_transformed = self.transformer.transform(X_train)
        X_test_transformed = self.transformer.transform(X_test)
        
        print("Data preprocessing complete.")
        print(f"X_train shape: {X_train_transformed.shape}")
        print(f"X_test shape: {X_test_transformed.shape}")
        
        return X_train_transformed, X_test_transformed, y_train, y_test, test_dates

    def _load_and_clean_data(self) -> pd.DataFrame:
        """Loads and cleans the raw data."""
        df = pd.read_csv(self.file_path)
        
        # Convert Date column
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Handle missing values (median imputation for numeric)
        for col in df.select_dtypes(include=np.number).columns:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
        
        # Remove duplicates
        df.drop_duplicates(inplace=True)
        
        # Sort by date to ensure correct time series operations
        df.sort_values(by=['Region', 'Product_ID', 'Date'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        return df

    def _feature_engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Creates new features from existing ones."""
        # Temporal features
        df['Month'] = df['Date'].dt.month
        df['Week_of_Year'] = df['Date'].dt.isocalendar().week.astype(int)
        df['Day_of_Week'] = df['Date'].dt.dayofweek

        # Grouped features for lags and rolling means
        grouped = df.groupby(['Region', 'Product_ID'])['Sales_Quantity']
        
        # Lag features
        df['Sales_Lag_7'] = grouped.shift(7)
        df['Sales_Lag_30'] = grouped.shift(30)
        
        # Rolling mean
        df['Rolling_Mean_7'] = grouped.transform(lambda x: x.rolling(window=7).mean())
        
        return df

if __name__ == '__main__':
    # Example of how to use the class
    DATA_FILE = 'supply_chain_data.csv'
    
    preprocessor = DataPreprocessor(file_path=DATA_FILE)
    X_train, X_test, y_train, y_test = preprocessor.preprocess()
    
    print("\n--- Preprocessing Output ---")
    print(f"Training features shape: {X_train.shape}")
    print(f"Testing features shape: {X_test.shape}")
    print(f"Training target shape: {y_train.shape}")
    print(f"Testing target shape: {y_test.shape}")
    print("\nSample of transformed training data:")
    print(X_train[:5])
