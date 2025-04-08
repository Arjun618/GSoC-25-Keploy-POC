"""
Sample utility functions for data processing and analysis.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union, Optional

def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load a dataset from a file.
    
    Args:
        file_path: Path to the data file (csv, json, etc.)
        
    Returns:
        DataFrame containing the data
    """
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        return pd.read_json(file_path)
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        return pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess a DataFrame by handling missing values and normalizing numeric columns.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Preprocessed DataFrame
    """
    # Make a copy to avoid modifying the original
    result = df.copy()
    
    # Fill missing values
    for col in result.columns:
        if pd.api.types.is_numeric_dtype(result[col]):
            # Fill numeric columns with mean
            result[col] = result[col].fillna(result[col].mean())
        else:
            # Fill non-numeric columns with mode
            result[col] = result[col].fillna(result[col].mode()[0] if not result[col].mode().empty else "unknown")
    
    # Normalize numeric columns
    for col in result.select_dtypes(include=['float64', 'int64']).columns:
        min_val = result[col].min()
        max_val = result[col].max()
        if max_val > min_val:
            result[col] = (result[col] - min_val) / (max_val - min_val)
    
    return result

def calculate_statistics(data: np.ndarray) -> Dict[str, float]:
    """
    Calculate basic statistics for a numpy array.
    
    Args:
        data: Input data array
        
    Returns:
        Dictionary of statistics
    """
    return {
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "std": float(np.std(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "q1": float(np.percentile(data, 25)),
        "q3": float(np.percentile(data, 75))
    }

def find_outliers(data: np.ndarray, method: str = "iqr") -> np.ndarray:
    """
    Find outliers in a dataset.
    
    Args:
        data: Input data array
        method: Method to use for outlier detection ("iqr" or "zscore")
        
    Returns:
        Boolean array indicating outliers
    """
    if method == "iqr":
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return (data < lower_bound) | (data > upper_bound)
    elif method == "zscore":
        z_scores = (data - np.mean(data)) / np.std(data)
        return np.abs(z_scores) > 3
    else:
        raise ValueError(f"Unsupported method: {method}")

def encode_categorical(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    One-hot encode categorical columns in a DataFrame.
    
    Args:
        df: Input DataFrame
        columns: List of categorical columns to encode
        
    Returns:
        DataFrame with encoded columns
    """
    result = df.copy()
    for col in columns:
        if col in result.columns:
            dummies = pd.get_dummies(result[col], prefix=col, drop_first=False)
            result = pd.concat([result, dummies], axis=1)
            result = result.drop(col, axis=1)
    return result

def split_train_test(df: pd.DataFrame, test_size: float = 0.2, random_state: Optional[int] = None) -> tuple:
    """
    Split a DataFrame into training and testing sets.
    
    Args:
        df: Input DataFrame
        test_size: Proportion of the data to include in the test split
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, test_df)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Shuffle the DataFrame
    df_shuffled = df.sample(frac=1)
    
    # Split the DataFrame
    train_size = int(len(df) * (1 - test_size))
    train_df = df_shuffled.iloc[:train_size]
    test_df = df_shuffled.iloc[train_size:]
    
    return train_df, test_df