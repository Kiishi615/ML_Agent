import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ACTIVE_DATAFRAMES = {}

def check_state(session_id: int, required_keys: list) -> dict:
    """READ: Safely fetches data."""
    if session_id not in ACTIVE_DATAFRAMES:
        return {"error": f"No session {session_id} found."}
    
    state = ACTIVE_DATAFRAMES[session_id]
    
    if required_keys:
        missing = [key for key in required_keys if key not in state]
        if missing:
            return {"error": f"Missing required data: {missing}."}
            
    return state

def get_df(session_id: int, df_name: str = "main"):
    try:
        return ACTIVE_DATAFRAMES[session_id][df_name]
    except KeyError:
        return None

# [x] 1. load_dataset (DONE)
def load_dataset(filepath: str, session_id: int, df_name: str ="main") -> dict:
    """
    Load a CSV file into memory. CALL THIS FIRST before anything else.
    Requires: filepath to a CSV file.
    Stores: DataFrame as 'main' (or custom df_name).
    Returns: row count, column count, column names, dtypes.
    
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        return {"error": f"File not found: {filepath}"}
    except pd.errors.EmptyDataError:
        return {"error": "File is empty"}
    except Exception as e:
        return {"error": f"Failed to load: {str(e)}"}


    if session_id not in ACTIVE_DATAFRAMES:
        ACTIVE_DATAFRAMES[session_id]= {}
    
    ACTIVE_DATAFRAMES[session_id][df_name] = df.copy()

    dtypes_dict = {col: str(dtype) for col, dtype in df.dtypes.items()}

    return {
            "status": f"Loaded {df_name}",
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": df.columns.tolist(),
            "dtypes": dtypes_dict
            }

# [x] 2. get_basic_info
def get_basic_info(session_id: int, df_name: str = "main") -> dict:
    """Inspect current state of a dataframe. Call anytime to see shape, types, missing values, and preview.
        Requires: load_dataset must have been called.
        Returns: shape, dtypes, missing count, missing percent, first 5 rows, summary statistics."""

    
    df = get_df(session_id, df_name)
    if df is None:
        return {"error": f"No dataframe '{df_name}' or session id {session_id} found"}
    return{
        "shape": {"rows": df.shape[0], "columns": df.shape[1]},
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing": df.isnull().sum().to_dict(),
        "missing_percent": (df.isnull().mean() * 100).round(2).to_dict(),
        "head": df.head().to_dict("records"),
        "summary": df.describe().to_dict()
        }

# [x] 3. identify_target_column
def identify_target_column(session_id: int, target:str, df_name: str = "main") -> dict:
    """
    Find which column is the prediction target. Auto-detects common names or accepts user-specified target.
    Requires: load_dataset must have been called.
    Returns: target column name and detection method, or list of available columns if not found.
    """
    df = get_df(session_id, df_name)
    if df is None:
        return {"error": f"No dataframe '{df_name}' or session id {session_id} found"}
    
    if target and target in df.columns:
        return {
            "target": target,
            "method": "user_specified"
            }
    
    common_targets = [
        "target", "label", "class", "churn", "attrition",
        "price", "salary", "survived", "outcome", "y", "output"
    ]

    for name in common_targets:
        for col in df.columns:
            if col.lower().strip() == name:
                return {
                "target": col,
                "method": "auto_detected"
                }
    
    return {
        "target": None,
        "method": "not_found",
        "available_columns": list(df.columns),
        "message": "Could not auto-detect. Ask the user to pick from available columns."
    }

# [x] 4. separate_features_and_target
def separate_features_and_target(session_id: int, target_column: str, df_name: str = "main") -> dict:
    """
    Split dataframe into X (features) and y (target). MUST call identify_target_column first to know which column.
    Requires: load_dataset, identify_target_column.
    Stores: 'X' and 'y' separately.
    Returns: shapes of X and y.
    """

    
    df = get_df(session_id, df_name)
    if df is None:
        return {"error": f"No dataframe '{df_name}' found"}
    
    if target_column not in df.columns:
        return {"error": f"Column '{target_column}' not found. Available: {df.columns.tolist()}"}
    y = df[target_column]
    X = df.drop(columns= target_column)

    ACTIVE_DATAFRAMES[session_id]["X"]= X
    ACTIVE_DATAFRAMES[session_id]["y"]= y

    return{
        "status": f"Successfully separated into X and y.",
        "X_shape": X.shape,  
        "y_shape": y.shape
    }

# [x] 5. split_data
def split_data(session_id: int, test_size: float = 0.2) -> dict:
    """
    Split X and y into training and test sets. Auto-stratifies for classification.
    Requires: separate_features_and_target must have been called.
    Stores: X_train, X_test, y_train, y_test.
    Returns: shapes of all splits, whether stratification was used.
    """
    if session_id not in ACTIVE_DATAFRAMES:
        return {"error": f"No session {session_id} found"}


    if "X" not in ACTIVE_DATAFRAMES[session_id] or "y" not in ACTIVE_DATAFRAMES[session_id]:
        return {"error": "X and y not found. Run separate_features_and_target first."}

    X = ACTIVE_DATAFRAMES[session_id]["X"]
    y = ACTIVE_DATAFRAMES[session_id]["y"]

    if y.dtype == 'object' or y.nunique() < 20:
        stratify = y
    else:
        stratify = None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=stratify, random_state=42
    )

    ACTIVE_DATAFRAMES[session_id]["X_train"] = X_train
    ACTIVE_DATAFRAMES[session_id]["X_test"] = X_test
    ACTIVE_DATAFRAMES[session_id]["y_train"] = y_train
    ACTIVE_DATAFRAMES[session_id]["y_test"] = y_test

    return {
        "status": "Split complete",
        "stratified": stratify is not None,
        "X_train_shape": X_train.shape,
        "X_test_shape": X_test.shape,
        "y_train_shape": y_train.shape,
        "y_test_shape": y_test.shape
    }

# [x] 6. train_single_model
def train_single_model(session_id: int) -> dict:
    """
    Train a LogisticRegression model. ALL features must be numeric — if not, run encode_onehot_features first.
    Requires: split_data must have been called. All columns in X must be numeric.
    Stores: trained_model, y_pred.
    Returns: accuracy, classification report.
    """
    if session_id not in ACTIVE_DATAFRAMES:
        return {"error": f"No session {session_id} found"}
    
    required = ["X_train", "X_test", "y_train", "y_test"]
    missing = [key for key in required if key not in ACTIVE_DATAFRAMES[session_id]]
    if missing:
        return {"error": f"{missing} not found. Run split_data first."}
    
    
    
    X_train = ACTIVE_DATAFRAMES[session_id]["X_train"]
    X_test = ACTIVE_DATAFRAMES[session_id]["X_test"] 
    y_train = ACTIVE_DATAFRAMES[session_id]["y_train"]
    y_test= ACTIVE_DATAFRAMES[session_id]["y_test"]

    non_numeric = X_train.select_dtypes(exclude='number').columns.tolist()
    if non_numeric:
        return {"error": f"Non-numeric columns found: {non_numeric}. Run encode_onehot_features first."}


    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    ACTIVE_DATAFRAMES[session_id]["trained_model"] = model
    ACTIVE_DATAFRAMES[session_id]["y_pred"] = y_pred

    return {
        "status" : "Training succesfully",
        "model" : "Logistical Regression",
        "accuracy": f"{accuracy_score(y_test, y_pred)}",
        "classification_report": f"{classification_report(y_test, y_pred, output_dict=True)}"
    }

# [x] 7. generate_predictions
def generate_predictions(session_id: int, n_predictions: int = 10) -> dict:
    """
    Show a side-by-side preview of predicted vs actual values on test data.
    Requires: train_single_model must have been called.
    Returns: lists of predicted and actual values for first N rows.
    """
    if session_id not in ACTIVE_DATAFRAMES:
        return {"error": f"No session {session_id} found"}
    
    required = ["trained_model", "X_test", "y_test"]
    missing = [key for key in required if key not in ACTIVE_DATAFRAMES[session_id]]
    if missing:
        return {"error": f"{missing} not found. Run train_single_model first."}

    X_test = ACTIVE_DATAFRAMES[session_id]["X_test"].head(n_predictions)
    y_test = ACTIVE_DATAFRAMES[session_id]["y_test"].head(n_predictions)
    model = ACTIVE_DATAFRAMES[session_id]["trained_model"]
    y_pred = model.predict(X_test)

    return {
        "status" : "generated predictions successfuly",
        "predicted_y_values": y_pred.tolist(),
        "actual_y_values": y_test.tolist()
    }

# [x] 8. handle_missing_features
def handle_missing_features(session_id: int, target_column: str, df_name: str = "main") -> dict:
    """Fill all missing feature values. Median for numbers, mode for categories.
    Run this BEFORE encode_categorical or train_single_model."""
    if session_id not in ACTIVE_DATAFRAMES:
        return {"error": f"No session {session_id} found"}

    df = get_df(session_id, df_name)
    if df is None:
        return {"error": f"No dataframe '{df_name}' found"}
    
    filled = {}

    num_cols = df.select_dtypes(include='number').columns
    for col in num_cols:
        if col == target_column:
                continue
        if df[col].isnull().sum()>0:
            median_val = df[col].median()
            fill_val = 0 if pd.isna(median_val) else median_val
            
            df[col] = df[col].fillna(fill_val)
            filled[col] = f"median ({fill_val})"    
    
    cat_cols = df.select_dtypes(exclude='number').columns
    for col in cat_cols:
            if col == target_column:
                continue
            mode_series = df[col].mode()

            if len(mode_series) > 0:
                fill_val = mode_series.iloc[0]
                filled[col] = f"mode ({fill_val})"
            else:
                fill_val = "Unknown"
                filled[col] = "placeholder ('Unknown')"
                
            df[col] = df[col].fillna(fill_val)


    ACTIVE_DATAFRAMES[session_id][df_name] = df
    return {
        "status": "Missing values handled",
        "filled": filled,
        "remaining_nulls": df.isnull().sum().sum()
    }

# [x] 9. encode_categorical
def encode_categorical(session_id: int, target_column: str, df_name: str = "main", max_unique_values: int = 15) -> dict:
    """One-hot encode categorical columns. Label encode if too many unique values.
    Run this AFTER handle_missing_values, BEFORE split_data."""

    if session_id not in ACTIVE_DATAFRAMES:
        return {"error": f"No session {session_id} found"}

    df = get_df(session_id, df_name)
    if df is None:
        return {"error": f"No dataframe '{df_name}' found"}

    prev_shape = df.shape
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()
    
    if target_column in cat_cols:
        cat_cols.remove(target_column)
        le_target = LabelEncoder()
        df[target_column] = le_target.fit_transform(df[target_column].astype(str))

    onehot_cols = []
    label_cols = []

    for col in cat_cols:
        if df[col].nunique() <= max_unique_values:
            onehot_cols.append(col)
        else:
            label_cols.append(col)

    # One-hot encode low cardinality
    if onehot_cols:
        df = pd.get_dummies(df, columns=onehot_cols, drop_first=True)

    le = LabelEncoder()
    for col in label_cols:
        df[col] = le.fit_transform(df[col].astype(str)) # type: ignore


    ACTIVE_DATAFRAMES[session_id][df_name] = df
    return {
        "status": "Categorical data encoded",
        "onehot_encoded": onehot_cols,
        "label_encoded": label_cols,
        "previous_shape": prev_shape,
        "current_shape": df.shape
    }

# [x] 10. drop_missing_target_rows
def drop_missing_target_rows(session_id: int, target_column: str, df_name: str = "main")-> dict:
    """
    Drop rows where the target column is null. Run this BEFORE separate_features_and_target.
    Requires: load_dataset and identify_target_column.
    """
    if session_id not in ACTIVE_DATAFRAMES:
        return {"error": f"No session {session_id} found"}

    df = get_df(session_id, df_name)
    if df is None:
        return {"error": f"No dataframe '{df_name}' found"}
    
    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found. Available: {df.columns.tolist()}"}
    
    previous_row_count = len(df)
    if df[target_column].isnull().sum() == 0:
        return {"status": "No missing targets found"}
    
    df = df.dropna(subset=[target_column])

    current_row_count = len(df)
    ACTIVE_DATAFRAMES[session_id][df_name] = df

    return {
        "status": "Dropped rows successfully",
        "previous_row_count": previous_row_count,
        "current_row_count": current_row_count
    }

# [x] 11. drop_hig_cardinalty_columns
def drop_high_cardinality_columns(session_id: int, target_column: str, df_name: str = "main", threshold: float = 0.8) -> dict: 
    """
    Drops categorical columns that are almost entirely unique (like IDs, Names, or Hashes).
    Run this BEFORE encode_categorical.
    """
    if session_id not in ACTIVE_DATAFRAMES:
        return {"error": f"No session {session_id} found"}

    df = get_df(session_id, df_name)
    if df is None:
        return {"error": f"No dataframe '{df_name}' found"}

    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found."}

    cat_cols = df.select_dtypes(exclude="number").columns.tolist()
    
    if target_column in cat_cols:
        cat_cols.remove(target_column)

    dropped_cols = []
    total_rows = len(df)

    for col in cat_cols:
        unique_ratio = df[col].nunique() / total_rows
        
        if unique_ratio >= threshold:
            dropped_cols.append(col)

    if dropped_cols:
        df = df.drop(columns=dropped_cols)
        ACTIVE_DATAFRAMES[session_id][df_name] = df

    return {
        "status": "High cardinality check complete",
        "threshold_used": threshold,
        "columns_dropped": dropped_cols,
        "remaining_columns": df.columns.tolist()
    }