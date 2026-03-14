import os

import matplotlib

matplotlib.use('Agg')

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, auc, classification_report,
                             confusion_matrix, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

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

# [x] 1. load_dataset
def load_dataset(filepath: str, session_id: int, df_name: str = "main") -> dict:
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
        ACTIVE_DATAFRAMES[session_id] = {}

    ACTIVE_DATAFRAMES[session_id][df_name] = df.copy()

    return {
        "status": f"Loaded {df_name}",
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
    }


# [x] 2. get_basic_info
def get_basic_info(session_id: int, df_name: str = "main") -> dict:
    """
    Inspect current state of a dataframe. Call anytime to see shape, types, missing values, and preview.
    Requires: load_dataset must have been called.
    Returns: shape, dtypes, missing count, missing percent, first 5 rows, summary statistics.
    """
    state = check_state(session_id, [df_name])
    if "error" in state:
        return state

    df = state[df_name]

    return {
        "shape": {"rows": df.shape[0], "columns": df.shape[1]},
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing": df.isnull().sum().to_dict(),
        "missing_percent": (df.isnull().mean() * 100).round(2).to_dict(),
        "head": df.head().to_dict("records"),
        "summary": df.describe().to_dict()
    }


# [x] 3. identify_target_column
def identify_target_column(session_id: int, target: str , df_name: str = "main") -> dict:
    """
    Find which column is the prediction target. Auto-detects common names or accepts user-specified target.
    Requires: load_dataset must have been called.
    Returns: target column name and detection method, or list of available columns if not found.
    """
    state = check_state(session_id, [df_name])
    if "error" in state:
        return state

    df = state[df_name]

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
        "available_columns": df.columns.tolist(),
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
    state = check_state(session_id, [df_name])
    if "error" in state:
        return state

    df = state[df_name]

    if target_column not in df.columns:
        return {"error": f"Column '{target_column}' not found. Available: {df.columns.tolist()}"}

    y = df[target_column]
    X = df.drop(columns=target_column)

    ACTIVE_DATAFRAMES[session_id]["X"] = X
    ACTIVE_DATAFRAMES[session_id]["y"] = y

    return {
        "status": "Successfully separated into X and y.",
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
    state = check_state(session_id, ["X", "y"])
    if "error" in state:
        return state

    X = state["X"]
    y = state["y"]

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
    Train a LogisticRegression model. ALL features must be numeric — if not, run encode_categorical first.
    Requires: split_data must have been called. All columns in X must be numeric.
    Stores: trained_model, y_pred.
    Returns: accuracy, classification report.
    """
    state = check_state(session_id, ["X_train", "X_test", "y_train", "y_test"])
    if "error" in state:
        return state

    X_train = state["X_train"]
    X_test = state["X_test"]
    y_train = state["y_train"]
    y_test = state["y_test"]

    non_numeric = X_train.select_dtypes(exclude='number').columns.tolist()
    if non_numeric:
        return {"error": f"Non-numeric columns found: {non_numeric}. Run encode_categorical first."}

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    ACTIVE_DATAFRAMES[session_id]["trained_model"] = model
    ACTIVE_DATAFRAMES[session_id]["y_pred"] = y_pred

    return {
        "status": "Training complete",
        "model": "LogisticRegression",
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, output_dict=True)
    }


# [x] 7. generate_predictions
def generate_predictions(session_id: int, n_predictions: int = 10) -> dict:
    """
    Show a side-by-side preview of predicted vs actual values on test data.
    Requires: train_single_model must have been called.
    Returns: lists of predicted and actual values for first N rows.
    """
    state = check_state(session_id, ["trained_model", "X_test", "y_test"])
    if "error" in state:
        return state

    X_test = state["X_test"].head(n_predictions)
    y_test = state["y_test"].head(n_predictions)
    model = state["trained_model"]
    y_pred = model.predict(X_test)

    return {
        "status": "Generated predictions successfully",
        "predicted_y_values": y_pred.tolist(),
        "actual_y_values": y_test.tolist()
    }


# [x] 8. handle_missing_features
def handle_missing_features(session_id: int, target_column: str, df_name: str = "main") -> dict:
    """
    Fill all missing feature values. Median for numbers, mode for categories.
    Run this BEFORE encode_categorical or train_single_model.
    Requires: load_dataset must have been called.
    Stores: Overwrites dataframe with missing values filled.
    Returns: fill methods used per column, remaining null count.
    """
    state = check_state(session_id, [df_name])
    if "error" in state:
        return state

    df = state[df_name]

    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found."}

    filled = {}

    num_cols = df.select_dtypes(include='number').columns
    for col in num_cols:
        if col == target_column:
            continue
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            fill_val = 0 if pd.isna(median_val) else median_val
            df[col] = df[col].fillna(fill_val)
            filled[col] = f"median ({fill_val})"

    cat_cols = df.select_dtypes(exclude='number').columns
    for col in cat_cols:
        if col == target_column:
            continue
        if df[col].isnull().sum() > 0:
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
        "remaining_nulls": int(df.isnull().sum().sum())
    }


# [x] 9. encode_categorical
def encode_categorical(session_id: int, target_column: str, df_name: str = "main", max_unique_values: int = 15) -> dict:
    """
    One-hot encode categorical columns. Label encode if too many unique values.
    Run this AFTER handle_missing_features, BEFORE split_data.
    Requires: load_dataset must have been called.
    Stores: Overwrites dataframe with encoded columns.
    Returns: columns one-hot encoded, columns label encoded, shape before and after.
    """
    state = check_state(session_id, [df_name])
    if "error" in state:
        return state

    df = state[df_name]

    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found."}

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
def drop_missing_target_rows(session_id: int, target_column: str, df_name: str = "main") -> dict:
    """
    Drop rows where the target column is null. Run this BEFORE separate_features_and_target.
    Requires: load_dataset and identify_target_column.
    Stores: Overwrites dataframe with null-target rows removed.
    Returns: previous row count, current row count.
    """
    state = check_state(session_id, [df_name])
    if "error" in state:
        return state

    df = state[df_name]

    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found. Available: {df.columns.tolist()}"}

    previous_row_count = len(df)

    if df[target_column].isnull().sum() == 0:
        return {"status": "No missing targets found", "row_count": previous_row_count}

    df = df.dropna(subset=[target_column])
    ACTIVE_DATAFRAMES[session_id][df_name] = df

    return {
        "status": "Dropped rows successfully",
        "previous_row_count": previous_row_count,
        "current_row_count": len(df),
        "rows_removed": previous_row_count - len(df)
    }


# [x] 11. drop_high_cardinality_columns
def drop_high_cardinality_columns(session_id: int, target_column: str, df_name: str = "main", threshold: float = 0.8) -> dict:
    """
    Drops categorical columns that are almost entirely unique (like IDs, Names, or Hashes).
    Run this BEFORE encode_categorical.
    Requires: load_dataset must have been called.
    Stores: Overwrites dataframe with high-cardinality columns removed.
    Returns: threshold used, columns dropped, remaining columns.
    """
    state = check_state(session_id, [df_name])
    if "error" in state:
        return state

    df = state[df_name]

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

# [x] 12. drop_columns
def drop_columns(session_id: int, columns: list, df_name: str = "main") -> dict:
    """
    Drop one or more specific columns by name.
    Run this anytime BEFORE separate_features_and_target.
    Requires: load_dataset must have been called.
    Stores: Overwrites dataframe with specified columns removed.
    Returns: columns dropped, remaining columns.
    """
    state = check_state(session_id, [df_name])
    if "error" in state:
        return state

    df = state[df_name]

    not_found = [col for col in columns if col not in df.columns]
    if not_found:
        return {"error": f"Columns not found: {not_found}. Available: {df.columns.tolist()}"}

    df = df.drop(columns=columns)
    ACTIVE_DATAFRAMES[session_id][df_name] = df

    return {
        "status": "Columns dropped",
        "columns_dropped": columns,
        "remaining_columns": df.columns.tolist()
    }

# [x] 12. drop_duplicates
def drop_duplicates(session_id: int, df_name: str = "main") -> dict:
    """
    Remove duplicate rows from the dataframe.
    Run this AFTER load_dataset, BEFORE handle_missing_features.
    Requires: load_dataset must have been called.
    Stores: Overwrites dataframe with duplicates removed.
    Returns: previous row count, current row count, rows removed.
    """
    state = check_state(session_id, [df_name])
    if "error" in state:
        return state

    df = state[df_name]
    previous_row_count = len(df)
    df = df.drop_duplicates()
    ACTIVE_DATAFRAMES[session_id][df_name] = df

    return {
        "status": "Duplicates removed",
        "previous_row_count": previous_row_count,
        "current_row_count": len(df),
        "rows_removed": previous_row_count - len(df)
    }


# [x] 13. detect_outliers
def detect_outliers(session_id: int, target_column: str, df_name: str = "main") -> dict:
    """
    Detect outliers in numeric feature columns using the IQR method.
    Run this AFTER handle_missing_features, BEFORE remove_outliers.
    Requires: load_dataset must have been called.
    Returns: outlier counts per column, bounds, total outlier rows.
    """
    state = check_state(session_id, [df_name])
    if "error" in state:
        return state

    df = state[df_name]

    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found."}

    num_cols = df.select_dtypes(include='number').columns.tolist()
    if target_column in num_cols:
        num_cols.remove(target_column)

    outlier_info = {}
    outlier_rows = set()

    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        mask = (df[col] < lower) | (df[col] > upper)
        count = int(mask.sum())
        if count > 0:
            outlier_info[col] = {
                "count": count,
                "lower_bound": round(float(lower), 4),
                "upper_bound": round(float(upper), 4)
            }
            outlier_rows.update(df[mask].index.tolist())

    return {
        "status": "Outlier detection complete",
        "outlier_info": outlier_info,
        "total_outlier_rows": len(outlier_rows),
        "total_rows": len(df)
    }


# [x] 14. remove_outliers
def remove_outliers(session_id: int, target_column: str, df_name: str = "main") -> dict:
    """
    Remove rows containing outliers in any numeric feature column using the IQR method.
    Run this AFTER detect_outliers, BEFORE encode_categorical.
    Requires: load_dataset must have been called.
    Stores: Overwrites dataframe with outlier rows removed.
    Returns: previous row count, current row count, rows removed.
    """
    state = check_state(session_id, [df_name])
    if "error" in state:
        return state

    df = state[df_name]

    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found."}

    num_cols = df.select_dtypes(include='number').columns.tolist()
    if target_column in num_cols:
        num_cols.remove(target_column)

    previous_row_count = len(df)
    keep_mask = pd.Series(True, index=df.index)

    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        keep_mask = keep_mask & (df[col] >= lower) & (df[col] <= upper)

    df = df[keep_mask]
    ACTIVE_DATAFRAMES[session_id][df_name] = df

    return {
        "status": "Outliers removed",
        "previous_row_count": previous_row_count,
        "current_row_count": len(df),
        "rows_removed": previous_row_count - len(df)
    }


# [x] 15. scale_features
def scale_features(session_id: int) -> dict:
    """
    Standardize features to zero mean and unit variance. Fits on X_train, transforms both X_train and X_test.
    Run this AFTER split_data, BEFORE train_single_model.
    Requires: split_data must have been called.
    Stores: Overwrites X_train and X_test with scaled versions. Stores scaler.
    Returns: columns scaled, means and stds from training data.
    """
    state = check_state(session_id, ["X_train", "X_test"])
    if "error" in state:
        return state

    X_train = state["X_train"]
    X_test = state["X_test"]

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    ACTIVE_DATAFRAMES[session_id]["X_train"] = X_train_scaled
    ACTIVE_DATAFRAMES[session_id]["X_test"] = X_test_scaled
    ACTIVE_DATAFRAMES[session_id]["scaler"] = scaler

    return {
        "status": "Features scaled",
        "columns_scaled": X_train.columns.tolist(),
        "means": dict(zip(X_train.columns, scaler.mean_.round(4).tolist())),
        "stds": dict(zip(X_train.columns, scaler.scale_.round(4).tolist()))
    }


# [x] 16. compute_correlations
def compute_correlations(session_id: int, df_name: str = "main") -> dict:
    """
    Compute pairwise Pearson correlation matrix for all numeric columns.
    Run this AFTER encode_categorical.
    Requires: load_dataset must have been called.
    Stores: correlation_matrix.
    Returns: matrix shape, top 10 correlated pairs.
    """
    state = check_state(session_id, [df_name])
    if "error" in state:
        return state

    df = state[df_name]
    num_df = df.select_dtypes(include='number')

    if num_df.empty:
        return {"error": "No numeric columns found."}

    corr = num_df.corr().round(4)
    ACTIVE_DATAFRAMES[session_id]["correlation_matrix"] = corr

    pairs = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            pairs.append({
                "col_1": corr.columns[i],
                "col_2": corr.columns[j],
                "correlation": float(corr.iloc[i, j])
            })
    pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)

    return {
        "status": "Correlations computed",
        "shape": corr.shape,
        "top_pairs": pairs[:10]
    }


# [x] 17. drop_low_variance
def drop_low_variance(session_id: int, target_column: str, df_name: str = "main", threshold: float = 0.01) -> dict:
    """
    Drop numeric feature columns with variance below threshold.
    Run this AFTER encode_categorical, BEFORE separate_features_and_target.
    Requires: load_dataset must have been called.
    Stores: Overwrites dataframe with low-variance columns removed.
    Returns: columns dropped, remaining columns.
    """
    state = check_state(session_id, [df_name])
    if "error" in state:
        return state

    df = state[df_name]

    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found."}

    num_cols = df.select_dtypes(include='number').columns.tolist()
    if target_column in num_cols:
        num_cols.remove(target_column)

    dropped = []
    for col in num_cols:
        if df[col].var() < threshold:
            dropped.append(col)

    if dropped:
        df = df.drop(columns=dropped)
        ACTIVE_DATAFRAMES[session_id][df_name] = df

    return {
        "status": "Low variance check complete",
        "threshold": threshold,
        "columns_dropped": dropped,
        "remaining_columns": df.columns.tolist()
    }


# [x] 18. drop_correlated
def drop_correlated(session_id: int, target_column: str, df_name: str = "main", threshold: float = 0.95) -> dict:
    """
    Drop one column from each pair of highly correlated features.
    Run this AFTER compute_correlations, BEFORE separate_features_and_target.
    Requires: load_dataset must have been called.
    Stores: Overwrites dataframe with redundant columns removed.
    Returns: columns dropped, remaining columns.
    """
    state = check_state(session_id, [df_name])
    if "error" in state:
        return state

    df = state[df_name]

    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found."}

    num_cols = df.select_dtypes(include='number').columns.tolist()
    if target_column in num_cols:
        num_cols.remove(target_column)

    corr_matrix = df[num_cols].corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    if to_drop:
        df = df.drop(columns=to_drop)
        ACTIVE_DATAFRAMES[session_id][df_name] = df

    return {
        "status": "Correlated feature check complete",
        "threshold": threshold,
        "columns_dropped": to_drop,
        "remaining_columns": df.columns.tolist()
    }


# [x] 19. rank_features
def rank_features(session_id: int) -> dict:
    """
    Rank features by importance using trained model coefficients.
    Requires: train_single_model must have been called.
    Returns: feature rankings sorted by absolute importance.
    """
    state = check_state(session_id, ["trained_model", "X_train"])
    if "error" in state:
        return state

    model = state["trained_model"]
    feature_names = state["X_train"].columns.tolist()

    if not hasattr(model, 'coef_'):
        return {"error": "Model does not have coefficients for ranking."}

    if model.coef_.ndim > 1:
        importances = np.abs(model.coef_).mean(axis=0)
    else:
        importances = np.abs(model.coef_[0])

    rankings = sorted(
        zip(feature_names, importances.tolist()),
        key=lambda x: x[1],
        reverse=True
    )

    return {
        "status": "Features ranked",
        "rankings": [{"feature": name, "importance": round(imp, 4)} for name, imp in rankings]
    }


# [x] 20. plot_distribution
def plot_distribution(session_id: int, column: str, df_name: str = "main", output_path: str = None) -> dict:
    """
    Plot the distribution of a single column. Histogram for numeric, bar chart for categorical.
    Requires: load_dataset must have been called.
    Returns: path to saved plot.
    """
    state = check_state(session_id, [df_name])
    if "error" in state:
        return state

    df = state[df_name]

    if column not in df.columns:
        return {"error": f"Column '{column}' not found. Available: {df.columns.tolist()}"}

    if output_path is None:
        output_path = f"plot_distribution_{column}.png"

    plt.figure(figsize=(10, 6))

    if df[column].dtype in ['object', 'category']:
        counts = df[column].value_counts()
        sns.barplot(x=counts.index, y=counts.values)
        plt.xticks(rotation=45, ha='right')
    else:
        sns.histplot(df[column].dropna(), kde=True)

    plt.title(f"Distribution of {column}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return {
        "status": "Plot saved",
        "path": output_path
    }


# [x] 21. plot_correlations
def plot_correlations(session_id: int, output_path: str = None) -> dict:
    """
    Plot a heatmap of the correlation matrix.
    Requires: compute_correlations must have been called.
    Returns: path to saved plot.
    """
    state = check_state(session_id, ["correlation_matrix"])
    if "error" in state:
        return state

    corr = state["correlation_matrix"]

    if output_path is None:
        output_path = "plot_correlations.png"

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f',
                square=True, linewidths=0.5)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return {
        "status": "Plot saved",
        "path": output_path
    }


# [x] 22. plot_feature_importance
def plot_feature_importance(session_id: int, top_n: int = 20, output_path: str = None) -> dict:
    """
    Plot horizontal bar chart of feature importances from model coefficients.
    Requires: train_single_model must have been called.
    Returns: path to saved plot.
    """
    state = check_state(session_id, ["trained_model", "X_train"])
    if "error" in state:
        return state

    model = state["trained_model"]
    feature_names = state["X_train"].columns.tolist()

    if not hasattr(model, 'coef_'):
        return {"error": "Model does not support feature importance."}

    if model.coef_.ndim > 1:
        importances = np.abs(model.coef_).mean(axis=0)
    else:
        importances = np.abs(model.coef_[0])

    indices = np.argsort(importances)[::-1][:top_n]

    if output_path is None:
        output_path = "plot_feature_importance.png"

    plt.figure(figsize=(10, 8))
    plt.barh(
        range(len(indices)),
        importances[indices][::-1]
    )
    plt.yticks(
        range(len(indices)),
        [feature_names[i] for i in indices][::-1]
    )
    plt.xlabel("Importance (|coefficient|)")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return {
        "status": "Plot saved",
        "path": output_path
    }


# [x] 23. plot_confusion_matrix
def plot_confusion_matrix(session_id: int, output_path: str = None) -> dict:
    """
    Plot the confusion matrix heatmap from test predictions.
    Requires: train_single_model must have been called.
    Returns: path to saved plot, confusion matrix values.
    """
    state = check_state(session_id, ["y_test", "y_pred"])
    if "error" in state:
        return state

    y_test = state["y_test"]
    y_pred = state["y_pred"]

    if output_path is None:
        output_path = "plot_confusion_matrix.png"

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return {
        "status": "Plot saved",
        "path": output_path,
        "confusion_matrix": cm.tolist()
    }


# [x] 24. plot_roc
def plot_roc(session_id: int, output_path: str = None) -> dict:
    """
    Plot ROC curve with AUC score. Binary classification only.
    Requires: train_single_model must have been called. Target must be binary.
    Returns: path to saved plot, AUC score.
    """
    state = check_state(session_id, ["trained_model", "X_test", "y_test"])
    if "error" in state:
        return state

    model = state["trained_model"]
    X_test = state["X_test"]
    y_test = state["y_test"]

    if y_test.nunique() != 2:
        return {"error": f"ROC curve requires binary target. Found {y_test.nunique()} classes."}

    if not hasattr(model, 'predict_proba'):
        return {"error": "Model does not support probability predictions."}

    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = float(auc(fpr, tpr))

    if output_path is None:
        output_path = "plot_roc.png"

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return {
        "status": "Plot saved",
        "path": output_path,
        "auc": round(roc_auc, 4)
    }


# [x] 25. save_predictions
def save_predictions(session_id: int, output_path: str = "predictions.csv") -> dict:
    """
    Save test set predictions alongside actual values to a CSV file.
    Requires: train_single_model must have been called.
    Returns: path to saved file, row count.
    """
    state = check_state(session_id, ["X_test", "y_test", "y_pred"])
    if "error" in state:
        return state

    X_test = state["X_test"]
    y_test = state["y_test"]
    y_pred = state["y_pred"]

    results = X_test.copy()
    results["actual"] = y_test.values
    results["predicted"] = y_pred

    results.to_csv(output_path, index=False)

    return {
        "status": "Predictions saved",
        "path": output_path,
        "row_count": len(results)
    }


# [x] 26. save_model
def save_model(session_id: int, output_path: str = "model.joblib") -> dict:
    """
    Save the trained model to disk using joblib.
    Requires: train_single_model must have been called.
    Returns: path to saved model file, model type.
    """
    state = check_state(session_id, ["trained_model"])
    if "error" in state:
        return state

    model = state["trained_model"]
    joblib.dump(model, output_path)

    return {
        "status": "Model saved",
        "path": output_path,
        "model_type": type(model).__name__
    }


# [x] 27. load_model
def load_model(session_id: int, model_path: str = "model.joblib") -> dict:
    """
    Load a previously saved model from disk.
    Requires: A model file must exist at the given path.
    Stores: trained_model.
    Returns: model type, load status.
    """
    if session_id not in ACTIVE_DATAFRAMES:
        ACTIVE_DATAFRAMES[session_id] = {}

    if not os.path.exists(model_path):
        return {"error": f"Model file not found: {model_path}"}

    model = joblib.load(model_path)
    ACTIVE_DATAFRAMES[session_id]["trained_model"] = model

    return {
        "status": "Model loaded",
        "model_type": type(model).__name__,
        "path": model_path
    }


# [x] 28. generate_report
def generate_report(session_id: int, output_path: str = "report.txt") -> dict:
    """
    Generate a text summary of the full pipeline run including data shape, model type, and metrics.
    Requires: train_single_model must have been called.
    Returns: path to saved report, report content.
    """
    state = check_state(session_id, ["trained_model", "y_test", "y_pred"])
    if "error" in state:
        return state

    y_test = state["y_test"]
    y_pred = state["y_pred"]
    model = state["trained_model"]

    lines = ["=" * 60, "ML PIPELINE REPORT", "=" * 60, ""]

    if "main" in state:
        df = state["main"]
        lines.append(f"Dataset shape: {df.shape}")
        lines.append(f"Columns: {df.columns.tolist()}")
        lines.append("")

    if "X_train" in state:
        lines.append(f"X_train shape: {state['X_train'].shape}")
        lines.append(f"X_test shape:  {state['X_test'].shape}")
        lines.append("")

    lines.append(f"Model: {type(model).__name__}")
    lines.append(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    lines.append("")
    lines.append("Classification Report:")
    lines.append(classification_report(y_test, y_pred))

    report = "\n".join(lines)

    with open(output_path, "w") as f:
        f.write(report)

    return {
        "status": "Report generated",
        "path": output_path,
        "content": report
    }


# [x] 29. get_pipeline_state
def get_pipeline_state(session_id: int) -> dict:
    """
    Check what steps have been completed in the current session. Call anytime.
    Requires: Any session must exist.
    Returns: all stored object names, their types, and shapes where applicable.
    """
    state = check_state(session_id, [])
    if "error" in state:
        return state

    objects = {}
    for key, value in state.items():
        if isinstance(value, pd.DataFrame):
            objects[key] = {"type": "DataFrame", "shape": value.shape}
        elif isinstance(value, pd.Series):
            objects[key] = {"type": "Series", "shape": value.shape}
        elif isinstance(value, np.ndarray):
            objects[key] = {"type": "ndarray", "shape": value.shape}
        else:
            objects[key] = {"type": type(value).__name__}

    return {
        "status": "State retrieved",
        "session_id": session_id,
        "objects_stored": len(objects),
        "objects": objects
    }      
    
# [x] 31. concat_csvs
def concat_csvs(session_id: int, filepaths: list, df_name: str = "main") -> dict:
    """
    Load and concatenate multiple CSV files into a single dataframe.
    Run this instead of load_dataset when data is split across files.
    Requires: list of filepaths to CSV files.
    Stores: Combined DataFrame.
    Returns: row count per file, total shape, column names.
    """
    if session_id not in ACTIVE_DATAFRAMES:
        ACTIVE_DATAFRAMES[session_id] = {}

    dfs = []
    file_counts = {}

    for fp in filepaths:
        try:
            df = pd.read_csv(fp)
            file_counts[fp] = len(df)
            dfs.append(df)
        except FileNotFoundError:
            return {"error": f"File not found: {fp}"}
        except Exception as e:
            return {"error": f"Failed to load {fp}: {str(e)}"}

    if not dfs:
        return {"error": "No files loaded."}

    combined = pd.concat(dfs, ignore_index=True)
    ACTIVE_DATAFRAMES[session_id][df_name] = combined

    return {
        "status": "Files concatenated",
        "files_loaded": len(dfs),
        "rows_per_file": file_counts,
        "total_shape": combined.shape,
        "columns": combined.columns.tolist()
    }


# [x] 32. cast_types
def cast_types(session_id: int, type_map: dict, df_name: str = "main") -> dict:
    """
    Cast columns to specified dtypes. Use when pandas infers wrong types.
    Run this AFTER load_dataset, BEFORE any cleaning.
    Accepts: dict like {"age": "int", "price": "float", "zip_code": "str"}.
    Requires: load_dataset must have been called.
    Stores: Overwrites dataframe with retyped columns.
    Returns: previous and new dtypes for each cast column.
    """
    state = check_state(session_id, [df_name])
    if "error" in state:
        return state

    df = state[df_name]

    not_found = [col for col in type_map if col not in df.columns]
    if not_found:
        return {"error": f"Columns not found: {not_found}. Available: {df.columns.tolist()}"}

    changes = {}
    errors = {}

    for col, dtype in type_map.items():
        old_dtype = str(df[col].dtype)
        try:
            df[col] = df[col].astype(dtype)
            changes[col] = {"from": old_dtype, "to": str(df[col].dtype)}
        except (ValueError, TypeError) as e:
            errors[col] = {"from": old_dtype, "target": dtype, "error": str(e)}

    ACTIVE_DATAFRAMES[session_id][df_name] = df

    result = {
        "status": "Type casting complete",
        "changes": changes,
        "remaining_dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
    }
    if errors:
        result["errors"] = errors

    return result


# [x] 33. rename_columns
def rename_columns(session_id: int, rename_map: dict = None, clean_all: bool = False, df_name: str = "main") -> dict:
    """
    Rename columns by map or auto-clean all names (lowercase, strip, replace spaces with underscores).
    Run this AFTER load_dataset, BEFORE any processing.
    Accepts: rename_map like {"Old Name": "new_name"} OR clean_all=True.
    Requires: load_dataset must have been called.
    Stores: Overwrites dataframe with renamed columns.
    Returns: mapping of old names to new names.
    """
    state = check_state(session_id, [df_name])
    if "error" in state:
        return state

    df = state[df_name]
    old_names = df.columns.tolist()

    if rename_map:
        not_found = [col for col in rename_map if col not in df.columns]
        if not_found:
            return {"error": f"Columns not found: {not_found}. Available: {df.columns.tolist()}"}
        df = df.rename(columns=rename_map)

    elif clean_all:
        clean_map = {}
        for col in df.columns:
            new_name = col.strip().lower().replace(" ", "_").replace("-", "_")
            new_name = ''.join(c if c.isalnum() or c == '_' else '' for c in new_name)
            clean_map[col] = new_name
        df = df.rename(columns=clean_map)
        rename_map = clean_map

    else:
        return {"error": "Provide 'rename_map' (dict) or set clean_all=True."}

    ACTIVE_DATAFRAMES[session_id][df_name] = df

    return {
        "status": "Columns renamed",
        "renamed": rename_map,
        "columns": df.columns.tolist()
    }


# [x] 34. filter_rows
def filter_rows(session_id: int, column: str, condition: str, value=None, df_name: str = "main") -> dict:
    """
    Filter rows based on a condition applied to a column.
    Run this anytime BEFORE separate_features_and_target.
    Accepts conditions: 'eq', 'neq', 'gt', 'gte', 'lt', 'lte', 'contains', 'not_contains', 'isin', 'notin'.
    For 'isin'/'notin', pass value as a list.
    Requires: load_dataset must have been called.
    Stores: Overwrites dataframe with filtered rows.
    Returns: previous row count, current row count, rows removed.
    """
    state = check_state(session_id, [df_name])
    if "error" in state:
        return state

    df = state[df_name]

    if column not in df.columns:
        return {"error": f"Column '{column}' not found. Available: {df.columns.tolist()}"}

    if value is None and condition not in ['notnull', 'isnull']:
        return {"error": "Provide a value for the condition."}

    previous_row_count = len(df)

    conditions = {
        "eq": lambda: df[column] == value,
        "neq": lambda: df[column] != value,
        "gt": lambda: df[column] > value,
        "gte": lambda: df[column] >= value,
        "lt": lambda: df[column] < value,
        "lte": lambda: df[column] <= value,
        "contains": lambda: df[column].astype(str).str.contains(str(value), na=False),
        "not_contains": lambda: ~df[column].astype(str).str.contains(str(value), na=False),
        "isin": lambda: df[column].isin(value),
        "notin": lambda: ~df[column].isin(value),
        "isnull": lambda: df[column].isnull(),
        "notnull": lambda: df[column].notna()
    }

    if condition not in conditions:
        return {"error": f"Unknown condition '{condition}'. Available: {list(conditions.keys())}"}

    try:
        mask = conditions[condition]()
        df = df[mask]
    except Exception as e:
        return {"error": f"Filter failed: {str(e)}"}

    ACTIVE_DATAFRAMES[session_id][df_name] = df

    return {
        "status": "Rows filtered",
        "condition": f"{column} {condition} {value}",
        "previous_row_count": previous_row_count,
        "current_row_count": len(df),
        "rows_removed": previous_row_count - len(df)
    }


# [x] 35. clip_values
def clip_values(session_id: int, column: str, lower: float = None, upper: float = None, df_name: str = "main") -> dict:
    """
    Clip numeric column values to specified bounds. Values outside bounds are set to the bound.
    Run this AFTER handle_missing_features as an alternative to remove_outliers.
    Requires: load_dataset must have been called. Column must be numeric.
    Stores: Overwrites dataframe with clipped values.
    Returns: column clipped, bounds used, values affected.
    """
    state = check_state(session_id, [df_name])
    if "error" in state:
        return state

    df = state[df_name]

    if column not in df.columns:
        return {"error": f"Column '{column}' not found. Available: {df.columns.tolist()}"}

    if not pd.api.types.is_numeric_dtype(df[column]):
        return {"error": f"Column '{column}' is not numeric. dtype: {df[column].dtype}"}

    if lower is None and upper is None:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

    clipped_lower = int((df[column] < lower).sum()) if lower is not None else 0
    clipped_upper = int((df[column] > upper).sum()) if upper is not None else 0

    df[column] = df[column].clip(lower=lower, upper=upper)
    ACTIVE_DATAFRAMES[session_id][df_name] = df

    return {
        "status": "Values clipped",
        "column": column,
        "lower_bound": lower,
        "upper_bound": upper,
        "values_clipped_lower": clipped_lower,
        "values_clipped_upper": clipped_upper,
        "total_clipped": clipped_lower + clipped_upper
    }


# [x] 36. replace_values
def replace_values(session_id: int, column: str, replace_map: dict, df_name: str = "main") -> dict:
    """
    Replace specific values in a column.
    Run this anytime BEFORE encode_categorical.
    Accepts: replace_map like {"old_value": "new_value", "typo": "correct"}.
    Requires: load_dataset must have been called.
    Stores: Overwrites dataframe with replaced values.
    Returns: column name, replacements made, value counts after.
    """
    state = check_state(session_id, [df_name])
    if "error" in state:
        return state

    df = state[df_name]

    if column not in df.columns:
        return {"error": f"Column '{column}' not found. Available: {df.columns.tolist()}"}

    counts_before = {}
    for old_val in replace_map:
        counts_before[str(old_val)] = int((df[column] == old_val).sum())

    df[column] = df[column].replace(replace_map)
    ACTIVE_DATAFRAMES[session_id][df_name] = df

    return {
        "status": "Values replaced",
        "column": column,
        "replacements": replace_map,
        "occurrences_found": counts_before,
        "unique_values_after": int(df[column].nunique())
    }


# [x] 37. strip_whitespace
def strip_whitespace(session_id: int, df_name: str = "main") -> dict:
    """
    Strip leading/trailing whitespace from all string columns and column names.
    Run this AFTER load_dataset, BEFORE any processing.
    Requires: load_dataset must have been called.
    Stores: Overwrites dataframe with cleaned strings.
    Returns: columns cleaned, column names cleaned.
    """
    state = check_state(session_id, [df_name])
    if "error" in state:
        return state

    df = state[df_name]

    old_names = df.columns.tolist()
    df.columns = df.columns.str.strip()
    names_changed = [
        {"from": old, "to": new}
        for old, new in zip(old_names, df.columns)
        if old != new
    ]

    str_cols = df.select_dtypes(include='object').columns.tolist()
    for col in str_cols:
        df[col] = df[col].str.strip()

    ACTIVE_DATAFRAMES[session_id][df_name] = df

    return {
        "status": "Whitespace stripped",
        "column_names_cleaned": names_changed,
        "string_columns_stripped": str_cols
    }


# [x] 38. bin_continuous
def bin_continuous(session_id: int, column: str, n_bins: int = 5, strategy: str = "quantile", labels: list = None, df_name: str = "main") -> dict:
    """
    Bin a continuous column into discrete intervals.
    Run this AFTER handle_missing_features, BEFORE encode_categorical.
    Accepts strategy: 'quantile' (equal frequency) or 'uniform' (equal width).
    Requires: load_dataset must have been called. Column must be numeric.
    Stores: Overwrites column with binned values. Original column saved as '{column}_original'.
    Returns: bin edges, value counts per bin.
    """
    state = check_state(session_id, [df_name])
    if "error" in state:
        return state

    df = state[df_name]

    if column not in df.columns:
        return {"error": f"Column '{column}' not found. Available: {df.columns.tolist()}"}

    if not pd.api.types.is_numeric_dtype(df[column]):
        return {"error": f"Column '{column}' is not numeric. dtype: {df[column].dtype}"}

    df[f"{column}_original"] = df[column].copy()

    try:
        if strategy == "quantile":
            df[column], bin_edges = pd.qcut(df[column], q=n_bins, labels=labels, retbins=True, duplicates='drop')
        elif strategy == "uniform":
            df[column], bin_edges = pd.cut(df[column], bins=n_bins, labels=labels, retbins=True)
        else:
            return {"error": f"Unknown strategy '{strategy}'. Use 'quantile' or 'uniform'."}
    except Exception as e:
        return {"error": f"Binning failed: {str(e)}"}

    ACTIVE_DATAFRAMES[session_id][df_name] = df

    return {
        "status": "Column binned",
        "column": column,
        "strategy": strategy,
        "n_bins": n_bins,
        "bin_edges": [round(float(e), 4) for e in bin_edges],
        "value_counts": df[column].value_counts().sort_index().to_dict()
    }


# [x] 39. log_transform
def log_transform(session_id: int, columns: list = None, target_column: str = None, df_name: str = "main") -> dict:
    """
    Apply log1p transform to skewed numeric columns. Auto-detects skewed columns if none specified.
    Run this AFTER handle_missing_features, BEFORE scale_features.
    Requires: load_dataset must have been called. Columns must be numeric and non-negative.
    Stores: Overwrites dataframe with transformed values.
    Returns: columns transformed, skewness before and after.
    """
    state = check_state(session_id, [df_name])
    if "error" in state:
        return state

    df = state[df_name]

    num_cols = df.select_dtypes(include='number').columns.tolist()
    if target_column and target_column in num_cols:
        num_cols.remove(target_column)

    if columns is None:
        columns = []
        for col in num_cols:
            skew = abs(df[col].skew())
            if skew > 1.0 and (df[col] >= 0).all():
                columns.append(col)
        if not columns:
            return {"status": "No skewed columns found", "skipped": True}

    not_found = [col for col in columns if col not in df.columns]
    if not_found:
        return {"error": f"Columns not found: {not_found}. Available: {df.columns.tolist()}"}

    transforms = {}
    skipped = []

    for col in columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            skipped.append({"column": col, "reason": "not numeric"})
            continue
        if (df[col] < 0).any():
            skipped.append({"column": col, "reason": "contains negative values"})
            continue

        skew_before = round(float(df[col].skew()), 4)
        df[col] = np.log1p(df[col])
        skew_after = round(float(df[col].skew()), 4)

        transforms[col] = {
            "skew_before": skew_before,
            "skew_after": skew_after
        }

    ACTIVE_DATAFRAMES[session_id][df_name] = df

    result = {
        "status": "Log transform complete",
        "columns_transformed": list(transforms.keys()),
        "transform_details": transforms
    }
    if skipped:
        result["skipped"] = skipped

    return result


# [x] 40. extract_datetime_parts
def extract_datetime_parts(session_id: int, column: str, parts: list = None, drop_original: bool = True, df_name: str = "main") -> dict:
    """
    Parse a datetime column and extract year, month, day, dayofweek, hour as separate features.
    Run this AFTER load_dataset, BEFORE encode_categorical.
    Accepts parts: any subset of ['year', 'month', 'day', 'dayofweek', 'hour', 'minute', 'quarter', 'is_weekend'].
    Requires: load_dataset must have been called.
    Stores: Overwrites dataframe with new datetime feature columns.
    Returns: new columns created, rows that failed to parse.
    """
    state = check_state(session_id, [df_name])
    if "error" in state:
        return state

    df = state[df_name]

    if column not in df.columns:
        return {"error": f"Column '{column}' not found. Available: {df.columns.tolist()}"}

    try:
        dt_series = pd.to_datetime(df[column], infer_datetime_format=True, errors='coerce')
    except Exception as e:
        return {"error": f"Failed to parse datetime: {str(e)}"}

    failed_count = int(dt_series.isnull().sum() - df[column].isnull().sum())

    if parts is None:
        parts = ['year', 'month', 'day', 'dayofweek']

    extractors = {
        'year': lambda dt: dt.dt.year,
        'month': lambda dt: dt.dt.month,
        'day': lambda dt: dt.dt.day,
        'dayofweek': lambda dt: dt.dt.dayofweek,
        'hour': lambda dt: dt.dt.hour,
        'minute': lambda dt: dt.dt.minute,
        'quarter': lambda dt: dt.dt.quarter,
        'is_weekend': lambda dt: dt.dt.dayofweek.isin([5, 6]).astype(int)
    }

    invalid_parts = [p for p in parts if p not in extractors]
    if invalid_parts:
        return {"error": f"Invalid parts: {invalid_parts}. Available: {list(extractors.keys())}"}

    new_cols = []
    for part in parts:
        col_name = f"{column}_{part}"
        df[col_name] = extractors[part](dt_series)
        new_cols.append(col_name)

    if drop_original:
        df = df.drop(columns=[column])

    ACTIVE_DATAFRAMES[session_id][df_name] = df

    return {
        "status": "Datetime features extracted",
        "original_column": column,
        "new_columns": new_cols,
        "original_dropped": drop_original,
        "parse_failures": failed_count
    }


# [x] 41. create_interactions
def create_interactions(session_id: int, column_pairs: list = None, target_column: str = None, df_name: str = "main") -> dict:
    """
    Create interaction features by multiplying pairs of numeric columns.
    Run this AFTER encode_categorical, BEFORE separate_features_and_target.
    Accepts: list of tuples like [("col_a", "col_b"), ("col_c", "col_d")].
    If None, creates interactions for top 5 correlated pairs with target.
    Requires: load_dataset must have been called.
    Stores: Overwrites dataframe with new interaction columns.
    Returns: new columns created, shapes before and after.
    """
    state = check_state(session_id, [df_name])
    if "error" in state:
        return state

    df = state[df_name]
    prev_shape = df.shape

    num_cols = df.select_dtypes(include='number').columns.tolist()
    if target_column and target_column in num_cols:
        num_cols.remove(target_column)

    if column_pairs is None:
        if target_column and target_column in df.columns:
            corrs = df[num_cols].corrwith(df[target_column]).abs().sort_values(ascending=False)
            top_cols = corrs.head(5).index.tolist()
        else:
            top_cols = num_cols[:5]

        column_pairs = []
        for i in range(len(top_cols)):
            for j in range(i + 1, len(top_cols)):
                column_pairs.append((top_cols[i], top_cols[j]))

    new_cols = []
    skipped = []

    for pair in column_pairs:
        if len(pair) != 2:
            skipped.append({"pair": pair, "reason": "must be length 2"})
            continue
        col_a, col_b = pair
        if col_a not in df.columns or col_b not in df.columns:
            skipped.append({"pair": pair, "reason": "column not found"})
            continue
        if not pd.api.types.is_numeric_dtype(df[col_a]) or not pd.api.types.is_numeric_dtype(df[col_b]):
            skipped.append({"pair": pair, "reason": "not numeric"})
            continue

        col_name = f"{col_a}_x_{col_b}"
        df[col_name] = df[col_a] * df[col_b]
        new_cols.append(col_name)

    ACTIVE_DATAFRAMES[session_id][df_name] = df

    result = {
        "status": "Interactions created",
        "new_columns": new_cols,
        "previous_shape": prev_shape,
        "current_shape": df.shape
    }
    if skipped:
        result["skipped"] = skipped

    return result


# [x] 42. create_polynomials
def create_polynomials(session_id: int, columns: list = None, degree: int = 2, target_column: str = None, df_name: str = "main") -> dict:
    """
    Create polynomial features (squared, cubed, etc.) for specified numeric columns.
    Run this AFTER encode_categorical, BEFORE separate_features_and_target.
    Requires: load_dataset must have been called. Columns must be numeric.
    Stores: Overwrites dataframe with new polynomial columns.
    Returns: new columns created, shapes before and after.
    """
    state = check_state(session_id, [df_name])
    if "error" in state:
        return state

    df = state[df_name]
    prev_shape = df.shape

    num_cols = df.select_dtypes(include='number').columns.tolist()
    if target_column and target_column in num_cols:
        num_cols.remove(target_column)

    if columns is None:
        columns = num_cols[:10]

    not_found = [col for col in columns if col not in df.columns]
    if not_found:
        return {"error": f"Columns not found: {not_found}. Available: {df.columns.tolist()}"}

    new_cols = []
    for col in columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        for d in range(2, degree + 1):
            col_name = f"{col}_pow{d}"
            df[col_name] = df[col] ** d
            new_cols.append(col_name)

    ACTIVE_DATAFRAMES[session_id][df_name] = df

    return {
        "status": "Polynomial features created",
        "degree": degree,
        "new_columns": new_cols,
        "previous_shape": prev_shape,
        "current_shape": df.shape
    }


# [x] 43. select_features
def select_features(session_id: int, columns: list, df_name: str = "main") -> dict:
    """
    Keep only specified columns, drop everything else. The inverse of drop_columns.
    Run this BEFORE separate_features_and_target.
    Requires: load_dataset must have been called.
    Stores: Overwrites dataframe with only selected columns.
    Returns: columns kept, columns removed, shape.
    """
    state = check_state(session_id, [df_name])
    if "error" in state:
        return state

    df = state[df_name]

    not_found = [col for col in columns if col not in df.columns]
    if not_found:
        return {"error": f"Columns not found: {not_found}. Available: {df.columns.tolist()}"}

    removed = [col for col in df.columns if col not in columns]
    df = df[columns]
    ACTIVE_DATAFRAMES[session_id][df_name] = df

    return {
        "status": "Features selected",
        "columns_kept": columns,
        "columns_removed": removed,
        "shape": df.shape
    }


# [x] 44. create_folds
def create_folds(session_id: int, n_folds: int = 5) -> dict:
    """
    Create stratified k-fold indices for cross-validation.
    Run this AFTER separate_features_and_target.
    Requires: separate_features_and_target must have been called.
    Stores: fold_indices as list of (train_idx, val_idx) tuples.
    Returns: number of folds, rows per fold.
    """
    state = check_state(session_id, ["X", "y"])
    if "error" in state:
        return state

    X = state["X"]
    y = state["y"]

    if y.dtype == 'object' or y.nunique() < 20:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        folds = list(skf.split(X, y))
        stratified = True
    else:
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        folds = list(kf.split(X))
        stratified = False

    ACTIVE_DATAFRAMES[session_id]["fold_indices"] = folds

    fold_info = []
    for i, (train_idx, val_idx) in enumerate(folds):
        fold_info.append({
            "fold": i + 1,
            "train_rows": len(train_idx),
            "val_rows": len(val_idx)
        })

    return {
        "status": "Folds created",
        "n_folds": n_folds,
        "stratified": stratified,
        "fold_details": fold_info
    }


# [x] 45. cross_validate
def cross_validate_model(session_id: int, n_folds: int = 5) -> dict:
    """
    Run stratified k-fold cross-validation on the full X and y.
    Run this AFTER separate_features_and_target. Alternative to split_data + train_single_model.
    Requires: separate_features_and_target must have been called. All features must be numeric.
    Returns: per-fold accuracy, mean accuracy, std.
    """
    state = check_state(session_id, ["X", "y"])
    if "error" in state:
        return state

    X = state["X"]
    y = state["y"]

    non_numeric = X.select_dtypes(exclude='number').columns.tolist()
    if non_numeric:
        return {"error": f"Non-numeric columns found: {non_numeric}. Run encode_categorical first."}

    if y.dtype == 'object' or y.nunique() < 20:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    else:
        from sklearn.model_selection import KFold
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    model = LogisticRegression(max_iter=1000)

    try:
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    except Exception as e:
        return {"error": f"Cross-validation failed: {str(e)}"}

    return {
        "status": "Cross-validation complete",
        "model": "LogisticRegression",
        "n_folds": n_folds,
        "fold_scores": [round(float(s), 4) for s in scores],
        "mean_accuracy": round(float(scores.mean()), 4),
        "std_accuracy": round(float(scores.std()), 4)
    }


# [x] 46. tune_hyperparameters
def tune_hyperparameters(session_id: int, n_folds: int = 5) -> dict:
    """
    Grid search over LogisticRegression hyperparameters using cross-validation.
    Run this AFTER split_data, scale_features.
    Requires: split_data must have been called. All features must be numeric.
    Stores: Overwrites trained_model with best estimator, stores y_pred.
    Returns: best parameters, best score, all results.
    """
    state = check_state(session_id, ["X_train", "X_test", "y_train", "y_test"])
    if "error" in state:
        return state

    X_train = state["X_train"]
    X_test = state["X_test"]
    y_train = state["y_train"]
    y_test = state["y_test"]

    non_numeric = X_train.select_dtypes(exclude='number').columns.tolist()
    if non_numeric:
        return {"error": f"Non-numeric columns found: {non_numeric}. Run encode_categorical first."}

    param_grid = {
        'C': [0.01, 0.1, 1.0, 10.0],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }

    if y_train.dtype == 'object' or y_train.nunique() < 20:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    else:
        from sklearn.model_selection import KFold
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    grid = GridSearchCV(
        LogisticRegression(max_iter=1000),
        param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        return_train_score=True
    )

    try:
        grid.fit(X_train, y_train)
    except Exception as e:
        return {"error": f"Grid search failed: {str(e)}"}

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    ACTIVE_DATAFRAMES[session_id]["trained_model"] = best_model
    ACTIVE_DATAFRAMES[session_id]["y_pred"] = y_pred

    results = []
    for i in range(len(grid.cv_results_['params'])):
        results.append({
            "params": grid.cv_results_['params'][i],
            "mean_score": round(float(grid.cv_results_['mean_test_score'][i]), 4),
            "std_score": round(float(grid.cv_results_['std_test_score'][i]), 4)
        })
    results.sort(key=lambda x: x["mean_score"], reverse=True)

    return {
        "status": "Hyperparameter tuning complete",
        "best_params": grid.best_params_,
        "best_cv_score": round(float(grid.best_score_), 4),
        "test_accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "all_results": results
    }


# [x] 47. compare_models
def compare_models(session_id: int) -> dict:
    """
    Train and compare multiple classifiers on the same train/test split.
    Run this AFTER split_data, scale_features.
    Requires: split_data must have been called. All features must be numeric.
    Stores: trained_model and y_pred from the best performing model.
    Returns: accuracy per model, best model name.
    """
    state = check_state(session_id, ["X_train", "X_test", "y_train", "y_test"])
    if "error" in state:
        return state

    X_train = state["X_train"]
    X_test = state["X_test"]
    y_train = state["y_train"]
    y_test = state["y_test"]

    non_numeric = X_train.select_dtypes(exclude='number').columns.tolist()
    if non_numeric:
        return {"error": f"Non-numeric columns found: {non_numeric}. Run encode_categorical first."}

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "KNeighbors": KNeighborsClassifier()
    }

    results = []
    best_acc = -1
    best_name = None
    best_model = None
    best_pred = None

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            results.append({
                "model": name,
                "accuracy": round(float(acc), 4),
                "status": "success"
            })
            if acc > best_acc:
                best_acc = acc
                best_name = name
                best_model = model
                best_pred = y_pred
        except Exception as e:
            results.append({
                "model": name,
                "accuracy": None,
                "status": f"failed: {str(e)}"
            })

    results.sort(key=lambda x: x["accuracy"] if x["accuracy"] is not None else -1, reverse=True)

    if best_model is not None:
        ACTIVE_DATAFRAMES[session_id]["trained_model"] = best_model
        ACTIVE_DATAFRAMES[session_id]["y_pred"] = best_pred

    return {
        "status": "Model comparison complete",
        "results": results,
        "best_model": best_name,
        "best_accuracy": round(float(best_acc), 4)
    }


# [x] 48. compute_metrics
def compute_metrics(session_id: int) -> dict:
    """
    Compute detailed classification metrics from test predictions.
    Run this AFTER train_single_model or compare_models.
    Requires: train_single_model must have been called.
    Returns: accuracy, per-class precision/recall/f1, confusion matrix, support.
    """
    state = check_state(session_id, ["y_test", "y_pred"])
    if "error" in state:
        return state

    y_test = state["y_test"]
    y_pred = state["y_pred"]

    acc = float(accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    result = {
        "status": "Metrics computed",
        "accuracy": round(acc, 4),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "total_predictions": len(y_test)
    }

    if hasattr(state.get("trained_model"), 'predict_proba') and y_test.nunique() == 2:
        model = state["trained_model"]
        X_test = state["X_test"]
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        result["auc"] = round(float(auc(fpr, tpr)), 4)

    return result


# [x] 49. plot_learning_curve
def plot_learning_curve(session_id: int, n_points: int = 10, output_path: str = None) -> dict:
    """
    Plot training and validation accuracy as a function of training set size.
    Run this AFTER split_data, scale_features.
    Requires: split_data must have been called. All features must be numeric.
    Returns: path to saved plot, train/val scores at each size.
    """
    state = check_state(session_id, ["X_train", "X_test", "y_train", "y_test"])
    if "error" in state:
        return state

    X_train = state["X_train"]
    X_test = state["X_test"]
    y_train = state["y_train"]
    y_test = state["y_test"]

    non_numeric = X_train.select_dtypes(exclude='number').columns.tolist()
    if non_numeric:
        return {"error": f"Non-numeric columns found: {non_numeric}."}

    sizes = np.linspace(0.1, 1.0, n_points)
    train_scores = []
    val_scores = []
    actual_sizes = []

    for frac in sizes:
        n = max(int(len(X_train) * frac), 2)
        X_sub = X_train.iloc[:n]
        y_sub = y_train.iloc[:n]

        if y_sub.nunique() < 2:
            continue

        model = LogisticRegression(max_iter=1000)
        try:
            model.fit(X_sub, y_sub)
            train_acc = float(accuracy_score(y_sub, model.predict(X_sub)))
            val_acc = float(accuracy_score(y_test, model.predict(X_test)))
            train_scores.append(train_acc)
            val_scores.append(val_acc)
            actual_sizes.append(n)
        except Exception:
            continue

    if not train_scores:
        return {"error": "Could not generate learning curve."}

    if output_path is None:
        output_path = "plot_learning_curve.png"

    plt.figure(figsize=(10, 6))
    plt.plot(actual_sizes, train_scores, 'o-', label='Train')
    plt.plot(actual_sizes, val_scores, 'o-', label='Validation')
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return {
        "status": "Plot saved",
        "path": output_path,
        "train_scores": [round(s, 4) for s in train_scores],
        "val_scores": [round(s, 4) for s in val_scores],
        "sizes": actual_sizes
    }


# [x] 50. plot_predictions
def plot_predictions(session_id: int, n_samples: int = 50, output_path: str = None) -> dict:
    """
    Plot predicted vs actual values as a bar chart or scatter for visual comparison.
    Run this AFTER train_single_model.
    Requires: train_single_model must have been called.
    Returns: path to saved plot.
    """
    state = check_state(session_id, ["y_test", "y_pred"])
    if "error" in state:
        return state

    y_test = state["y_test"]
    y_pred = state["y_pred"]

    n = min(n_samples, len(y_test))
    y_actual = y_test.values[:n]
    y_predicted = y_pred[:n]
    indices = range(n)

    if output_path is None:
        output_path = "plot_predictions.png"

    fig, ax = plt.subplots(figsize=(14, 6))

    width = 0.35
    x = np.arange(n)
    ax.bar(x - width / 2, y_actual, width, label='Actual', alpha=0.8)
    ax.bar(x + width / 2, y_predicted, width, label='Predicted', alpha=0.8)

    mismatches = [i for i in range(n) if y_actual[i] != y_predicted[i]]
    for i in mismatches:
        ax.axvspan(i - 0.5, i + 0.5, alpha=0.15, color='red')

    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Class")
    ax.set_title(f"Predicted vs Actual (first {n} samples, {len(mismatches)} mismatches highlighted)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return {
        "status": "Plot saved",
        "path": output_path,
        "samples_shown": n,
        "mismatches": len(mismatches)
    }


# [x] 51. plot_calibration
def plot_calibration(session_id: int, n_bins: int = 10, output_path: str = None) -> dict:
    """
    Plot calibration curve showing predicted probability vs actual frequency. Binary classification only.
    Run this AFTER train_single_model.
    Requires: train_single_model must have been called. Target must be binary. Model must support predict_proba.
    Returns: path to saved plot, brier-like summary.
    """
    state = check_state(session_id, ["trained_model", "X_test", "y_test"])
    if "error" in state:
        return state

    model = state["trained_model"]
    X_test = state["X_test"]
    y_test = state["y_test"]

    if y_test.nunique() != 2:
        return {"error": f"Calibration plot requires binary target. Found {y_test.nunique()} classes."}

    if not hasattr(model, 'predict_proba'):
        return {"error": "Model does not support probability predictions."}

    y_prob = model.predict_proba(X_test)[:, 1]

    try:
        fraction_pos, mean_predicted = calibration_curve(y_test, y_prob, n_bins=n_bins)
    except Exception as e:
        return {"error": f"Calibration curve failed: {str(e)}"}

    if output_path is None:
        output_path = "plot_calibration.png"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(mean_predicted, fraction_pos, 'o-', label='Model')
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    ax1.set_xlabel("Mean Predicted Probability")
    ax1.set_ylabel("Fraction of Positives")
    ax1.set_title("Calibration Curve")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.hist(y_prob, bins=n_bins * 2, edgecolor='black', alpha=0.7)
    ax2.set_xlabel("Predicted Probability")
    ax2.set_ylabel("Count")
    ax2.set_title("Prediction Distribution")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    mean_gap = float(np.mean(np.abs(fraction_pos - mean_predicted)))

    return {
        "status": "Plot saved",
        "path": output_path,
        "n_bins": n_bins,
        "mean_calibration_gap": round(mean_gap, 4),
        "bin_predicted": [round(float(x), 4) for x in mean_predicted],
        "bin_actual": [round(float(x), 4) for x in fraction_pos]
    }