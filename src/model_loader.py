"""
Model Loader and Schema Validation Module.
Provides safe loading of the trained fraud detection pipeline, feature preprocessing,
and strict schema validation against the empirical training data.
"""

import os
import warnings
import joblib
import pandas as pd
from typing import Tuple, Any, List, Dict, Optional

warnings.filterwarnings("ignore", category=UserWarning)
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass

# Compatibility patch for cross-version unpickling (e.g., sklearn 1.6.1 -> 1.9.x)
import sklearn.compose._column_transformer
if not hasattr(sklearn.compose._column_transformer, "_RemainderColsList"):
    class _RemainderColsList(list):
        pass
    sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList

from sklearn.impute import SimpleImputer
_orig_simple_imputer_transform = SimpleImputer.transform

def _patched_simple_imputer_transform(self, X):
    if not hasattr(self, "_fill_dtype") and hasattr(self, "_fit_dtype"):
        self._fill_dtype = self._fit_dtype
    return _orig_simple_imputer_transform(self, X)

SimpleImputer.transform = _patched_simple_imputer_transform

# Exactly the 12 features expected by the fitted ColumnTransformer in fraud_detection_model.joblib
REQUIRED_FEATURES: List[str] = [
    "Transaction_Amount",
    "Transaction_Type",
    "Account_Balance",
    "Device_Type",
    "Location",
    "Merchant_Category",
    "Previous_Fraudulent_Activity",
    "Daily_Transaction_Count",
    "Card_Type",
    "Card_Age",
    "Transaction_Month",
    "Transaction_DayOfWeek"
]

# Exact distinct domain categories from synthetic_fraud_dataset1.csv and fitted OneHotEncoder
VALID_CATEGORIES: Dict[str, List[str]] = {
    "Transaction_Type": ["ATM Withdrawal", "Bank Transfer", "Online", "POS"],
    "Device_Type": ["Laptop", "Mobile", "Tablet"],
    "Location": ["London", "Mumbai", "New York", "Sydney", "Tokyo"],
    "Merchant_Category": ["Clothing", "Electronics", "Groceries", "Restaurants", "Travel"],
    "Card_Type": ["Amex", "Discover", "Mastercard", "Visa"]
}

# Known non-feature identifier and ground truth columns that should not be fed to the model
EXCLUDED_METADATA_COLS: List[str] = [
    "Transaction_ID",
    "User_ID",
    "Customer_ID",
    "Fraud_Label",
    "is_fraud",
    "Date"
]

def load_pipeline(model_path: str = "fraud_detection_model.joblib") -> Any:
    """Loads the joblib pipeline with cross-version compatibility."""
    if not os.path.exists(model_path):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        alt_path = os.path.join(base_dir, model_path)
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            raise FileNotFoundError(f"Model file not found at {model_path} or {alt_path}")
    
    return joblib.load(model_path)

def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures input DataFrame contains all 12 required model features,
    deriving temporal features from Date when provided.
    Raises ValueError if required features are missing or temporal features cannot be derived.
    """
    if df is None or len(df) == 0:
        raise ValueError("Input DataFrame is empty.")

    df = df.copy()

    # Extract temporal features from Date if present
    if "Date" in df.columns:
        date_series = pd.to_datetime(df["Date"], errors="coerce")
        if date_series.isnull().all():
            raise ValueError("All values in 'Date' column failed to parse into valid timestamps.")
        
        if "Transaction_Month" not in df.columns or df["Transaction_Month"].isnull().any():
            df["Transaction_Month"] = date_series.dt.month.fillna(1).astype(int)
        if "Transaction_DayOfWeek" not in df.columns or df["Transaction_DayOfWeek"].isnull().any():
            df["Transaction_DayOfWeek"] = date_series.dt.dayofweek.fillna(0).astype(int)
    else:
        # If Date is absent, Transaction_Month and Transaction_DayOfWeek must be provided explicitly
        if "Transaction_Month" not in df.columns or "Transaction_DayOfWeek" not in df.columns:
            raise ValueError(
                "Missing temporal features: Please provide either a 'Date' column (e.g., '2023-06-15') "
                "or both 'Transaction_Month' (1-12) and 'Transaction_DayOfWeek' (0-6)."
            )

    # Check for missing required features
    missing = [c for c in REQUIRED_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Input data is missing required model features: {missing}")

    return df[REQUIRED_FEATURES]

def validate_batch_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validates an uploaded DataFrame against the model's required input schema before scoring.
    Reports missing, unexpected, and out-of-domain columns without mutating data.
    """
    if df is None or len(df) == 0:
        return {
            "is_valid": False,
            "error_message": "Uploaded file is empty (0 rows).",
            "missing_features": REQUIRED_FEATURES,
            "unexpected_features": [],
            "ground_truth_col": None,
            "out_of_domain_categories": {},
            "has_nulls": False,
            "null_counts": {},
            "row_count": 0
        }

    cols = list(df.columns)
    
    # Check temporal capability
    has_date = "Date" in cols
    has_month = "Transaction_Month" in cols
    has_dow = "Transaction_DayOfWeek" in cols
    has_temporal = has_date or (has_month and has_dow)

    # Required base features (excluding temporal which can be derived from Date)
    base_required = [c for c in REQUIRED_FEATURES if c not in ["Transaction_Month", "Transaction_DayOfWeek"]]
    missing = [c for c in base_required if c not in cols]
    
    if not has_temporal:
        if not has_month:
            missing.append("Transaction_Month (or 'Date')")
        if not has_dow:
            missing.append("Transaction_DayOfWeek (or 'Date')")

    # Detect ground truth labels
    ground_truth_col = None
    for gt in ["Fraud_Label", "is_fraud", "fraud"]:
        if gt in cols:
            ground_truth_col = gt
            break

    # Identify unexpected/extra columns
    recognized = set(REQUIRED_FEATURES).union(EXCLUDED_METADATA_COLS)
    unexpected = [c for c in cols if c not in recognized]

    # Check for missing values
    null_counts = {c: int(df[c].isnull().sum()) for c in cols if df[c].isnull().sum() > 0}
    has_nulls = len(null_counts) > 0

    # Check categorical domain values
    out_of_domain = {}
    for col, valid_vals in VALID_CATEGORIES.items():
        if col in cols:
            unique_vals = df[col].dropna().unique().tolist()
            invalid_vals = [v for v in unique_vals if v not in valid_vals]
            if invalid_vals:
                out_of_domain[col] = invalid_vals

    is_valid = (len(missing) == 0)
    error_msg = None
    if not is_valid:
        error_msg = f"Missing required columns: {', '.join(missing)}."

    return {
        "is_valid": is_valid,
        "error_message": error_msg,
        "missing_features": missing,
        "unexpected_features": unexpected,
        "ground_truth_col": ground_truth_col,
        "out_of_domain_categories": out_of_domain,
        "has_nulls": has_nulls,
        "null_counts": null_counts,
        "row_count": len(df)
    }
