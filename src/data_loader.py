"""
Data Loading and Analytics Helper Module.
Provides caching and structured access to datasets, model metrics, and comparisons.
"""

import os
import json
import pandas as pd
from typing import Dict, Any, Tuple

DATASET_PATH = "synthetic_fraud_dataset1.csv"
METRICS_PATH = "fraud_model_metrics.json"
COMPARISON_PATH = "validation_model_comparison.csv"

def load_main_dataset(filepath: str = DATASET_PATH) -> pd.DataFrame:
    """Loads and formats the primary 50,000-row synthetic transactions dataset."""
    if not os.path.exists(filepath):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(base_dir, filepath)
    
    df = pd.read_csv(filepath)
    df["Date"] = pd.to_datetime(df["Date"])
    df["YearMonth"] = df["Date"].dt.to_period("M").astype(str)
    df["MonthName"] = df["Date"].dt.strftime("%b %Y")
    df["DayOfWeekName"] = df["Date"].dt.day_name()
    return df

def load_metrics_and_comparison() -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Loads the saved test metrics and validation model comparison table."""
    metrics_path = METRICS_PATH
    comparison_path = COMPARISON_PATH
    
    if not os.path.exists(metrics_path):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        metrics_path = os.path.join(base_dir, metrics_path)
    
    if not os.path.exists(comparison_path):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        comparison_path = os.path.join(base_dir, comparison_path)
    
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    
    comparison_df = pd.read_csv(comparison_path)
    return metrics, comparison_df

def compute_dataset_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculates empirical KPIs from the transaction dataset."""
    total_tx = len(df)
    fraud_tx = int((df["Fraud_Label"] == 1).sum()) if "Fraud_Label" in df.columns else 0
    non_fraud_tx = total_tx - fraud_tx
    fraud_pct = (fraud_tx / total_tx * 100.0) if total_tx > 0 else 0.0
    non_fraud_pct = (non_fraud_tx / total_tx * 100.0) if total_tx > 0 else 0.0
    total_val = float(df["Transaction_Amount"].sum()) if "Transaction_Amount" in df.columns else 0.0
    avg_val = float(df["Transaction_Amount"].mean()) if (total_tx > 0 and "Transaction_Amount" in df.columns) else 0.0
    
    date_min = str(df["Date"].min()) if "Date" in df.columns and len(df) > 0 else "N/A"
    date_max = str(df["Date"].max()) if "Date" in df.columns and len(df) > 0 else "N/A"

    return {
        "total_transactions": total_tx,
        "fraud_transactions": fraud_tx,
        "non_fraud_transactions": non_fraud_tx,
        "fraud_percentage": fraud_pct,
        "non_fraud_percentage": non_fraud_pct,
        "total_value": total_val,
        "avg_value": avg_val,
        "date_min": date_min,
        "date_max": date_max
    }
