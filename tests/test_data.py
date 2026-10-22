"""
Unit tests for Dataset loading, integrity verification, distinct domain values, and metrics.
"""

import os
import pandas as pd
from src.data_loader import load_main_dataset, load_metrics_and_comparison, compute_dataset_kpis
from src.model_loader import VALID_CATEGORIES

def test_dataset_loading():
    df = load_main_dataset()
    assert len(df) == 50000
    assert "Fraud_Label" in df.columns
    assert "Transaction_Amount" in df.columns
    assert "Merchant_Category" in df.columns

def test_dataset_kpis():
    df = load_main_dataset()
    kpis = compute_dataset_kpis(df)
    assert kpis["total_transactions"] == 50000
    assert kpis["fraud_transactions"] == 16067
    assert kpis["non_fraud_transactions"] == 33933
    assert round(kpis["fraud_percentage"], 2) == 32.13
    assert round(kpis["non_fraud_percentage"], 2) == 67.87
    assert abs(kpis["total_value"] - 4970550.61) < 1.0
    assert round(kpis["avg_value"], 2) == 99.41
    assert "2023-01-01" in kpis["date_min"]
    assert "2023-12-31" in kpis["date_max"]

def test_distinct_categorical_values():
    df = load_main_dataset()
    for col, expected_cats in VALID_CATEGORIES.items():
        actual_cats = sorted(df[col].dropna().unique().tolist())
        assert actual_cats == expected_cats, f"Mismatch in {col}: {actual_cats} != {expected_cats}"

def test_metrics_loading():
    metrics, comp_df = load_metrics_and_comparison()
    assert "selected_model" in metrics
    assert metrics["selected_model"] == "Decision Tree"
    assert "test_accuracy" in metrics
    assert "test_recall" in metrics
    assert "test_precision" in metrics
    assert "test_f1" in metrics
    assert "test_roc_auc" in metrics
    assert "test_average_precision" in metrics
    assert len(comp_df) >= 4
    assert "Model" in comp_df.columns
    assert "F1 Score" in comp_df.columns
