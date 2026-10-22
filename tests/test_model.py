"""
Unit tests for Fraud Detection Model loading, preprocessing, schema validation, and inference.
"""

import pytest
import pandas as pd
from src.model_loader import (
    load_pipeline,
    preprocess_features,
    validate_batch_schema,
    REQUIRED_FEATURES,
    VALID_CATEGORIES
)
from src.predictor import FraudPredictor

def test_load_pipeline():
    pipeline = load_pipeline()
    assert pipeline is not None
    assert hasattr(pipeline, "predict")
    assert hasattr(pipeline, "predict_proba")

def test_preprocess_features_with_date():
    sample_raw = pd.DataFrame([{
        "Transaction_ID": "TXN_TEST_1",
        "Transaction_Amount": 250.0,
        "Transaction_Type": "POS",
        "Date": "2023-06-15 14:30:00",
        "Account_Balance": 5000.0,
        "Device_Type": "Mobile",
        "Location": "New York",
        "Merchant_Category": "Electronics",
        "Previous_Fraudulent_Activity": 0,
        "Daily_Transaction_Count": 2,
        "Card_Type": "Visa",
        "Card_Age": 24
    }])
    processed = preprocess_features(sample_raw)
    assert list(processed.columns) == REQUIRED_FEATURES
    assert processed["Transaction_Month"].iloc[0] == 6
    assert processed["Transaction_DayOfWeek"].iloc[0] == 3  # Thursday

def test_preprocess_features_missing_temporal():
    sample_no_date = pd.DataFrame([{
        "Transaction_Amount": 250.0,
        "Transaction_Type": "POS",
        "Account_Balance": 5000.0,
        "Device_Type": "Mobile",
        "Location": "New York",
        "Merchant_Category": "Electronics",
        "Previous_Fraudulent_Activity": 0,
        "Daily_Transaction_Count": 2,
        "Card_Type": "Visa",
        "Card_Age": 24
    }])
    with pytest.raises(ValueError, match="Missing temporal features"):
        preprocess_features(sample_no_date)

def test_validate_batch_schema():
    # 1. Valid DataFrame
    valid_df = pd.DataFrame([{
        "Transaction_ID": "TXN_1",
        "Date": "2023-05-10",
        "Transaction_Amount": 100.0,
        "Transaction_Type": "Online",
        "Account_Balance": 2000.0,
        "Device_Type": "Laptop",
        "Location": "London",
        "Merchant_Category": "Clothing",
        "Previous_Fraudulent_Activity": 0,
        "Daily_Transaction_Count": 1,
        "Card_Type": "Mastercard",
        "Card_Age": 12
    }])
    val_res = validate_batch_schema(valid_df)
    assert val_res["is_valid"] is True
    assert len(val_res["missing_features"]) == 0

    # 2. Missing columns
    invalid_df = pd.DataFrame([{"Transaction_Amount": 100.0}])
    val_res_inv = validate_batch_schema(invalid_df)
    assert val_res_inv["is_valid"] is False
    assert len(val_res_inv["missing_features"]) > 0

    # 3. Empty DataFrame
    empty_res = validate_batch_schema(pd.DataFrame())
    assert empty_res["is_valid"] is False
    assert "empty" in empty_res["error_message"].lower()

def test_single_prediction():
    predictor = FraudPredictor()
    sample_record = {
        "Transaction_Amount": 890.0,
        "Transaction_Type": "Online",
        "Account_Balance": 450.0,
        "Device_Type": "Mobile",
        "Location": "Tokyo",
        "Merchant_Category": "Travel",
        "Previous_Fraudulent_Activity": 1,
        "Daily_Transaction_Count": 11,
        "Card_Type": "Mastercard",
        "Card_Age": 3,
        "Date": "2023-11-20 22:00:00"
    }
    result = predictor.predict_single(sample_record, threshold=0.50)
    assert "fraud_probability" in result
    assert 0.0 <= result["fraud_probability"] <= 1.0
    assert result["risk_tier"] in ["High Risk", "Medium Risk", "Low Risk"]
    assert len(result["risk_factors"]) > 0
    assert result["threshold_used"] == 0.50

def test_batch_prediction():
    predictor = FraudPredictor()
    df_sample = pd.DataFrame([
        {
            "Transaction_ID": f"TXN_{i}",
            "User_ID": f"USER_{i}",
            "Transaction_Amount": 50.0 * (i + 1),
            "Transaction_Type": "POS",
            "Account_Balance": 1000.0,
            "Device_Type": "Mobile",
            "Location": "London",
            "Merchant_Category": "Groceries",
            "Previous_Fraudulent_Activity": 0,
            "Daily_Transaction_Count": 1,
            "Card_Type": "Visa",
            "Card_Age": 12,
            "Date": "2023-04-10"
        }
        for i in range(5)
    ])
    scored_df = predictor.predict_batch(df_sample, threshold=0.50)
    assert "Fraud_Probability" in scored_df.columns
    assert "Model_Prediction" in scored_df.columns
    assert "Risk_Tier" in scored_df.columns
    assert "Transaction_ID" in scored_df.columns  # Preserved original columns
    assert "User_ID" in scored_df.columns
    assert len(scored_df) == 5

def test_batch_prediction_empty():
    predictor = FraudPredictor()
    with pytest.raises(ValueError, match="Cannot score an empty DataFrame"):
        predictor.predict_batch(pd.DataFrame())
