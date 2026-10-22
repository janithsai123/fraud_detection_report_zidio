"""
Prediction and Risk Assessment Module.
Provides risk scoring, category assignment, and explanation metrics.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np
from src.model_loader import (
    load_pipeline,
    preprocess_features,
    validate_batch_schema,
    REQUIRED_FEATURES,
    VALID_CATEGORIES
)

class FraudPredictor:
    def __init__(self, model_path: str = "fraud_detection_model.joblib"):
        self.pipeline = load_pipeline(model_path)
        if not hasattr(self.pipeline, "predict_proba"):
            raise AttributeError("The loaded model pipeline does not support probability predictions (predict_proba).")
    
    def predict_single(self, record: Dict[str, Any], threshold: float = 0.50) -> Dict[str, Any]:
        """
        Scores a single transaction dictionary.
        Returns probability, binary prediction, risk level, and primary risk factors.
        """
        df = pd.DataFrame([record])
        processed_df = preprocess_features(df)
        
        # Get class probability (class 1 = Fraud)
        probs = self.pipeline.predict_proba(processed_df)[0]
        fraud_prob = float(probs[1])
        is_fraud = int(fraud_prob >= threshold)
        
        # Determine risk tier (Consistent visual indicator colors)
        if fraud_prob >= 0.65:
            risk_tier = "High Risk"
            risk_color = "#DC2626"  # Red
            action = "Flag for Immediate Analyst Review"
        elif fraud_prob >= 0.40:
            risk_tier = "Medium Risk"
            risk_color = "#F59E0B"  # Muted Amber
            action = "Secondary Verification Recommended"
        else:
            risk_tier = "Low Risk"
            risk_color = "#10B981"  # Emerald
            action = "Approve Standard Processing"
        
        # Identify domain heuristic risk factors
        risk_factors = []
        if record.get("Previous_Fraudulent_Activity", 0) == 1:
            risk_factors.append("Prior fraudulent activity recorded on customer profile.")
        if record.get("Daily_Transaction_Count", 0) >= 8:
            risk_factors.append(f"Elevated daily transaction velocity ({record.get('Daily_Transaction_Count')} transactions today).")
        if record.get("Transaction_Amount", 0) >= 500:
            risk_factors.append(f"Above-average transaction ticket value (₹{record.get('Transaction_Amount', 0):,.2f}).")
        if record.get("Account_Balance", 0) < record.get("Transaction_Amount", 0):
            risk_factors.append("Transaction amount exceeds recorded account balance.")
        if record.get("Card_Age", 100) < 6:
            risk_factors.append(f"New account issue tenure ({record.get('Card_Age')} months active).")
        
        if not risk_factors:
            risk_factors.append("No adverse velocity or balance triggers identified.")
        
        return {
            "fraud_probability": round(fraud_prob, 4),
            "prediction": is_fraud,
            "risk_tier": risk_tier,
            "risk_color": risk_color,
            "recommended_action": action,
            "risk_factors": risk_factors,
            "threshold_used": threshold
        }
    
    def predict_batch(self, df: pd.DataFrame, threshold: float = 0.50) -> pd.DataFrame:
        """
        Scores an entire DataFrame of transactions.
        Preserves all original input rows and columns in the returned output.
        Appends Fraud_Probability, Model_Prediction, Risk_Tier, and Recommended_Action.
        """
        if df is None or len(df) == 0:
            raise ValueError("Cannot score an empty DataFrame.")

        # Validate schema before scoring
        validation = validate_batch_schema(df)
        if not validation["is_valid"]:
            raise ValueError(f"Batch validation failed: {validation['error_message']}")

        # Preprocess features (isolates the 12 model features from metadata/identifiers)
        processed_df = preprocess_features(df)
        probs = self.pipeline.predict_proba(processed_df)[:, 1]
        
        result_df = df.copy()
        result_df["Fraud_Probability"] = np.round(probs, 4)
        result_df["Model_Prediction"] = (probs >= threshold).astype(int)
        
        conditions = [
            (probs >= 0.65),
            (probs >= 0.40) & (probs < 0.65),
            (probs < 0.40)
        ]
        choices = ["High Risk", "Medium Risk", "Low Risk"]
        result_df["Risk_Tier"] = np.select(conditions, choices, default="Low Risk")
        
        action_choices = [
            "Flag for Immediate Review",
            "Secondary Verification",
            "Approve Standard"
        ]
        result_df["Recommended_Action"] = np.select(conditions, action_choices, default="Approve Standard")
        
        return result_df
