"""
Automated validation of all 7 Streamlit dashboard pages and scoring workflows.
Uses streamlit.testing.v1.AppTest to execute the application headless.
"""

import pytest
from streamlit.testing.v1 import AppTest

def test_page_1_overview():
    at = AppTest.from_file("app.py")
    at.run(timeout=30)
    assert not at.exception
    assert "Overview" in at.radio[0].value
    assert len(at.dataframe) >= 1

def test_page_2_transaction_scoring_single_and_grid():
    at = AppTest.from_file("app.py")
    at.run(timeout=30)
    
    # Navigate to Transaction Scoring
    at.radio[0].set_value(at.radio[0].options[1])
    at.run(timeout=30)
    assert not at.exception
    assert "Transaction Scoring" in at.radio[0].value
    
    # Test Tab 1: Grid Scoring
    score_buttons = [b for b in at.button if "Score Transactions in Grid" in b.label]
    if score_buttons:
        score_buttons[0].click()
        at.run(timeout=30)
        assert not at.exception
        assert len(at.dataframe) >= 1

    # Test Tab 2: Single Transaction Form
    low_risk_btns = [b for b in at.button if "Standard Low-Risk POS" in b.label]
    if low_risk_btns:
        low_risk_btns[0].click()
        at.run(timeout=30)
        assert not at.exception
        
    # Submit single prediction form
    submit_btns = [b for b in at.button if "Run Decision Tree Scoring" in b.label]
    if submit_btns:
        submit_btns[0].click()
        at.run(timeout=30)
        assert not at.exception

def test_page_3_batch_scoring():
    at = AppTest.from_file("app.py")
    at.run(timeout=30)
    
    # Navigate to Batch Transaction Scoring
    at.radio[0].set_value(at.radio[0].options[2])
    at.run(timeout=30)
    assert not at.exception
    assert "Batch Transaction Scoring" in at.radio[0].value
    
    # Click Load 250 Sample Records
    sample_btns = [b for b in at.button if "Load 250 Sample Records" in b.label]
    assert len(sample_btns) > 0
    sample_btns[0].click()
    at.run(timeout=30)
    assert not at.exception
    assert len(at.dataframe) >= 1

def test_page_4_model_performance():
    at = AppTest.from_file("app.py")
    at.run(timeout=30)
    
    # Navigate to Model Performance & Audit
    at.radio[0].set_value(at.radio[0].options[3])
    at.run(timeout=30)
    assert not at.exception
    assert "Model Performance & Audit" in at.radio[0].value
    assert len(at.dataframe) >= 1

def test_page_5_data_explorer():
    at = AppTest.from_file("app.py")
    at.run(timeout=30)
    
    # Navigate to Transaction Data Explorer
    at.radio[0].set_value(at.radio[0].options[4])
    at.run(timeout=30)
    assert not at.exception
    assert "Transaction Data Explorer" in at.radio[0].value
    assert len(at.dataframe) >= 1

def test_page_6_project_artifacts():
    at = AppTest.from_file("app.py")
    at.run(timeout=30)
    
    # Navigate to Project Files & Artifacts
    at.radio[0].set_value(at.radio[0].options[5])
    at.run(timeout=30)
    assert not at.exception
    assert "Project Files & Artifacts" in at.radio[0].value
    assert len(at.markdown) >= 5

def test_page_7_about_and_governance():
    at = AppTest.from_file("app.py")
    at.run(timeout=30)
    
    # Navigate to About & Governance
    at.radio[0].set_value(at.radio[0].options[6])
    at.run(timeout=30)
    assert not at.exception
    assert "About & Governance" in at.radio[0].value
