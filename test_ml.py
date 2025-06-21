import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference

data = pd.read_csv("data/census.csv")

# TODO: implement the first test. Change the function name and input as needed
def test_model_type():
    """
    Test that the train_model returns a RandomForestClassifier.
    """
    cat_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]
    X, y, _, _ = process_data(data, categorical_features=cat_features, label="salary", training=True)
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)


# TODO: implement the second test. Change the function name and input as needed
def test_compute_model_metrics_output():
    """
    Test compute_model_metrics returns expected types and values for simple input.
    """
    y_true = np.array([1, 0, 1, 1])
    y_pred = np.array([1, 0, 0, 1]) 

    precision, recall, f1 = compute_model_metrics(y_true, y_pred)
    assert round(precision, 2) == 1.00
    assert round(recall, 2) == 0.67
    assert round(f1, 2) == 0.80


# TODO: implement the third test. Change the function name and input as needed
def test_inference_shape():
    """
    Test that inference returns predictions of the correct shape.
    """
    cat_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]
    X, y, _, _ = process_data(data, categorical_features=cat_features, label="salary", training=True)
    model = train_model(X, y)
    preds = inference(model, X)
    assert len(preds) == len(X)
