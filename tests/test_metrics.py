import pytest
from echoprime_transfer.metrics import regression_metrics

def test_regression_metrics_perfect_prediction():
    m = regression_metrics([40,50,60],[40,50,60])
    assert m["mae"] == pytest.approx(0.0)
    assert m["rmse"] == pytest.approx(0.0)
    assert m["r2"] == pytest.approx(1.0)
