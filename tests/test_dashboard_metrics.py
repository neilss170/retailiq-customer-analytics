import pandas as pd

from src.dashboard_metrics import campaign_simulation, executive_metrics, metric_deltas


def test_executive_metrics_reconcile_net_revenue() -> None:
    frame = pd.DataFrame(
        {
            "transaction_status": ["valid_sale", "cancellation_or_return"],
            "invoice": ["1", "C1"],
            "customer_id": pd.Series([10, 10], dtype="Int64"),
            "gross_line_revenue": [100.0, 0.0],
            "returned_value": [0.0, 20.0],
            "net_line_revenue": [100.0, -20.0],
        }
    )
    metrics = executive_metrics(frame)
    assert metrics["net_revenue"] == 80.0
    assert metrics["orders"] == 1


def test_delta_is_unavailable_when_comparison_is_zero() -> None:
    delta = metric_deltas({"net_revenue": 10.0}, {"net_revenue": 0.0})
    assert delta["net_revenue"] is None


def test_campaign_simulation_reconciles_contacts_and_cost() -> None:
    scored = pd.DataFrame(
        {
            "estimated_return_likelihood": [0.9, 0.7, 0.4, 0.2],
            "returned_within_90_days": [1, 0, 1, 1],
        }
    )
    result = campaign_simulation(
        scored, threshold=0.5, contact_cost=1.0, missed_returner_cost=2.0
    )
    assert result["contacts"] == 2
    assert result["precision"] == 0.5
    assert result["recall"] == 1 / 3
    assert result["illustrative_cost"] == 6.0
