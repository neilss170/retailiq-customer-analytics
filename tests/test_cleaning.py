import pandas as pd

from src.cleaning import clean_transactions


def sample_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Invoice": ["100", "100", "C101", "102", "103", "104"],
            "StockCode": ["A", "A", "A", "POST", "B", "C"],
            "Description": ["Item", "Item", "Item", "Postage", "Gift", "Broken"],
            "Quantity": [2, 2, -1, 1, 1, 1],
            "InvoiceDate": pd.to_datetime(["2011-01-01"] * 6),
            "Price": [5.0, 5.0, 5.0, 2.0, 0.0, -1.0],
            "Customer ID": [1, 1, 1, 2, None, 3],
            "Country": ["United Kingdom"] * 6,
            "source_sheet": ["Year 2010-2011"] * 6,
        }
    )


def test_cleaning_classifies_rows_and_removes_exact_duplicates() -> None:
    cleaned, duplicates = clean_transactions(sample_transactions())

    assert len(duplicates) == 1
    assert len(cleaned) == 5
    assert set(cleaned["transaction_status"]) == {
        "valid_sale",
        "cancellation_or_return",
        "adjustment",
        "zero_price",
        "invalid",
    }


def test_revenue_components_are_not_double_counted() -> None:
    cleaned, _ = clean_transactions(sample_transactions())

    assert cleaned["gross_line_revenue"].sum() == 10.0
    assert cleaned["returned_value"].sum() == 5.0
    assert cleaned["net_line_revenue"].sum() == 5.0


def test_anonymous_sales_are_kept_but_not_model_eligible() -> None:
    cleaned, _ = clean_transactions(sample_transactions())
    anonymous = cleaned.loc[cleaned["customer_id"].isna()].iloc[0]

    assert anonymous["transaction_status"] == "zero_price"
    assert not anonymous["eligible_customer_model"]
