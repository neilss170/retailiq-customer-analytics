from pathlib import Path

import nbformat


def test_feature_engineering_notebook_does_not_load_test_partition() -> None:
    path = Path("notebooks/04_return_risk_feature_engineering.ipynb")
    if not path.exists():
        return
    notebook = nbformat.read(path, as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")

    assert "read_parquet(ROOT / 'data/processed/return_risk_test.parquet')" not in source
    assert "pd.read_parquet(ROOT / \"data/processed/return_risk_test.parquet\")" not in source


def test_modeling_notebook_does_not_load_test_partition() -> None:
    path = Path("notebooks/05_return_risk_modeling.ipynb")
    if not path.exists():
        return
    notebook = nbformat.read(path, as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")

    assert "return_risk_test.parquet" not in source


def test_final_test_notebook_does_not_tune_or_fit_models() -> None:
    path = Path("notebooks/06_final_classification_test.ipynb")
    if not path.exists():
        return
    notebook = nbformat.read(path, as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")

    assert "GridSearchCV" not in source
    assert "RandomizedSearchCV" not in source
    assert ".fit(" not in source
    assert "choose_threshold" not in source
