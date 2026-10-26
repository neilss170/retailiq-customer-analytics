"""Rebuild every RetailIQ artifact from the raw UCI workbook in one command.

The notebooks read tables written by the command-line steps in ``src``, so the
steps must run in this order. Executed notebooks are saved in place.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = ROOT / "data/raw/source/online_retail_II.xlsx"
SOURCE_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
WORKBOOK_SHA256 = "bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980"

PIPELINE_STEPS = [
    "src.cleaning",
    "src.analytics",
    "src.return_risk_features",
    "src.return_risk_modeling",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_workbook() -> None:
    if not WORKBOOK.exists():
        print(f"Downloading {SOURCE_URL}", flush=True)
        with urllib.request.urlopen(SOURCE_URL, timeout=300) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
        WORKBOOK.parent.mkdir(parents=True, exist_ok=True)
        WORKBOOK.write_bytes(archive.read(WORKBOOK.name))
    if sha256(WORKBOOK) != WORKBOOK_SHA256:
        raise RuntimeError(
            f"{WORKBOOK} does not match the workbook the published results used."
        )


def run_step(module: str) -> None:
    started = time.time()
    subprocess.run([sys.executable, "-m", module], cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL)
    print(f"{module}: {time.time() - started:.0f}s", flush=True)


def run_notebook(path: Path) -> None:
    started = time.time()
    notebook = nbformat.read(path, as_version=4)
    NotebookClient(
        notebook,
        timeout=1800,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
    ).execute()
    nbformat.write(notebook, path)
    print(f"{path.name}: {time.time() - started:.0f}s", flush=True)


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()

    ensure_workbook()
    for module in PIPELINE_STEPS:
        run_step(module)
    # Notebooks 05-07 select the model, score the sealed test and forecast.
    for path in sorted((ROOT / "notebooks").glob("0*.ipynb")):
        run_notebook(path)


if __name__ == "__main__":
    main()
