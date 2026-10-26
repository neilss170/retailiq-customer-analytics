from datetime import date
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")
PAGES = [
    "Executive Summary",
    "Revenue Drivers",
    "Customer Intelligence",
    "Retention & Risk",
    "Revenue Outlook",
    "Data Explorer",
    "Methods & Trust",
]


@pytest.mark.parametrize("page", PAGES)
def test_every_dashboard_page_renders(page: str) -> None:
    app = AppTest.from_file(APP, default_timeout=120)
    app.run()
    app.radio[0].set_value(page).run()
    assert not app.exception, [error.value for error in app.exception]


def test_narrow_filters_do_not_break_pages() -> None:
    app = AppTest.from_file(APP, default_timeout=120)
    app.run()
    app.multiselect[0].set_value(["Iceland"]).run()
    for page in PAGES:
        app.radio[0].set_value(page).run()
        assert not app.exception, (page, [error.value for error in app.exception])


def test_date_slider_changes_the_figures() -> None:
    app = AppTest.from_file(APP, default_timeout=120)
    app.run()
    before = [block.value for block in app.markdown if "kpi-value" in block.value]
    app.slider(key="date_range").set_value((date(2011, 1, 1), date(2011, 3, 31))).run()
    after = [block.value for block in app.markdown if "kpi-value" in block.value]
    assert not app.exception
    assert after != before


@pytest.mark.parametrize("dates", [(date(2010, 3, 1), date(2010, 3, 1)), (date(2011, 12, 9), date(2011, 12, 9))])
def test_single_day_ranges_render(dates: tuple) -> None:
    app = AppTest.from_file(APP, default_timeout=120)
    app.run()
    app.slider(key="date_range").set_value(dates).run()
    for page in PAGES:
        app.radio[0].set_value(page).run()
        assert not app.exception, (page, [error.value for error in app.exception])
