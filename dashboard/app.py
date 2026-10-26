"""RetailIQ storytelling dashboard for customer and revenue decisions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dashboard_metrics import (  # noqa: E402
    campaign_simulation,
    cohort_retention,
    country_performance,
    executive_metrics,
    executive_narrative,
    filter_transactions,
    flag_partial_months,
    metric_deltas,
    monthly_revenue,
    product_performance,
)

st.set_page_config(
    page_title="RetailIQ | Customer & Revenue Intelligence",
    page_icon="RI",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = {
    "background": "#0B1220",
    "surface": "#151F30",
    "surface_2": "#1B273A",
    "text": "#F3F6FA",
    "muted": "#9EABC0",
    "blue": "#3B82F6",
    "blue_light": "#93C5FD",
    "gold": "#F59E0B",
    "positive": "#22C55E",
    "negative": "#EF4444",
}

st.markdown(
    f"""
    <style>
    .stApp {{ background: {COLORS['background']}; color: {COLORS['text']}; }}
    [data-testid="stSidebar"] {{ background: #101827; border-right: 1px solid #22304A; }}
    [data-testid="stHeader"] {{ background: rgba(11,18,32,.88); }}
    h1, h2, h3 {{ letter-spacing: -0.025em; }}
    .block-container {{ max-width: 1500px; padding-top: 4.2rem; padding-bottom: 4rem; }}
    .brand {{ font-size: 1.45rem; font-weight: 800; color: {COLORS['text']}; }}
    .brand-sub {{ color: {COLORS['muted']}; font-size: .82rem; margin-bottom: 1.4rem; }}
    .eyebrow {{ color: {COLORS['blue_light']}; font-size: .76rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: .13em; margin-bottom: .25rem; }}
    .page-subtitle {{ color: {COLORS['muted']}; font-size: 1rem; margin-top: -.7rem; margin-bottom: 1.2rem; }}
    .kpi-card {{ background: linear-gradient(145deg, {COLORS['surface']}, #111A29);
        border: 1px solid #253451; border-radius: 14px; padding: 1rem 1.05rem;
        min-height: 122px; box-shadow: 0 10px 24px rgba(0,0,0,.16); }}
    .kpi-card:hover {{ border-color: #3B82F6; transform: translateY(-1px);
        transition: border-color .18s ease, transform .18s ease; }}
    .kpi-label {{ color: {COLORS['muted']}; font-size: .78rem; font-weight: 650;
        text-transform: uppercase; letter-spacing: .06em; }}
    .kpi-value {{ color: {COLORS['text']}; font-size: 1.75rem; font-weight: 750; margin: .28rem 0; }}
    .kpi-delta {{ font-size: .78rem; color: {COLORS['muted']}; }}
    .insight {{ background: {COLORS['surface_2']}; border: 1px solid #2A3A57;
        border-left: 4px solid {COLORS['blue']}; border-radius: 12px; padding: 1rem 1.15rem; margin: .5rem 0 1.2rem; }}
    .insight-title {{ color: {COLORS['blue_light']}; text-transform: uppercase;
        letter-spacing: .08em; font-size: .72rem; font-weight: 800; }}
    .insight-body {{ color: {COLORS['text']}; font-size: 1rem; margin-top: .3rem; }}
    .action {{ color: {COLORS['muted']}; font-size: .86rem; margin-top: .45rem; }}
    .decision-card {{ background: #101A2A; border: 1px solid #263A59;
        border-radius: 12px; padding: 1rem; min-height: 150px; }}
    .decision-card h4 {{ color: {COLORS['blue_light']}; margin: 0 0 .45rem; }}
    .decision-card p {{ color: {COLORS['muted']}; margin: 0; line-height: 1.45; }}
    .section-note {{ color: {COLORS['muted']}; font-size: .85rem; }}
    div[data-testid="stDataFrame"] {{ border: 1px solid #263653; border-radius: 10px; }}
    .stTabs [data-baseweb="tab-list"] {{ gap: .4rem; }}
    .stTabs [data-baseweb="tab"] {{ background: {COLORS['surface']}; border-radius: 8px; padding: .4rem .9rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_data() -> dict[str, object]:
    transactions = pd.read_parquet(ROOT / "data/processed/transactions.parquet")
    segments = pd.read_parquet(ROOT / "data/processed/customer_segments.parquet")
    forecast = pd.read_parquet(ROOT / "data/processed/revenue_forecast.parquet")
    test_predictions = pd.read_parquet(
        ROOT / "reports/generated/classification_final_test_predictions.parquet"
    )
    forecast_test = pd.read_parquet(
        ROOT / "reports/generated/forecast_final_test_predictions.parquet"
    )
    classification_meta = json.loads(
        (ROOT / "models/return_risk_metadata.json").read_text(encoding="utf-8")
    )
    classification_test = json.loads(
        (
            ROOT / "reports/generated/classification_final_test_metrics.json"
        ).read_text(encoding="utf-8")
    )
    forecast_meta = json.loads(
        (ROOT / "models/revenue_forecast_metadata.json").read_text(encoding="utf-8")
    )
    return {
        "transactions": transactions,
        "segments": segments,
        "forecast": forecast,
        "test_predictions": test_predictions,
        "forecast_test": forecast_test,
        "classification_meta": classification_meta,
        "classification_test": classification_test,
        "forecast_meta": forecast_meta,
    }


def money(value: float) -> str:
    sign = "-" if value < 0 else ""
    value = abs(value)
    if value >= 1_000_000:
        return f"{sign}£{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{sign}£{value / 1_000:.1f}K"
    return f"{sign}£{value:,.0f}"


def percentage(value: float) -> str:
    return f"{value:.1%}"


def delta_label(delta: float | None) -> str:
    if delta is None:
        return "No comparable prior period"
    direction = "up" if delta >= 0 else "down"
    return f"{direction} {abs(delta):.1%} vs prior period"


def kpi(label: str, value: str, delta: float | None) -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{value}</div>
          <div class="kpi-delta">{delta_label(delta)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def insight(takeaway: str, action: str) -> None:
    st.markdown(
        f"""
        <div class="insight">
          <div class="insight-title">Key takeaway</div>
          <div class="insight-body">{takeaway}</div>
          <div class="action"><b>What to do:</b> {action}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_chart(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["surface"],
        plot_bgcolor=COLORS["surface"],
        font={"color": COLORS["text"], "family": "Inter, Segoe UI, sans-serif"},
        height=height,
        margin=dict(l=35, r=25, t=70, b=40),
        title_font={"size": 17},
        legend_title_text="",
        hoverlabel={"bgcolor": "#111827"},
    )
    fig.update_xaxes(gridcolor="#263653", zerolinecolor="#52617A")
    fig.update_yaxes(gridcolor="#263653", zerolinecolor="#52617A")
    return fig


def revenue_bars(frame: pd.DataFrame, label: str, total: float, title: str, color: str) -> go.Figure:
    """Horizontal net-revenue bars labelled with value and share of the total."""
    data = frame.sort_values("net_revenue")
    share = data["net_revenue"] / total if total else np.nan
    fig = go.Figure(
        go.Bar(
            x=data["net_revenue"],
            y=data[label],
            orientation="h",
            marker_color=color,
            text=[f"{money(v)} · {s:.0%}" if s >= .01 else f"{money(v)} · {s:.1%}" for v, s in zip(data["net_revenue"], share)],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: £%{x:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(title=title)
    fig.update_xaxes(tickprefix="£", tickformat="~s", range=[0, data["net_revenue"].max() * 1.3])
    return fig


def page_header(title: str, subtitle: str, question: str) -> None:
    st.markdown(f'<div class="eyebrow">{question}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def decision_card(title: str, body: str) -> None:
    st.markdown(
        f'<div class="decision-card"><h4>{title}</h4><p>{body}</p></div>',
        unsafe_allow_html=True,
    )


def priority_action(recency: float, value: float, return_likelihood: float | None) -> str:
    """Translate historical signals into a cautious, readable next action."""
    if return_likelihood is not None and return_likelihood >= 0.65:
        return "Prioritize for a measured retention test; the model indicates relatively high natural return likelihood."
    if recency <= 45 and value > 1_000:
        return "Protect the relationship with service-led communication rather than a broad discount."
    if recency >= 181:
        return "Treat as a low-confidence reactivation case; the model is weakest for long-inactive customers."
    return "Place in a standard nurture group and compare outcomes against an untreated control."


data = load_data()
transactions: pd.DataFrame = data["transactions"]  # type: ignore[assignment]

with st.sidebar:
    st.markdown('<div class="brand">RetailIQ</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="brand-sub">Customer intelligence and revenue outlook</div>',
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Story",
        [
            "Executive Summary",
            "Revenue Drivers",
            "Customer Intelligence",
            "Retention & Risk",
            "Revenue Outlook",
            "Data Explorer",
            "Methods & Trust",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    min_date = transactions["invoice_date"].min().date()
    max_date = transactions["invoice_date"].max().date()
    date_range = st.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    countries_available = sorted(transactions["country"].dropna().unique().tolist())
    countries = st.multiselect(
        "Countries",
        countries_available,
        placeholder="All countries",
    )
    st.caption(f"Source coverage: {min_date:%d %b %Y}–{max_date:%d %b %Y}")
    st.caption("Historical snapshot — not a live retail feed")

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = map(pd.Timestamp, date_range)
else:
    start_date = end_date = pd.Timestamp(date_range)

filtered = filter_transactions(transactions, start_date, end_date, countries)
period_days = max(1, (end_date - start_date).days + 1)
previous_end = start_date - pd.Timedelta(days=1)
previous_start = previous_end - pd.Timedelta(days=period_days - 1)
previous = filter_transactions(transactions, previous_start, previous_end, countries)
current_metrics = executive_metrics(filtered)
previous_metrics = executive_metrics(previous)
deltas = metric_deltas(current_metrics, previous_metrics)
takeaway, recommended_action = executive_narrative(
    current_metrics, previous_metrics, deltas
)

if filtered.empty:
    st.warning("No transactions match the selected filters. Expand the date range or country selection.")
    st.stop()

if page == "Executive Summary":
    page_header(
        "Executive Summary",
        "A concise view of commercial performance, customer activity, and the next revenue signal.",
        "How is the business performing?",
    )
    insight(takeaway, recommended_action)
    cards = [
        ("Net revenue", money(current_metrics["net_revenue"]), deltas["net_revenue"]),
        ("Gross revenue", money(current_metrics["gross_revenue"]), deltas["gross_revenue"]),
        ("Orders", f"{current_metrics['orders']:,.0f}", deltas["orders"]),
        ("Active customers", f"{current_metrics['active_customers']:,.0f}", deltas["active_customers"]),
        ("Average order value", money(current_metrics["average_order_value"]), deltas["average_order_value"]),
        ("Repeat-purchase rate", percentage(current_metrics["repeat_purchase_rate"]), deltas["repeat_purchase_rate"]),
        ("Returned-value rate", percentage(current_metrics["return_value_rate"]), deltas["return_value_rate"]),
    ]
    columns = st.columns(4)
    for column, card in zip(columns, cards[:4]):
        with column:
            kpi(*card)
    columns = st.columns(3)
    for column, card in zip(columns, cards[4:]):
        with column:
            kpi(*card)

    trend = flag_partial_months(
        monthly_revenue(filtered),
        max(start_date, transactions["invoice_date"].min()),
        min(end_date + pd.Timedelta(days=1) - pd.Timedelta(seconds=1), transactions["invoice_date"].max()),
    )
    complete = trend.loc[~trend["partial"]]
    partial = trend.loc[trend["partial"]]
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[.72, .28], vertical_spacing=.06
    )
    fig.add_trace(go.Scatter(x=complete["month"], y=complete["net_revenue"], name="Net revenue", mode="lines+markers", line={"color": COLORS["blue"], "width": 3}, hovertemplate="%{x|%b %Y}: £%{y:,.0f}<extra></extra>"), row=1, col=1)
    if len(partial):
        fig.add_trace(go.Scatter(x=partial["month"], y=partial["net_revenue"], name="Partial month (not comparable)", mode="markers", marker={"color": COLORS["background"], "size": 9, "line": {"color": COLORS["blue_light"], "width": 2}}, hovertemplate="%{x|%b %Y} (partial): £%{y:,.0f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Bar(x=trend["month"], y=trend["returned_value"], name="Returned value", marker_color=np.where(trend["partial"], COLORS["muted"], COLORS["gold"]), hovertemplate="%{x|%b %Y}: £%{y:,.0f} returned<extra></extra>"), row=2, col=1)
    fig.update_layout(title="Monthly net revenue and returned value", legend={"orientation": "h", "y": -0.12})
    fig.update_yaxes(tickprefix="£", tickformat="~s", rangemode="tozero")
    fig.update_yaxes(title_text="Net revenue", row=1, col=1)
    fig.update_yaxes(title_text="Returned", row=2, col=1)
    st.plotly_chart(style_chart(fig, 480), width="stretch")
    if len(partial):
        st.caption("Hollow points are months the selected range covers only in part (the source ends on 9 Dec 2011), so their totals are not comparable with full months.")

    left, right = st.columns([1.15, .85])
    with left:
        country = country_performance(filtered)
        fig = revenue_bars(country.head(8), "country", country["net_revenue"].sum(), "Which countries contribute most to net revenue?", COLORS["blue"])
        st.plotly_chart(style_chart(fig), width="stretch")
    with right:
        st.subheader("Decision snapshot")
        forecast_meta = data["forecast_meta"]
        test_meta = data["classification_test"]
        st.markdown(
            f"""
            - **Revenue outlook method:** {str(forecast_meta['selected_model']).replace('_', ' ').title()}
            - **Forecast holdout sMAPE:** {forecast_meta['final_test_metrics']['smape']:.1f}%
            - **Customer return model:** Gradient Boosting
            - **Final return-model PR-AUC:** {test_meta['metrics']['pr_auc']:.3f}
            - **Important risk:** recall is weaker for long-inactive customers
            """
        )
        st.caption("Model results use their fixed historical evaluation periods and do not change with dashboard filters.")
    st.subheader("Management agenda")
    agenda = st.columns(3)
    with agenda[0]:
        decision_card(
            "Protect revenue",
            "Start with markets and products contributing the most net revenue, then verify whether growth is broad or concentrated.",
        )
    with agenda[1]:
        decision_card(
            "Control leakage",
            "Review products with material returned value before increasing promotion or inventory exposure.",
        )
    with agenda[2]:
        decision_card(
            "Plan cautiously",
            "Use the forecast range for capacity planning and retain a buffer because the holdout error widened versus backtesting.",
        )

elif page == "Revenue Drivers":
    page_header(
        "Revenue Drivers",
        "Separate geographic contribution, product performance, and cancellation exposure.",
        "What contributed to the result?",
    )
    insight(takeaway, "Review high-contribution markets and products alongside cancellation exposure before changing assortment or promotions.")
    country = country_performance(filtered)
    product = product_performance(filtered)
    portfolio = product.loc[
        (product["gross_revenue"] > 0) & (product["invoices"] >= 5)
    ].copy()
    portfolio["revenue_band"] = np.where(
        portfolio["net_revenue"] >= portfolio["net_revenue"].median(),
        "Higher revenue",
        "Lower revenue",
    )
    left, right = st.columns(2)
    with left:
        domestic = country["country"].eq("United Kingdom")
        exclude_uk = domestic.any() and len(country) > 1 and st.toggle(
            "Compare markets outside the UK",
            value=True,
            help="The UK holds most revenue, which flattens every other bar.",
        )
        shown = country.loc[~domestic] if exclude_uk else country
        title = "Net revenue by country outside the UK — top 12" if exclude_uk else "Net revenue by country — top 12"
        fig = revenue_bars(shown.head(12), "country", country["net_revenue"].sum(), title, COLORS["blue"])
        st.plotly_chart(style_chart(fig), width="stretch")
    with right:
        fig = revenue_bars(product.head(12), "product_label", product["net_revenue"].sum(), "Net revenue by product — top 12", COLORS["gold"])
        st.plotly_chart(style_chart(fig), width="stretch")
    st.subheader("Product portfolio")
    st.caption(
        "Each point is a product sold on at least five invoices with positive net revenue; size shows invoice count. The upper-right area combines material revenue with elevated returned-value exposure."
    )
    portfolio = portfolio.loc[portfolio["net_revenue"].gt(0)]
    fig = px.scatter(
        portfolio,
        x="net_revenue",
        y="return_value_rate",
        size="invoices",
        color="revenue_band",
        hover_name="product_label",
        hover_data={"stock_code": True, "gross_revenue": ":,.0f", "invoices": True},
        color_discrete_map={"Higher revenue": COLORS["gold"], "Lower revenue": COLORS["blue_light"]},
        title="Product revenue and returned-value exposure",
        labels={"net_revenue": "Net revenue (GBP, log scale)", "return_value_rate": "Returned-value rate", "revenue_band": ""},
        log_x=True,
        opacity=.65,
        size_max=28,
    )
    fig.update_traces(marker={"sizemin": 3, "line": {"width": 0}})
    fig.update_xaxes(tickprefix="£", tickformat="~s", dtick=1)
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(style_chart(fig, 500), width="stretch")
    search_term = st.text_input(
        "Find a product",
        placeholder="Type a stock code or product description",
    ).strip()
    if search_term:
        matches = product.loc[
            product["stock_code"].astype(str).str.contains(search_term, case=False, na=False)
            | product["description"].astype(str).str.contains(search_term, case=False, na=False)
        ].head(25)
        if matches.empty:
            st.info("No product matched that search in the selected period.")
        else:
            st.dataframe(
                matches[
                    [
                        "stock_code",
                        "product_label",
                        "net_revenue",
                        "gross_revenue",
                        "returned_value",
                        "return_value_rate",
                        "invoices",
                    ]
                ].style.format(
                    {
                        "net_revenue": "£{:,.0f}",
                        "gross_revenue": "£{:,.0f}",
                        "returned_value": "£{:,.0f}",
                        "return_value_rate": "{:.1%}",
                    }
                ),
                width="stretch",
                hide_index=True,
            )
    st.subheader("Cancellation watchlist")
    watch = (
        product.loc[product["gross_revenue"].ge(1_000)]
        .sort_values("returned_value", ascending=False)
        .head(20)
    )
    st.dataframe(
        watch[["stock_code", "product_label", "gross_revenue", "returned_value", "return_value_rate"]].style.format(
            {"gross_revenue": "£{:,.0f}", "returned_value": "£{:,.0f}", "return_value_rate": "{:.1%}"}
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption("High cancellation value is a review signal, not evidence that the product caused returns.")

elif page == "Customer Intelligence":
    page_header(
        "Customer Intelligence",
        "Understand value concentration and the behavioural segments behind historical revenue.",
        "Who generates value?",
    )
    segments: pd.DataFrame = data["segments"]  # type: ignore[assignment]
    filtered_customers = set(filtered["customer_id"].dropna().astype(int))
    segment_view = segments.loc[segments["customer_id"].isin(filtered_customers)].copy()
    segment_summary = (
        segment_view.groupby("segment")
        .agg(customers=("customer_id", "size"), median_recency=("recency_days", "median"), median_orders=("frequency", "median"), total_value=("monetary", "sum"))
        .reset_index()
        .sort_values("total_value", ascending=False)
    )
    if len(segment_summary):
        leader = segment_summary.iloc[0]
        insight(
            f"{leader['segment']} represents the largest historical-value segment in the current customer selection.",
            "Protect high-value active relationships while using lower-cost reactivation tests for long-inactive customers.",
        )
    left, right = st.columns(2)
    with left:
        fig = revenue_bars(segment_summary.rename(columns={"total_value": "net_revenue"}), "segment", segment_summary["total_value"].sum(), "Historical customer value by segment", COLORS["blue"])
        st.plotly_chart(style_chart(fig), width="stretch")
    with right:
        fig = px.scatter(
            segment_view.loc[segment_view["monetary"].gt(0)],
            x="recency_days",
            y="monetary",
            color="segment",
            size="frequency",
            hover_data=["customer_id"],
            title="Customer recency and historical value",
            log_y=True,
            opacity=.6,
            size_max=22,
            color_discrete_sequence=[COLORS["blue_light"], COLORS["gold"], "#A78BFA"],
            labels={"recency_days": "Days since last purchase", "monetary": "Historical value (GBP, log scale)", "frequency": "Orders", "segment": ""},
        )
        fig.update_traces(marker={"sizemin": 2.5, "line": {"width": 0}})
        fig.update_yaxes(tickprefix="£", tickformat="~s", dtick=1)
        fig.update_layout(legend={"orientation": "h", "y": -0.25})
        st.plotly_chart(style_chart(fig), width="stretch")
    st.dataframe(
        segment_summary.style.format(
            {"median_recency": "{:,.0f} days", "median_orders": "{:,.0f}", "total_value": "£{:,.0f}"}
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption("Segments are descriptive summaries from historical transactions; they are not causal customer personas.")
    st.subheader("Customer 360 drill-down")
    st.caption("Search a pseudonymous customer ID to connect segment, purchase history, value, and model evidence.")
    customer_options = sorted(segment_view["customer_id"].dropna().astype(int).unique().tolist())
    selected_customer = st.selectbox(
        "Customer ID", customer_options, index=None, placeholder="Choose a customer"
    )
    if selected_customer is not None:
        customer_segment = segment_view.loc[
            segment_view["customer_id"].eq(selected_customer)
        ].iloc[0]
        customer_sales = filtered.loc[
            filtered["customer_id"].eq(selected_customer)
            & filtered["transaction_status"].eq("valid_sale")
        ].copy()
        scored_customer = data["test_predictions"].loc[
            data["test_predictions"]["customer_id"].eq(selected_customer)
        ].sort_values("observation_date")
        likelihood = (
            float(scored_customer.iloc[-1]["estimated_return_likelihood"])
            if len(scored_customer)
            else None
        )
        customer_columns = st.columns(5)
        customer_cards = [
            ("Segment", str(customer_segment["segment"]), None),
            ("Recency", f"{customer_segment['recency_days']:,.0f} days", None),
            ("Orders", f"{customer_segment['frequency']:,.0f}", None),
            ("Historical value", money(float(customer_segment["monetary"])), None),
            ("Return likelihood", percentage(likelihood) if likelihood is not None else "Not scored", None),
        ]
        for column, card in zip(customer_columns, customer_cards):
            with column:
                kpi(*card)
        st.info(
            priority_action(
                float(customer_segment["recency_days"]),
                float(customer_segment["monetary"]),
                likelihood,
            )
        )
        if customer_sales.empty:
            st.caption("No valid sales for this customer under the current dashboard filters.")
        else:
            customer_sales["month"] = (
                customer_sales["invoice_date"].dt.to_period("M").dt.to_timestamp()
            )
            history = (
                customer_sales.groupby("month")["gross_line_revenue"]
                .sum()
                .rename("revenue")
                .reset_index()
            )
            fig = px.bar(
                history,
                x="month",
                y="revenue",
                title="Monthly purchase history for selected customer",
                labels={"month": "Month", "revenue": "Gross revenue (GBP)"},
            )
            fig.update_traces(marker_color=COLORS["blue"])
            st.plotly_chart(style_chart(fig, 340), width="stretch")

elif page == "Retention & Risk":
    page_header(
        "Retention & Customer Risk",
        "See whether cohorts return and which historical customer snapshots the frozen model would prioritize.",
        "Are customers coming back, and who may need attention?",
    )
    retention = cohort_retention(filtered)
    if retention.empty:
        st.info("Known-customer activity is insufficient for a cohort view under these filters.")
    else:
        fig = px.imshow(
            retention,
            aspect="auto",
            color_continuous_scale=["#111827", "#1D4ED8", "#93C5FD"],
            title="Monthly cohort retention",
            labels={"x": "Months since first purchase", "y": "Acquisition cohort", "color": "Retention"},
        )
        fig.update_traces(
            text=retention.map(lambda value: "" if pd.isna(value) else f"{value:.0%}").to_numpy(),
            texttemplate="%{text}",
            hovertemplate="Cohort %{y}, month %{x}: %{z:.1%}<extra></extra>",
        )
        fig.update_coloraxes(colorbar_tickformat=".0%")
        st.plotly_chart(style_chart(fig, 560), width="stretch")
        st.caption("Blank cells are months that had not happened yet when the data ends. Example: month 1 shows the share of an acquisition cohort that purchased again one month later. Recent cohorts have shorter observable histories.")

    scored: pd.DataFrame = data["test_predictions"]  # type: ignore[assignment]
    scored = scored.copy()
    scored["priority"] = pd.cut(
        scored["estimated_return_likelihood"],
        bins=[-np.inf, .37, .65, np.inf],
        labels=["Low", "Medium", "High"],
    )
    selected = scored.loc[scored["selected_for_campaign"].eq(1)].sort_values(
        ["estimated_return_likelihood", "revenue_lifetime"], ascending=False
    )
    insight(
        f"The frozen model selected {len(selected):,} of {len(scored):,} historical test snapshots at the 0.37 campaign cutoff.",
        "Use the list to design a controlled campaign test; do not treat predicted return likelihood as incremental campaign lift.",
    )
    st.subheader("Campaign capacity simulator")
    st.caption(
        "Sensitivity analysis on the frozen historical test scores. This does not retune the final model or estimate campaign lift."
    )
    simulator_left, simulator_right = st.columns([0.7, 1.3])
    with simulator_left:
        simulation_threshold = st.slider(
            "Illustrative contact cutoff",
            min_value=0.10,
            max_value=0.90,
            value=0.37,
            step=0.01,
        )
        contact_cost = st.number_input(
            "Cost per contact (GBP)",
            min_value=0.0,
            value=1.0,
            step=0.25,
        )
        missed_value = st.number_input(
            "Illustrative cost of a missed returner (GBP)",
            min_value=0.0,
            value=2.0,
            step=0.5,
        )
    simulation = campaign_simulation(
        scored, simulation_threshold, contact_cost, missed_value
    )
    with simulator_right:
        simulator_cards = st.columns(4)
        for column, card in zip(
            simulator_cards,
            [
                ("Customers contacted", f"{simulation['contacts']:,.0f}", None),
                ("Observed precision", percentage(simulation["precision"]), None),
                ("Observed recall", percentage(simulation["recall"]), None),
                ("Illustrative cost", money(simulation["illustrative_cost"]), None),
            ],
        ):
            with column:
                kpi(*card)
        st.warning(
            "Choose a live campaign cutoff from validation evidence, operating capacity, and a randomized campaign test—not from final-test outcomes."
        )
    st.subheader("Historical campaign action table")
    st.dataframe(
        selected[["customer_id", "observation_date", "estimated_return_likelihood", "revenue_lifetime", "recency_days", "priority"]]
        .head(100)
        .style.format(
            {
                "estimated_return_likelihood": "{:.1%}",
                "revenue_lifetime": "£{:,.0f}",
                "recency_days": "{:,.0f} days",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption("Historical evaluation snapshot only. Customer IDs are pseudonymous source identifiers; no contact details are present.")

elif page == "Revenue Outlook":
    page_header(
        "Revenue Outlook",
        "Compare the frozen holdout forecast with actuals and view the eight-week expected range.",
        "What should we expect next?",
    )
    forecast_meta = data["forecast_meta"]
    forecast: pd.DataFrame = data["forecast"]  # type: ignore[assignment]
    forecast_test: pd.DataFrame = data["forecast_test"]  # type: ignore[assignment]
    insight(
        f"The 52-week seasonal baseline was the strongest rolling-backtest model. Final holdout sMAPE was {forecast_meta['final_test_metrics']['smape']:.1f}%.",
        "Use the range for planning scenarios, not as a guaranteed budget commitment.",
    )
    columns = st.columns(4)
    forecast_cards = [
        ("Selected method", "Seasonal naïve", None),
        ("Holdout MAE", money(forecast_meta["final_test_metrics"]["mae"]), None),
        ("Holdout sMAPE", f"{forecast_meta['final_test_metrics']['smape']:.1f}%", None),
        ("Forecast horizon", "8 weeks", None),
    ]
    for column, card in zip(columns, forecast_cards):
        with column:
            kpi(*card)
    st.subheader("Planning scenario")
    st.caption(
        "Apply a transparent management overlay to the statistical forecast. This is a planning assumption, not a new model prediction."
    )
    scenario_change = st.slider(
        "Expected commercial uplift or downside",
        min_value=-20,
        max_value=20,
        value=0,
        step=1,
        format="%d%%",
    )
    scenario_forecast = forecast.copy()
    scenario_forecast["scenario_revenue"] = (
        scenario_forecast["forecast_net_revenue"] * (1 + scenario_change / 100)
    )
    expected_total = float(forecast["forecast_net_revenue"].sum())
    scenario_total = float(scenario_forecast["scenario_revenue"].sum())
    scenario_columns = st.columns(3)
    with scenario_columns[0]:
        kpi("Baseline 8-week revenue", money(expected_total), None)
    with scenario_columns[1]:
        kpi("Scenario 8-week revenue", money(scenario_total), scenario_change / 100)
    with scenario_columns[2]:
        kpi("Planning difference", money(scenario_total - expected_total), None)
    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast_test["invoice_week"], y=forecast_test["net_revenue"], name="Actual", line={"color": COLORS["blue"]}, mode="lines+markers"))
        fig.add_trace(go.Scatter(x=forecast_test["invoice_week"], y=forecast_test["predicted_net_revenue"], name="Forecast", line={"color": COLORS["gold"]}, mode="lines+markers"))
        fig.update_layout(title="Frozen 16-week holdout", legend={"orientation": "h", "y": -0.15})
        fig.update_yaxes(tickprefix="£", tickformat="~s")
        st.plotly_chart(style_chart(fig), width="stretch")
    with right:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast["invoice_week"], y=forecast["upper_80"], mode="lines", line={"width": 0}, showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=forecast["invoice_week"], y=forecast["lower_80"], mode="lines", fill="tonexty", fillcolor="rgba(59,130,246,.25)", line={"width": 0}, name="Empirical 80% range"))
        fig.add_trace(go.Scatter(x=forecast["invoice_week"], y=forecast["forecast_net_revenue"], mode="lines+markers", name="Expected net revenue", line={"color": COLORS["blue"], "width": 3}))
        if scenario_change:
            fig.add_trace(
                go.Scatter(
                    x=scenario_forecast["invoice_week"],
                    y=scenario_forecast["scenario_revenue"],
                    mode="lines+markers",
                    name=f"Planning scenario ({scenario_change:+d}%)",
                    line={"color": COLORS["gold"], "width": 2, "dash": "dash"},
                )
            )
        fig.update_layout(title="Eight-week revenue outlook", legend={"orientation": "h", "y": -0.15})
        fig.update_yaxes(tickprefix="£", tickformat="~s", rangemode="tozero")
        st.plotly_chart(style_chart(fig), width="stretch")
    st.dataframe(
        forecast.style.format(
            {
                "forecast_net_revenue": "£{:,.0f}",
                "lower_80": "£{:,.0f}",
                "upper_80": "£{:,.0f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

elif page == "Data Explorer":
    page_header(
        "Data Explorer",
        "Inspect a bounded transaction sample behind the selected filters.",
        "What records support the current view?",
    )
    status = st.multiselect(
        "Transaction status",
        sorted(filtered["transaction_status"].unique().tolist()),
        placeholder="All statuses",
    )
    detail = filtered.loc[
        filtered["transaction_status"].isin(status) if status else np.ones(len(filtered), dtype=bool)
    ]
    st.caption(f"{len(detail):,} matching rows; displaying the first 500 in chronological order.")
    columns = [
        "invoice_date",
        "invoice",
        "stock_code",
        "description",
        "quantity",
        "unit_price",
        "country",
        "transaction_status",
        "net_line_revenue",
    ]
    st.dataframe(
        detail[columns].sort_values("invoice_date").head(500),
        width="stretch",
        hide_index=True,
    )
    st.download_button(
        "Download displayed rows",
        detail[columns].sort_values("invoice_date").head(500).to_csv(index=False),
        file_name="retailiq_filtered_sample.csv",
        mime="text/csv",
    )

else:
    page_header(
        "Methods & Trust",
        "Definitions, model evidence, source coverage, and limitations behind the dashboard.",
        "How much should we trust these numbers?",
    )
    st.subheader("Metric definitions")
    definitions = pd.DataFrame(
        [
            ("Gross revenue", "Positive quantity × positive unit price on valid sales."),
            ("Returned value", "Absolute value of cancellation/return rows with a positive source price."),
            ("Net revenue", "Gross revenue less returned value."),
            ("Active customer", "Known Customer ID with at least one valid sale in the selected period."),
            ("Repeat-purchase rate", "Share of known purchasing customers with at least two sales invoices in the selected period."),
            ("Estimated likelihood of returning", "Frozen model estimate of a valid purchase within the following 90 days."),
        ],
        columns=["Metric", "Definition"],
    )
    st.dataframe(definitions, width="stretch", hide_index=True)
    left, right = st.columns(2)
    with left:
        st.subheader("Classification evidence")
        test_meta = data["classification_test"]
        classification = pd.Series(test_meta["metrics"], name="Final test").to_frame()
        st.dataframe(classification.style.format("{:.3f}"), width="stretch")
        st.caption("Gradient Boosting; frozen cutoff 0.37; 14,341 historical test snapshots.")
    with right:
        st.subheader("Forecast evidence")
        forecast_meta = data["forecast_meta"]
        forecast_evidence = pd.Series(forecast_meta["final_test_metrics"], name="Final holdout").to_frame()
        st.dataframe(forecast_evidence.style.format("{:,.2f}"), width="stretch")
        st.caption("Seasonal naïve 52-week method; 16-week frozen historical holdout.")
    st.subheader("Model stability and errors")
    validation_metrics = data["classification_meta"]["validation_metrics"]
    test_metrics = data["classification_test"]["metrics"]
    comparison = pd.DataFrame(
        {
            "Metric": ["Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"],
            "Validation": [
                validation_metrics["precision"],
                validation_metrics["recall"],
                validation_metrics["f1"],
                validation_metrics["roc_auc"],
                validation_metrics["pr_auc"],
            ],
            "Final test": [
                test_metrics["precision"],
                test_metrics["recall"],
                test_metrics["f1"],
                test_metrics["roc_auc"],
                test_metrics["pr_auc"],
            ],
        }
    ).melt(id_vars="Metric", var_name="Evaluation", value_name="Score")
    evidence_left, evidence_right = st.columns([1.2, 0.8])
    with evidence_left:
        fig = px.bar(
            comparison,
            x="Metric",
            y="Score",
            color="Evaluation",
            barmode="group",
            title="Validation and final-test classification metrics",
            color_discrete_map={"Validation": COLORS["blue"], "Final test": COLORS["gold"]},
        )
        fig.update_yaxes(range=[0, 1], tickformat=".0%")
        st.plotly_chart(style_chart(fig, 390), width="stretch")
    with evidence_right:
        matrix = data["classification_test"]["confusion_matrix"]
        confusion = np.array(
            [
                [matrix["true_negative"], matrix["false_positive"]],
                [matrix["false_negative"], matrix["true_positive"]],
            ]
        )
        fig = px.imshow(
            confusion,
            text_auto=",d",
            x=["Predicted no return", "Predicted return"],
            y=["Actually no return", "Actually returned"],
            color_continuous_scale=["#111827", COLORS["blue"]],
            title="Final-test confusion matrix",
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style_chart(fig, 390), width="stretch")
    st.caption(
        "Recall declined on the final test while precision improved. This is why the dashboard presents both missed returners and unnecessary contacts—not accuracy alone."
    )
    st.subheader("Tree explainability")
    tree_left, tree_right = st.columns(2)
    with tree_left:
        st.image(
            ROOT / "reports/figures/decision_tree_structure.png",
            caption="Interpretable Decision Tree benchmark (standardized feature thresholds).",
            width="stretch",
        )
    with tree_right:
        st.image(
            ROOT / "reports/figures/random_forest_example_tree.png",
            caption="One illustrative tree from the Random Forest ensemble—not the entire forest.",
            width="stretch",
        )
    st.subheader("Known limitations")
    st.markdown(
        """
        - The source represents one UK-based retailer from 2009–2011.
        - Currency is assumed to be pounds from source context; no currency field is provided.
        - Missing customer IDs limit customer-level analysis.
        - A future purchase is not the same as causal campaign response.
        - The return model has weak recall among customers inactive for more than 181 days.
        - Forecasting uses only 104 complete weeks and lacks promotions, inventory, pricing, and macroeconomic drivers.
        - Forecast ranges are empirical historical ranges, not guarantees.
        """
    )
    st.caption(f"Data last observed: {max_date:%d %B %Y}. Dashboard is a static historical analytical product.")
