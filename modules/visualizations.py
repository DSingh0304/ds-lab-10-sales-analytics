"""
Interactive Visualizations Module
Creates high-performance, polished Plotly interactive charts with error handling.
"""

from typing import Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Modern corporate design palette
CHART_THEME = "plotly_white"
COLOR_PRIMARY = "#2563EB"    # Royal Blue
COLOR_SUCCESS = "#10B981"    # Emerald
COLOR_WARNING = "#F59E0B"    # Amber
COLOR_DANGER = "#EF4444"     # Rose
COLOR_PURPLE = "#8B5CF6"     # Purple
COLOR_PALETTE = ["#2563EB", "#10B981", "#F59E0B", "#8B5CF6", "#EC4899", "#06B6D4", "#64748B"]


def plot_sales_profit_trend(df: pd.DataFrame, freq: str = "Month") -> go.Figure:
    """
    Generate an interactive dual-axis / multi-series time-series trend of Sales and Profit.
    """
    if df.empty or "Order Date" not in df.columns or "Sales" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient date/sales data for trend analysis", showarrow=False)
        return fig

    temp_df = df.dropna(subset=["Order Date"]).copy()
    if freq == "Month":
        temp_df["Period"] = temp_df["Order Date"].dt.to_period("M").dt.to_timestamp()
    elif freq == "Quarter":
        temp_df["Period"] = temp_df["Order Date"].dt.to_period("Q").dt.to_timestamp()
    else:
        temp_df["Period"] = temp_df["Order Date"].dt.to_period("W").dt.to_timestamp()

    trend = (
        temp_df.groupby("Period")
        .agg(
            Total_Sales=("Sales", "sum"),
            Total_Profit=("Profit", "sum") if "Profit" in temp_df.columns else ("Sales", "count"),
            Order_Count=("Order ID", "nunique") if "Order ID" in temp_df.columns else ("Sales", "count"),
        )
        .reset_index()
        .sort_values(by="Period")
    )

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Sales Area / Bar
    fig.add_trace(
        go.Bar(
            x=trend["Period"],
            y=trend["Total_Sales"],
            name="Total Sales ($)",
            marker_color="rgba(37, 99, 235, 0.75)",
            hovertemplate="<b>%{x|%b %Y}</b><br>Sales: $%{y:,.2f}<extra></extra>",
        ),
        secondary_y=False,
    )

    # Profit Line with markers
    fig.add_trace(
        go.Scatter(
            x=trend["Period"],
            y=trend["Total_Profit"],
            name="Total Profit ($)",
            mode="lines+markers",
            line=dict(color=COLOR_SUCCESS, width=3),
            marker=dict(size=8, symbol="diamond"),
            hovertemplate="<b>%{x|%b %Y}</b><br>Profit: $%{y:,.2f}<extra></extra>",
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title=f"📈 Sales & Profit Trend over Time ({freq}ly Aggregation)",
        template=CHART_THEME,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        height=420,
    )

    fig.update_xaxes(title="Timeline", showgrid=True, gridcolor="#f1f5f9")
    fig.update_yaxes(title="Sales ($)", secondary_y=False, showgrid=True, gridcolor="#f1f5f9", tickprefix="$")
    fig.update_yaxes(title="Profit ($)", secondary_y=True, showgrid=False, tickprefix="$")

    return fig


def plot_category_performance(df: pd.DataFrame) -> go.Figure:
    """
    Generate an interactive comparative bar chart of Sales and Profit across Categories & Sub-Categories.
    """
    if df.empty or "Category" not in df.columns or "Sales" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient category data for chart", showarrow=False)
        return fig

    cat_perf = (
        df.groupby(["Category"])
        .agg(
            Total_Sales=("Sales", "sum"),
            Total_Profit=("Profit", "sum") if "Profit" in df.columns else ("Sales", "count"),
            Total_Quantity=("Quantity", "sum") if "Quantity" in df.columns else ("Sales", "count"),
        )
        .reset_index()
        .sort_values(by="Total_Sales", ascending=True)
    )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=cat_perf["Category"],
            x=cat_perf["Total_Sales"],
            name="Sales ($)",
            orientation="h",
            marker_color=COLOR_PRIMARY,
            hovertemplate="<b>%{y}</b><br>Sales: $%{x:,.2f}<extra></extra>",
        )
    )

    if "Profit" in df.columns:
        fig.add_trace(
            go.Bar(
                y=cat_perf["Category"],
                x=cat_perf["Total_Profit"],
                name="Profit ($)",
                orientation="h",
                marker_color=COLOR_SUCCESS,
                hovertemplate="<b>%{y}</b><br>Profit: $%{x:,.2f}<extra></extra>",
            )
        )

    fig.update_layout(
        title="🏷️ Sales vs. Profit Performance by Category",
        barmode="group",
        template=CHART_THEME,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        height=380,
    )
    fig.update_xaxes(title="Amount ($)", tickprefix="$", showgrid=True, gridcolor="#f1f5f9")
    fig.update_yaxes(title="Category")

    return fig


def plot_sub_category_breakdown(df: pd.DataFrame) -> go.Figure:
    """
    Generate a breakdown of Sales and Profit Margins across Sub-Categories.
    """
    if df.empty or "Sub-Category" not in df.columns or "Sales" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No sub-category data available", showarrow=False)
        return fig

    sub_perf = (
        df.groupby(["Category", "Sub-Category"])
        .agg(
            Sales=("Sales", "sum"),
            Profit=("Profit", "sum") if "Profit" in df.columns else ("Sales", "count"),
        )
        .reset_index()
        .sort_values(by="Sales", ascending=False)
    )

    fig = px.bar(
        sub_perf,
        x="Sub-Category",
        y="Sales",
        color="Category",
        color_discrete_sequence=COLOR_PALETTE,
        title="📦 Sales by Sub-Category with Category Categorization",
        hover_data={"Sales": ":$,.2f", "Profit": ":$,.2f"},
        template=CHART_THEME,
    )

    fig.update_layout(
        margin=dict(l=40, r=40, t=60, b=50),
        xaxis_tickangle=-45,
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(tickprefix="$", showgrid=True, gridcolor="#f1f5f9")

    return fig


def plot_regional_performance(df: pd.DataFrame) -> go.Figure:
    """
    Generate an interactive comparative chart of regional revenue and profitability.
    """
    if df.empty or "Region" not in df.columns or "Sales" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No regional data available", showarrow=False)
        return fig

    reg_perf = (
        df.groupby("Region")
        .agg(
            Sales=("Sales", "sum"),
            Profit=("Profit", "sum") if "Profit" in df.columns else ("Sales", "count"),
            Orders=("Order ID", "nunique") if "Order ID" in df.columns else ("Sales", "count"),
        )
        .reset_index()
        .sort_values(by="Sales", ascending=False)
    )

    reg_perf["Profit_Margin"] = (reg_perf["Profit"] / reg_perf["Sales"].replace(0, np.nan) * 100).fillna(0)

    fig = px.bar(
        reg_perf,
        x="Region",
        y="Sales",
        color="Profit_Margin",
        color_continuous_scale="Viridis",
        title="🗺️ Regional Performance: Sales with Profit Margin (%) Overlay",
        labels={"Profit_Margin": "Margin (%)"},
        hover_data={"Sales": ":$,.2f", "Profit": ":$,.2f", "Orders": True, "Profit_Margin": ":.1f%"},
        template=CHART_THEME,
    )

    fig.update_layout(
        margin=dict(l=40, r=40, t=60, b=40),
        height=380,
    )
    fig.update_yaxes(tickprefix="$", showgrid=True, gridcolor="#f1f5f9")

    return fig


def plot_sales_vs_profit_scatter(df: pd.DataFrame) -> go.Figure:
    """
    Generate an interactive scatter plot of Sales vs. Profit per transaction,
    colored by Category and sized by Quantity.
    """
    if df.empty or "Sales" not in df.columns or "Profit" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient data for scatter plot", showarrow=False)
        return fig

    plot_df = df.copy()
    if "Quantity" not in plot_df.columns:
        plot_df["Quantity"] = 5

    hover_cols = [c for c in ["Order ID", "Product Name", "Customer ID", "Discount"] if c in plot_df.columns]

    fig = px.scatter(
        plot_df,
        x="Sales",
        y="Profit",
        color="Category" if "Category" in plot_df.columns else None,
        size="Quantity",
        hover_data=hover_cols,
        title="🎯 Transaction Matrix: Sales vs. Profit (Bubble Size = Quantity)",
        color_discrete_sequence=COLOR_PALETTE,
        template=CHART_THEME,
    )

    # Add zero-profit benchmark line
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="#dc2626",
        annotation_text="Breakeven (Profit = $0)",
        annotation_position="bottom right",
    )

    fig.update_layout(
        margin=dict(l=40, r=40, t=60, b=40),
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(title="Sales Amount ($)", tickprefix="$", showgrid=True, gridcolor="#f1f5f9")
    fig.update_yaxes(title="Profit / Loss ($)", tickprefix="$", showgrid=True, gridcolor="#f1f5f9")

    return fig


def plot_segment_donut(df: pd.DataFrame) -> go.Figure:
    """
    Generate an interactive Donut chart showing customer segment revenue distribution.
    """
    if df.empty or "Segment" not in df.columns or "Sales" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No segment data available", showarrow=False)
        return fig

    seg_summary = df.groupby("Segment")["Sales"].sum().reset_index()

    fig = px.pie(
        seg_summary,
        names="Segment",
        values="Sales",
        hole=0.48,
        title="👥 Revenue Share by Customer Segment",
        color_discrete_sequence=COLOR_PALETTE,
        template=CHART_THEME,
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>Sales: $%{value:,.2f}<br>Share: %{percent}<extra></extra>",
    )

    fig.update_layout(
        margin=dict(l=30, r=30, t=50, b=30),
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
    )

    return fig


def plot_shipping_duration_box(df: pd.DataFrame) -> go.Figure:
    """
    Generate an interactive Box plot showing shipping duration distribution across Ship Modes.
    """
    if df.empty or "Shipping Days" not in df.columns or "Ship Mode" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No shipping days data available", showarrow=False)
        return fig

    fig = px.box(
        df,
        x="Ship Mode",
        y="Shipping Days",
        color="Ship Mode",
        color_discrete_sequence=COLOR_PALETTE,
        title="🚚 Shipping Duration (Days) Distribution by Shipping Mode",
        template=CHART_THEME,
        points="all",
    )

    fig.update_layout(
        margin=dict(l=40, r=40, t=60, b=40),
        height=380,
        showlegend=False,
    )
    fig.update_yaxes(title="Delivery Time (Days)", showgrid=True, gridcolor="#f1f5f9")

    return fig
