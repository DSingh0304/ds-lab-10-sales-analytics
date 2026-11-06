"""
Interactive Visualizations Module
Plotly charts sharing one restrained visual language: a fixed categorical order,
a single-hue emphasis colour, quiet grids, and no dual axes.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Fixed categorical order (assigned by entity, never cycled).
COLOR_PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
COLOR_PRIMARY = COLOR_PALETTE[0]
COLOR_NEGATIVE = "#c8402f"
INK = "#1c1b19"
INK_SOFT = "#5d5b55"
GRID = "#e9e7e1"
FONT = "Inter, 'Segoe UI', system-ui, sans-serif"


def _style(fig: go.Figure, title: str, height: int = 380, legend: bool = True) -> go.Figure:
    """Apply the shared chart chrome."""
    fig.update_layout(
        template="plotly_white",
        title=dict(text=title, x=0, xanchor="left", font=dict(size=15, color=INK)),
        font=dict(family=FONT, size=12, color=INK_SOFT),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=64, b=8),
        height=height,
        showlegend=legend,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0,
            title_text="", font=dict(size=12),
        ),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=GRID, font=dict(family=FONT, color=INK)),
        bargap=0.35,
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, ticks="", automargin=True, title_font=dict(size=12))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, ticks="", automargin=True, title_font=dict(size=12))
    return fig


def _empty(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(color=INK_SOFT, size=13))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _style(fig, "", height=300, legend=False)


def plot_sales_profit_trend(df: pd.DataFrame, freq: str = "Month") -> go.Figure:
    """Sales and profit over time as two aligned panels sharing one time axis."""
    if df.empty or "Order Date" not in df.columns or "Sales" not in df.columns:
        return _empty("Insufficient date/sales data for trend analysis")

    temp_df = df.dropna(subset=["Order Date"]).copy()
    code = {"Month": "M", "Quarter": "Q"}.get(freq, "W")
    temp_df["Period"] = temp_df["Order Date"].dt.to_period(code).dt.to_timestamp()

    has_profit = "Profit" in temp_df.columns
    agg = {"Total_Sales": ("Sales", "sum")}
    if has_profit:
        agg["Total_Profit"] = ("Profit", "sum")
    trend = temp_df.groupby("Period").agg(**agg).reset_index().sort_values("Period")

    rows = 2 if has_profit else 1
    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.1,
        row_heights=[0.6, 0.4] if has_profit else [1.0],
        subplot_titles=("Sales", "Profit") if has_profit else ("Sales",),
    )
    fig.add_trace(
        go.Bar(
            x=trend["Period"], y=trend["Total_Sales"], name="Sales",
            marker_color=COLOR_PRIMARY,
            hovertemplate="%{x|%b %Y}<br>Sales: $%{y:,.0f}<extra></extra>",
        ),
        row=1, col=1,
    )
    if has_profit:
        fig.add_trace(
            go.Scatter(
                x=trend["Period"], y=trend["Total_Profit"], name="Profit",
                mode="lines+markers", line=dict(color=COLOR_PALETTE[2], width=2),
                marker=dict(size=7, line=dict(color="#ffffff", width=2)),
                hovertemplate="%{x|%b %Y}<br>Profit: $%{y:,.0f}<extra></extra>",
            ),
            row=2, col=1,
        )
        fig.add_hline(y=0, line_width=1, line_color=INK_SOFT, row=2, col=1)

    _style(fig, f"Sales and profit by {freq.lower()}", height=480, legend=False)
    fig.update_yaxes(tickprefix="$", tickformat="~s")
    fig.update_annotations(font=dict(size=12, color=INK_SOFT), xanchor="left", x=0)
    return fig


def plot_category_performance(df: pd.DataFrame) -> go.Figure:
    """Sales vs profit by category (same unit, one axis)."""
    if df.empty or "Category" not in df.columns or "Sales" not in df.columns:
        return _empty("Insufficient category data for chart")

    agg = {"Total_Sales": ("Sales", "sum")}
    if "Profit" in df.columns:
        agg["Total_Profit"] = ("Profit", "sum")
    cat_perf = df.groupby("Category").agg(**agg).reset_index().sort_values("Total_Sales")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=cat_perf["Category"], x=cat_perf["Total_Sales"], name="Sales", orientation="h",
        marker_color=COLOR_PRIMARY,
        hovertemplate="<b>%{y}</b><br>Sales: $%{x:,.0f}<extra></extra>",
    ))
    if "Profit" in df.columns:
        fig.add_trace(go.Bar(
            y=cat_perf["Category"], x=cat_perf["Total_Profit"], name="Profit", orientation="h",
            marker_color=COLOR_PALETTE[1],
            hovertemplate="<b>%{y}</b><br>Profit: $%{x:,.0f}<extra></extra>",
        ))
    _style(fig, "Sales and profit by category")
    fig.update_layout(barmode="group", bargap=0.3)
    fig.update_xaxes(tickprefix="$", tickformat="~s", showgrid=True, gridcolor=GRID)
    fig.update_yaxes(showgrid=False)
    return fig


def plot_sub_category_breakdown(df: pd.DataFrame) -> go.Figure:
    """Sales by sub-category, coloured by parent category."""
    if df.empty or "Sub-Category" not in df.columns or "Sales" not in df.columns:
        return _empty("No sub-category data available")

    agg = {"Sales": ("Sales", "sum")}
    if "Profit" in df.columns:
        agg["Profit"] = ("Profit", "sum")
    sub_perf = df.groupby(["Category", "Sub-Category"]).agg(**agg).reset_index().sort_values("Sales")

    fig = px.bar(
        sub_perf, y="Sub-Category", x="Sales", color="Category", orientation="h",
        color_discrete_sequence=COLOR_PALETTE,
        hover_data={"Sales": ":$,.0f", **({"Profit": ":$,.0f"} if "Profit" in sub_perf else {})},
    )
    _style(fig, "Sales by sub-category", height=max(380, 24 * len(sub_perf) + 110))
    fig.update_layout(bargap=0.3)
    fig.update_xaxes(tickprefix="$", tickformat="~s", showgrid=True, gridcolor=GRID, title="")
    fig.update_yaxes(showgrid=False, title="")
    return fig


def plot_regional_performance(df: pd.DataFrame) -> go.Figure:
    """Regional sales; margin appears as a bar label rather than a second encoding."""
    if df.empty or "Region" not in df.columns or "Sales" not in df.columns:
        return _empty("No regional data available")

    agg = {"Sales": ("Sales", "sum")}
    if "Profit" in df.columns:
        agg["Profit"] = ("Profit", "sum")
    if "Order ID" in df.columns:
        agg["Orders"] = ("Order ID", "nunique")
    reg = df.groupby("Region").agg(**agg).reset_index().sort_values("Sales", ascending=False)

    if "Profit" in reg:
        reg["Margin"] = (reg["Profit"] / reg["Sales"].replace(0, np.nan) * 100).fillna(0)
        labels = [f"{m:.1f}% margin" for m in reg["Margin"]]
        custom = np.stack([reg["Profit"], reg["Margin"]], axis=-1)
        hover = "<b>%{x}</b><br>Sales: $%{y:,.0f}<br>Profit: $%{customdata[0]:,.0f}<br>Margin: %{customdata[1]:.1f}%<extra></extra>"
    else:
        labels, custom = None, None
        hover = "<b>%{x}</b><br>Sales: $%{y:,.0f}<extra></extra>"

    fig = go.Figure(go.Bar(
        x=reg["Region"], y=reg["Sales"], marker_color=COLOR_PRIMARY,
        text=labels, textposition="outside", cliponaxis=False,
        textfont=dict(color=INK_SOFT, size=11),
        customdata=custom, hovertemplate=hover,
    ))
    _style(fig, "Sales by region, labelled with profit margin", legend=False)
    fig.update_yaxes(tickprefix="$", tickformat="~s")
    return fig


def plot_sales_vs_profit_scatter(df: pd.DataFrame) -> go.Figure:
    """Per-transaction sales vs profit, coloured by category."""
    if df.empty or "Sales" not in df.columns or "Profit" not in df.columns:
        return _empty("Insufficient data for scatter plot")

    plot_df = df.copy()
    hover_cols = [c for c in ["Order ID", "Product Name", "Customer ID", "Discount"] if c in plot_df.columns]
    has_qty = "Quantity" in plot_df.columns
    if has_qty:
        plot_df["Quantity"] = plot_df["Quantity"].clip(lower=1)

    fig = px.scatter(
        plot_df, x="Sales", y="Profit",
        color="Category" if "Category" in plot_df.columns else None,
        size="Quantity" if has_qty else None, size_max=14, opacity=0.8,
        hover_data=hover_cols, color_discrete_sequence=COLOR_PALETTE,
    )
    fig.update_traces(marker=dict(line=dict(color="#ffffff", width=1.5)))
    fig.add_hline(
        y=0, line_dash="dot", line_color=COLOR_NEGATIVE, line_width=1,
        annotation_text="Breakeven", annotation_position="bottom right",
        annotation_font=dict(color=COLOR_NEGATIVE, size=11),
    )
    _style(fig, "Sales against profit per transaction (bubble size is quantity)", height=420)
    fig.update_xaxes(title="Sales", tickprefix="$", tickformat="~s")
    fig.update_yaxes(title="Profit", tickprefix="$", tickformat="~s")
    return fig


def plot_segment_donut(df: pd.DataFrame) -> go.Figure:
    """Revenue share by customer segment."""
    if df.empty or "Segment" not in df.columns or "Sales" not in df.columns:
        return _empty("No segment data available")

    seg = df.groupby("Segment")["Sales"].sum().reset_index().sort_values("Sales", ascending=False)
    fig = go.Figure(go.Pie(
        labels=seg["Segment"], values=seg["Sales"], hole=0.6, sort=False,
        marker=dict(colors=COLOR_PALETTE[: len(seg)], line=dict(color="#ffffff", width=2)),
        textinfo="percent", textfont=dict(color="#ffffff", size=12),
        hovertemplate="<b>%{label}</b><br>Sales: $%{value:,.0f}<br>Share: %{percent}<extra></extra>",
    ))
    _style(fig, "Revenue share by customer segment")
    fig.update_layout(legend=dict(orientation="h", y=-0.02, x=0.5, xanchor="center", yanchor="top"))
    return fig


def plot_shipping_duration_box(df: pd.DataFrame) -> go.Figure:
    """Distribution of shipping days by ship mode."""
    if df.empty or "Shipping Days" not in df.columns or "Ship Mode" not in df.columns:
        return _empty("No shipping days data available")

    fig = px.box(df, x="Ship Mode", y="Shipping Days", points="outliers")
    fig.update_traces(
        marker_color=COLOR_PRIMARY, line_color=COLOR_PRIMARY,
        fillcolor="rgba(42,120,214,0.18)", marker_size=4,
    )
    _style(fig, "Shipping days by ship mode", legend=False)
    fig.update_yaxes(title="Days")
    fig.update_xaxes(title="")
    return fig
