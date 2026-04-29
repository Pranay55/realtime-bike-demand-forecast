from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

def load_data(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df

def create_figure(df_actual: pd.DataFrame, df_predicted: pd.DataFrame | None, lookback_hours: int, datetime_col: str="datetime") -> go.Figure:
    df_actual['count'] = df_actual['registered'] + df_actual['casual']
    if df_predicted is not None and not df_predicted.empty:
        df_predicted[datetime_col] += pd.Timedelta(hours=1)
        max_time = df_predicted[datetime_col].max()
        current_time = max_time - pd.Timedelta(hours=1)
    else:
        current_time = df_actual[datetime_col].max()
        max_time = current_time

    min_time = current_time - pd.Timedelta(hours=lookback_hours)

    fig = go.Figure()

    if df_predicted is not None:
        df_pred_filtered = df_predicted[(df_predicted[datetime_col] >= min_time) & (df_predicted[datetime_col] <= max_time)]
        if not df_pred_filtered.empty:
            fig.add_trace(go.Scattergl(
                x=df_pred_filtered[datetime_col], y=df_pred_filtered['predicted_bike_count'],
                name = "Predicted",
                mode="lines+markers",
                line=dict(color='#1e8449', width=2),
                marker=dict(size=8),
            ))


    df_actual_filtered = df_actual[(df_actual[datetime_col] >= min_time) & (df_actual[datetime_col] <= max_time)]
    fig.add_trace(go.Scattergl(
        x=df_actual_filtered[datetime_col], y=df_actual_filtered['count'],
        name = "Actual",
        mode="lines+markers",
        line=dict(color='#c0392b', width=2),
        marker=dict(size=8),
    ))

    if not df_actual_filtered.empty:
        last_time  = str(df_actual_filtered[datetime_col].iloc[-1])
        fig.add_vline(x=last_time, line_width=2, line_dash="dash", line_color="gray")

    fig.update_layout(
        template="plotly_white",
        height=450,
        hovermode="x unified",
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis_title="Time",
        yaxis_title="Bike Count",
        legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=0.5),
        xaxis=dict(showspikes=True, spikemode="across", spikedash="dash", spikecolor="gray", spikethickness=1),
    )

    return fig