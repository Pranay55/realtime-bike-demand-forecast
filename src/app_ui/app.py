import os
import sys
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import yaml
from dash import Input, Output, callback, dcc, html

project_root = Path(__file__).parents[2]
sys.path.insert(0, str(project_root/"src"))
os.chdir(project_root)

from app_ui.utils import load_data, create_figure

with open(project_root/"conf"/"base"/"parameters.yml", "r") as f:
    config = yaml.safe_load(f)["ui"]

ACTUAL_DATA_PATH = project_root/config["actual_data_path"]  
PREDICTED_DATA_PATH = project_root/config["predictions_data_path"]

app = dash.Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dcc.Interval(id="interval", interval=config['update_interval_ms'], n_intervals=0),
    dbc.Row([
        dbc.Col([
            html.H4("Control Panel", style={"color":"#222"}),
            html.Div([
                html.Label("Plot diplay time (last N hours)", style={"color":"#222"}),
                dcc.Input(id="lookback-hours", type="number", min=1, step=1, value=config["default_lookback_hours"], style={"width":"100%"})
            ], className="card"),
            html.Div([
                html.H5("ML Application Overview", style={"marginTop":"8px", "color":"#222"}),
                html.Ul([
                    html.Li("ML App forecasts the amount of rented bikes for the next hour.", style={"marginBottom":"8px"}),
                    html.Li("The app consists of feature engineering, model training and inference pipelines.", style={"marginBottom":"8px"}),
                    html.Li("Inference runs every 2 seconds simulating 1 hour of dataset time", style={"marginBottom":"8px"}),
                    html.Li("The plot shows forecasted vs actual rented bike for the last N hours, where N is set in the control panel.", style={"marginBottom":"8px"}),
                    html.Li("The UI app and inference pipeline runs in Docker containers.", style={"marginBottom":"8px"}),
                    html.Li("The data and model are stored in Docker volumes.", style={"marginBottom":"8px"}),
                    html.Li("The app is built with Python Dash framework.", style={"marginBottom":"8px"}),
                    html.Li(["The code is available in the GitHub repository: ",
                        html.A("https://github.com/Pranay55/realtime-bike-demand-forecast", href="https://github.com/Pranay55/realtime-bike-demand-forecast", target="_blank")
                    ], style={"marginBottom":"8px"})
                ], style={"color":"#444", "fontSize":"14px","paddingLeft":"20px"}),
            ], className="card card-margin-top"),
        ],width=3,style={"paddingTop":"10px"}),

        dbc.Col([
            html.H5("Realtime Bike Count Predictions", style={"color":"#222","fontSize":"28px"}),
            dcc.Graph(id="graph", style={"backgroundColor":"#fff", "borderRadius":"12px", "padding":"10px"})
        ], width=9, style={"paddingTop":"10px"}),
    ],align="start"),
], fluid=True, style={"backgroundColor":"#e9e9f0", "minHeight":"100vh", "padding":"20px"})

@callback(
    Output("graph","figure"),
    [
        
        Input("lookback-hours","value"),
        Input("interval","n_intervals")
    ]
)
def update_graph(lookback_hours, _):
    df_actual = load_data(ACTUAL_DATA_PATH) 
    df_predicted = load_data(PREDICTED_DATA_PATH)

    if not lookback_hours or lookback_hours < 1:
        lookback_hours = config["default_lookback_hours"]
    
    figure = create_figure(df_actual, df_predicted, lookback_hours)
    return figure

server = app.server

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, host="0.0.0.0" , port=8050)