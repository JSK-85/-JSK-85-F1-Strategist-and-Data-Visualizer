"""Dash app static layout specifications."""
import dash_bootstrap_components as dbc
from dash import dcc, html

def upper_bound_slider(slider_id: str, **kwargs) -> dcc.Slider:
    return dcc.Slider(
        min=100,
        max=150,
        marks={i: str(i) for i in range(100, 116, 5)} | {150: "Show All"},
        value=107,
        tooltip={"placement": "top"},
        id=slider_id,
        **kwargs,
    )

def lap_numbers_slider(slider_id: str, **kwargs) -> dcc.RangeSlider:
    return dcc.RangeSlider(
        min=1, step=1, allowCross=False, tooltip={"placement": "bottom"}, id=slider_id, **kwargs
    )

session_picker_row = dbc.Row(
    [
        dbc.Col(dcc.Dropdown(options=[], placeholder="Select a season", value=None, id="season")),
        dbc.Col(dcc.Dropdown(options=[], placeholder="Select an event", value=None, id="event")),
        dbc.Col(dcc.Dropdown(options=[], placeholder="Select a session", value=None, id="session")),
        dbc.Col(
            dcc.Dropdown(
                options=[
                    {"label": "Finishing order", "value": False},
                    {"label": "Teammate side-by-side", "value": True},
                ],
                value=False,
                clearable=False,
                id="teammate-comp",
            )
        ),
        dbc.Col(
            dbc.Button(
                children="Load Session / Reorder Drivers",
                n_clicks=0,
                disabled=True,
                color="success",
                id="load-session",
            )
        ),
    ]
)

add_gap_row = dbc.Row(
    dbc.Card(
        [
            dbc.CardHeader("Calculate gaps between drivers"),
            dbc.CardBody(
                [
                    dbc.Row(
                        dcc.Dropdown(
                            options=[],
                            value=[],
                            placeholder="Select drivers",
                            disabled=True,
                            multi=True,
                            id="gap-drivers",
                        )
                    ),
                    html.Br(),
                    dbc.Row(
                        dbc.Col(
                            dbc.Button(
                                "Add Gap",
                                color="success",
                                disabled=True,
                                n_clicks=0,
                                id="add-gap",
                            )
                        )
                    ),
                ]
            ),
        ]
    )
)

strategy_tab = dbc.Tab(dbc.Card(dbc.CardBody(dcc.Loading(dcc.Graph(id="strategy-plot")))), label="Strategy")

scatter_y_options = [
    {"label": "Lap Time", "value": "LapTime"},
    {"label": "Seconds to Median", "value": "DeltaToRep"},
    {"label": "Percent from Median", "value": "PctFromRep"},
    {"label": "Seconds to Fastest", "value": "DeltaToFastest"},
    {"label": "Percent from Fastest", "value": "PctFromFastest"},
    {"label": "Seconds to Adjusted Representative Time", "value": "DeltaToLapRep"},
    {"label": "Percent from Adjusted Representative Time", "value": "PctFromLapRep"},
]

scatterplot_tab = dbc.Tab(
    dbc.Card(
        dbc.CardBody(
            [
                dbc.Row(
                    dcc.Dropdown(
                        options=scatter_y_options,
                        value="LapTime",
                        placeholder="Select the variable for y-axis",
                        clearable=False,
                        id="scatter-y",
                    )
                ),
                html.Br(),
                dbc.Row(dcc.Loading(dcc.Graph(id="scatterplot"))),
                html.Br(),
                html.P("Filter out slow laps (default is 107% of the fastest lap):"),
                dbc.Row(upper_bound_slider(slider_id="upper-bound-scatter")),
                html.Br(),
                html.P("Select the range of lap numbers to include:"),
                dbc.Row(lap_numbers_slider(slider_id="lap-numbers-scatter")),
            ]
        )
    ),
    label="Scatterplot",
)
line_y_options = [{"label": "Position", "value": "Position"}] + scatter_y_options
lineplot_tab = dbc.Tab(
    dbc.Card(
        dbc.CardBody(
            [
                dbc.Row(
                    dcc.Dropdown(
