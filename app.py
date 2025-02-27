import warnings
from collections import Counter
from contextlib import suppress
from pathlib import Path
from typing import Iterable, TypeAlias
import dash_bootstrap_components as dbc
import fastf1 as f
import pandas as pd
import tomli
from dash import Dash, Input, Output, State, callback, html
from plotly import graph_objects as go
import f1_visualization.plotly_dash.graphs as pg
from f1_visualization._consts import SPRINT_FORMATS
from f1_visualization.plotly_dash.layout import app_layout, line_y_options, scatter_y_options
from f1_visualization.visualization import get_session_info, load_laps, teammate_comp_order
pd.options.mode.chained_assignment = None
warnings.filterwarnings(action="ignore", message="Driver", category=FutureWarning)
SessionInfo: TypeAlias = tuple[int, str, tuple[str]]
# must not be modified
LAP_DATA = load_laps()
with open(
    Path(__file__).absolute().parent
    / "f1_visualization"
    / "plotly_dash"
    / "visualization_config.toml",
    "rb",
) as toml:
    COLOR_PALETTE = tomli.load(toml)["relative"]["high_contrast_palette"]
def get_latest_round(season: int) -> tuple[int, int]:
    last_race_round, last_sprint_round = 0, 0
    with suppress(KeyError):
        last_race_round = LAP_DATA[season]["R"]["RoundNumber"].max()
    with suppress(KeyError):
        last_sprint_round = LAP_DATA[season]["S"]["RoundNumber"].max()
    return last_race_round, last_sprint_round
def convert_timedelta(df: pd.DataFrame) -> pd.DataFrame:
    timedelta_columns = ["Time", "PitInTime", "PitOutTime"]
    df[timedelta_columns] = df[timedelta_columns].ffill()
    for column in timedelta_columns:
        df[column] = df[column].dt.total_seconds()
    return df
def calculate_gap(driver: str, laps_df: pd.DataFrame) -> pd.DataFrame:
    df_driver = laps_df[laps_df["Driver"] == driver][["LapNumber", "Time"]]
    timing_column_name = f"{driver}Time"
    df_driver = df_driver.rename(columns={"Time": timing_column_name})
    laps_df = laps_df.merge(df_driver, how="left", on="LapNumber", validate="many_to_one")
    laps_df[f"GapTo{driver}"] = laps_df["Time"] - laps_df[timing_column_name]
    return laps_df.drop(columns=timing_column_name)
def configure_lap_slider(data: dict) -> tuple[int, list[int], dict[int, str]]:
    if not data:
        return 60, [1, 60], {i: str(i) for i in [1] + list(range(5, 61, 5))}
    try:
        num_laps = max(data["LapNumber"].values())
    except TypeError:
        df = pd.DataFrame.from_dict(data)
        num_laps = df["LapNumber"].max()
    marks = {i: str(i) for i in [1] + list(range(5, int(num_laps + 1), 5))}
    return num_laps, [1, num_laps], marks
def style_compound_options(compounds: Iterable[str]) -> list[dict]:
    compound_order = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]
    compounds = [compound for compound in compounds if compound in compound_order]
    compound_index = [compound_order.index(compound) for compound in compounds]
    sorted_compounds = sorted(zip(compounds, compound_index), key=lambda x: x[1])
    compounds = [compound for compound, _ in sorted_compounds]
    return [
        {
            "label": html.Span(compound, style={"color": COLOR_PALETTE[compound]}),
            "value": compound,
        }
        for compound in compounds
    ]
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.SANDSTONE],
    title="Armchair Strategist - A F1 Strategy Dashboard by Jaskirat Singh",
    update_title="Crunching numbers...",
)
server = app.server
app.layout = app_layout
@callback(
    Output("season", "options"),
    Input("season", "placeholder"),
)
def set_season_options(_: str) -> list[int]:
    return sorted(LAP_DATA.keys(), reverse=True)
@callback(
    Output("event", "options"),
    Output("event", "value"),
    Output("event-schedule", "data"),
    Output("last-race-round", "data"),
    Output("last-sprint-round", "data"),
    Input("season", "value"),
    prevent_initial_call=True,
)
def set_event_options(season: int | None) -> tuple[list[str], None, dict, int, int]:
    if season is None:
        return [], None, None
    schedule_data = f.get_event_schedule(season, include_testing=False)
    last_round_numbers = get_latest_round(season)
    schedule_data = schedule_data[schedule_data["RoundNumber"] <= max(last_round_numbers)]
    return (
        list(schedule_data["EventName"]),
        None,
        schedule_data.set_index("EventName").to_dict(orient="index"),
        *last_round_numbers,
    )
@callback(
    Output("session", "options"),
    Output("session", "value"),
    Input("event", "value"),
    State("event-schedule", "data"),
    State("last-race-round", "data"),
    State("last-sprint-round", "data"),
    prevent_initial_call=True,
)
def set_session_options(event: str | None, schedule: dict, last_race_round: int,
                        last_sprint_round: int) -> tuple[list[dict], None]:
    if event is None:
        return [], None
    round_number = schedule[event]["RoundNumber"]
    return [
        {
            "label": "Race",
            "value": "R",
            "disabled": round_number > last_race_round,
        },
        {
            "label": "Sprint",
            "value": "S",
            "disabled": (schedule[event]["EventFormat"] not in SPRINT_FORMATS)
            or (round_number > last_sprint_round),
        },
    ], None
@callback(
    Output("load-session", "disabled"),
Input("season", "value"),
Input("event", "value"),
Input("session", "value"),
prevent_initial_call=True,
)
def enable_load_session(season: int | None, event: str | None,
                        session: str | None) -> bool:
return not (season is not None and event is not None and session is not None)
@callback(
Output("add-gap", "disabled"), Input("load-session", "n_clicks"), prevent_initial_call=True
)
def enable_add_gap(n_clicks: int) -> bool:
return n_clicks == 0
@callback(
Output("session-info", "data"),
Input("load-session", "n_clicks"),
State("season", "value"),
State("event", "value"),
State("session", "value"),
State("teammate-comp", "value"),
prevent_initial_call=True,
)
def get_session_metadata(_: int,
                          season: int,
                          event: str,
                          session: str,
                          teammate_comp: bool) -> SessionInfo:
round_number, event_name, drivers = get_session_info(season, event,
                                                      session,
                                                      teammate_comp=teammate_comp)
event_name = f"{season} {event_name}"
return round_number, event_name, drivers
@callback(
Output("laps", "data"),
Input("load-session", "n_clicks"),
State("season", "value"),
State("event", "value"),
State("session", "value"),
prevent_initial_call=True,
)
def get_session_laps(_: int,
                      season: int,
                      event: str,
                      session: str) -> dict:
included_laps_data = LAP_DATA[season][session]
included_laps_data = included_laps_data[included_laps_data["EventName"] == event]
included_laps_data = convert_timedelta(included_laps_data)
return included_laps_data.to_dict()
@callback(
Output("drivers", "options"),
Output("drivers", "value"),
Output("drivers", "disabled"),
Output("gap-drivers", "options"),
Output("gap-drivers", "value"),
Output("gap-drivers", "disabled"),
Input("session-info", "data"),
prevent_initial_call=True,
)
def set_driver_dropdowns(session_info: SessionInfo):
drivers_list = session_info[2]
return drivers_list, drivers_list, False, drivers_list, [], False
@callback(
Output("scatter-y", "options"),
Output("line-y", "options"),
Output("scatter-y", "value"),
Output("line-y", "value"),
Input("laps", "data"),
prevent_initial_call=True,
)
def set_y_axis_dropdowns(data: dict) -> tuple[list[dict[str, str]], list[dict[str, str]], str, str]:
def readable_gap_col_name(col: str) -> str:
return f"Gap to {col[-3:]} (s)"
gap_cols_list = filter(lambda x: x.startswith("Gap"), data.keys())
gap_col_options_list = [{"label": readable_gap_col_name(col),
                         "value": col} for col in gap_cols_list]
return (
scatter_y_options + gap_col_options_list,
line_y_options + gap_col_options_list,
"LapTime",
"Position",
)
@callback(
Output("compounds", "options"),
Output("compounds", "value"),
Output("compounds", "disabled"),
Input("laps", "data"), prevent_initial_call=True,
)
def set_compounds_dropdown(data: dict) -> tuple[list[dict], list, bool]:
compound_lap_count_data = Counter(data["Compound"].values())
eligible_compounds_list = [
compound for compound, count in compound_lap_count_data.items()
if count >= (compound_lap_count_data.total() // 20)
]
return style_compound_options(eligible_compounds_list), [], False
@callback(
Output("laps", "data", allow_duplicate=True),
Input("add-gap", "n_clicks"),
State("gap-drivers", "value"), State("laps",
"data"),
running=[
(Output("gap-drivers","disabled"), True ,False),
(Output ("add-gap","disabled"), True ,False),
(Output ("add-gap","children"),"Calculating...","Add Gap"),
(Output ("add-gap","color"),"warning","success")
],
prevent_initial_call=True,
)
def add_gap_to_driver(_: int , drivers : list[str], data : dict ) -> dict :
laps_dataframe= pd.DataFrame.from_dict(data)
for driver in drivers:
if f"GapTo{driver}" not in laps_dataframe.columns:
laps_dataframe= calculate_gap(driver,laps_dataframe)
return laps_dataframe.to_dict()
@callback(
Output ("lap-numbers-scatter","max"),
Output ("lap-numbers-scatter","value"),
Output ("lap-numbers-scatter","marks"),
Input ("laps","data")
)
def set_scatterplot_slider(data : dict ) -> tuple[int,list[int],dict[int,str]]:
return configure_lap_slider(data)
@callback(
Output ("lap-numbers-line","max"),
Output ("lap-numbers-line","value"),
Output ("lap-numbers-line","marks"),
Input ("laps","data")
)
def set_lineplot_slider(data : dict ) -> tuple[int,list[int],dict[int,str]]:
return configure_lap_slider(data)
@callback(
Output ("strategy-plot","figure"),
Input ("drivers","value"),
State ("laps","data"),
State ("session-info","data"),
State ("teammate-comp","value")
)
def render_strategy_plot(drivers:list[str],
                         included_laps:data ,
                         session_info : SessionInfo ,
                         teammate_comp : bool ) -> go.Figure :
if not included_laps or not drivers :
return go.Figure()
included_laps_dataframe= pd.DataFrame.from_dict(included_laps )
included_laps_dataframe= included_laps_dataframe[included_laps_dataframe["Driver"].isin(drivers)]
if teammate_comp :
drivers= teammate_comp_order(included_laps_dataframe , drivers , by="LapTime")
event_name= session_info[1]
fig= pg.strategy_barplot(included_laps_dataframe , drivers )
fig.update_layout(title= event_name )
return fig
@callback(
Output ("scatterplot","figure"),
Input ("drivers","value"),
Input ("scatter-y","value"),
Input ("upper-bound-scatter","value"),
Input ("lap-numbers-scatter","value"),
State ("laps","data"),
State ("session-info","data"),
State ("teammate-comp","value")
)
def render_scatterplot(drivers:list[str],
                        y:str ,
                        upper_bound : float ,
                        lap_numbers:list[int],
                        included_laps:data ,
                        session_info : SessionInfo ,
                        teammate_comp : bool ) -> go.Figure :
if not included_laps or not drivers :
return go.Figure()
minimum , maximum= lap_numbers
lap_interval= range(minimum , maximum + 1 )
included_laps_dataframe= pd.DataFrame.from_dict(included_laps )
included_laps_dataframe= included_laps_dataframe[
(included_laps_dataframe["Driver"].isin(drivers))
& (included_laps_dataframe["PctFromFastest"] < (upper_bound - 100))
& (included_laps_dataframe["LapNumber"].isin(lap_interval))
]
if teammate_comp :
drivers= teammate_comp_order(included_laps_dataframe , drivers , y )
fig= pg.stats_scatterplot(included_laps_dataframe , drivers , y )
event_name= session_info[1]
fig.update_layout(title= event_name )
return fig
@callback(
Output ("lineplot","figure"),
Input ("drivers","value"),
Input ("line-y","value"),
Input ("upper-bound-line","value"),
Input ("lap-numbers-line","value"),
State ("laps","data"),
State ("session-info","data")
)
def render_lineplot(drivers:list[str],
                    y:str ,
                    upper_bound : float ,
                    lap_numbers:list[int],
                    included_laps:data ,
                    session_info : SessionInfo ) -> go.Figure :
if not included_laps or not drivers :
return go.Figure()
minimum , maximum= lap_numbers
lap_interval= range(minimum , maximum + 1 )
included_laps_dataframe= pd.DataFrame.from_dict(included_laps )
included_laps_dataframe= included_laps_dataframe[
(included_laps_dataframe["Driver"].isin(drivers))
& (included_laps_dataframe["LapNumber"].isin(lap_interval))
]
fig= pg.stats_lineplot(included_laps_dataframe , drivers , y , upper_bound )
event_name= session_info[1]
fig.update_layout(title= event_name )
return fig
@callback(
Output ("distplot","figure"),
Input ("drivers","value"),
Input ("upper-bound-dist","value"),
Input ("boxplot","value"),
State ("laps","data"),
State ("session-info","data"),
State ("teammate-comp","value")
)
def render_distplot(drivers:list[str],
                    upper_bound:int ,
                    boxplot : bool ,
                    included_laps:data ,
                    session_info : SessionInfo ,
                    teammate_comp : bool ) -> go.Figure :
if not included_laps or not drivers :
return go.Figure()
included_laps_dataframe= pd.DataFrame.from_dict(included_laps )
included_laps_dataframe= included_labs_dataframe[
(included_labss_dataframe["Driver"].isin(drivers))
& (included_labss_dataframe["PctFromFastest"] < (upper_bound - 100))
]
if teammate_comp :
drivers= teammate_comp_order(included_labss_dataframe , drivers , by="LapTime")
fig= pg.stats_distplot(included_labss_dataframe , drivers , boxplot )
event_name=session_info[1]
fig.update_layout(title= event_name )
return fig
@callback(
Output ("compound-plot" ,"figure"),
Input ("compounds" ,"value") ,
Input ("compound-unit" ,"value") ,
State ("laps" ,"data") ,
State ("session-info" ,"data")
)
def render_compound_plot(compounds:list[str],
                         show_seconds : bool ,
                         included_labs:data ,
                         session_info : SessionInfo ) -> go.Figure :
if not included_labss or not compounds :
return go.Figure()
included_labss_dataframe=pd.DataFrame.from_dict(included_labss)
included_labss_dataframe=(included_labss_df[
(included_labss_df["Compound"].isin(compounds)) & (included_labss_df["TyreLife"] != 1)
])
y="DeltaToLapRep" if show_seconds else"PctFromLapRep"
fig=pg.compounds_lineplot(included_labss_df,y , compounds )
event_name=session_info[1]
fig.update_layout(title= event_name )
return fig
if __name__ == "__main__":
app.run(host="0.0.0.0" , port=8000)
