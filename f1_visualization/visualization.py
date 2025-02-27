"""Plotting functions and other visualization helpers."""
import logging
from functools import lru_cache
from math import ceil
from typing import Iterable, Literal, Optional

import fastf1 as f
import fastf1.plotting as p
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import rcParams

from f1_visualization._consts import (
    COMPOUND_SELECTION,
    DATA_PATH,
    SESSION_NAMES,
    VISUAL_CONFIG,
)
from f1_visualization._types import Figure, Session

logging.basicConfig(level=logging.INFO, format="%(levelname)s\t%(filename)s\t%(message)s")
jaskirat_logger = logging.getLogger(__name__)


def _correct_data_types(laps_df: pd.DataFrame) -> pd.DataFrame:
    laps_df[["Time", "PitInTime", "PitOutTime"]] = laps_df[["Time", "PitInTime", "PitOutTime"]].apply(pd.to_timedelta)
    laps_df["TrackStatus"] = laps_df["TrackStatus"].astype(str)
    laps_df["FreshTyre"] = laps_df["FreshTyre"].astype(str)

    return laps_df

def load_transformed_laps_data() -> dict[int, dict[str, pd.DataFrame]]:
    dataframes = {}

    for file_path in DATA_PATH.glob("**/transformed_*.csv"):
        session_year = int(file_path.stem.split("_")[-1])
        session_category = SESSION_NAMES[file_path.parent.name]
        laps_dataframe = pd.read_csv(
            file_path,
            header=0,
            true_values=["True"],
            false_values=["False"],
        )
        _correct_data_types(laps_dataframe)

        if session_year not in dataframes:
            dataframes[session_year] = {}

        dataframes[session_year][session_category] = laps_dataframe

    return dataframes

JASKIRAT_DF_DICT = load_transformed_laps_data()

def _find_compound_display_order(compound_labels: Iterable[str]) -> list[int]:
    old_indices = list(range(len(compound_labels)))
    ordered_labels = []

    if any(name in compound_labels for name in ("HYPERSOFT", "ULTRASOFT", "SUPERSOFT", "SUPERHARD")):
        ordered_labels = VISUAL_CONFIG["absolute"]["labels"]["18"]
    elif any(label.startswith("C") for label in compound_labels):
        ordered_labels = VISUAL_CONFIG["absolute"]["labels"]["19_22"]
    else:
        ordered_labels = VISUAL_CONFIG["relative"]["labels"]

    positions = [ordered_labels.index(label) for label in compound_labels]

    return [original_index for _, original_index in sorted(zip(positions, old_indices))]

def _get_plot_settings(session_year: int, use_absolute_compounds: bool) -> tuple:
    if use_absolute_compounds:
        if session_year == 2018:
            return (
                "CompoundName",
                VISUAL_CONFIG["absolute"]["palette"]["18"],
                VISUAL_CONFIG["absolute"]["markers"]["18"],
                VISUAL_CONFIG["absolute"]["labels"]["18"],
            )
        if session_year < 2023:
            return (
                "CompoundName",
                VISUAL_CONFIG["absolute"]["palette"]["19_22"],
                VISUAL_CONFIG["absolute"]["markers"]["19_22"],
                VISUAL_CONFIG["absolute"]["labels"]["19_22"],
            )

        return (
            "CompoundName",
            VISUAL_CONFIG["absolute"]["palette"]["23_"],
            VISUAL_CONFIG["absolute"]["markers"]["23_"],
            VISUAL_CONFIG["absolute"]["labels"]["23_"],
        )

    return (
        "Compound",
        VISUAL_CONFIG["relative"]["palette"],
        VISUAL_CONFIG["relative"]["markers"],
        VISUAL_CONFIG["relative"]["labels"],
    )

def get_session_drivers(
    session_data: Session,
    driver_identifiers: Optional[Iterable[str | int] | str | int] = None,
    sorting_key: str = "Position",
) -> list[str]:
    results = session_data.results.sort_values(by=sorting_key, kind="stable")
    if driver_identifiers is None:
        return list(results["Abbreviation"].unique())
    if isinstance(driver_identifiers, int):
        driver_identifiers = results["Abbreviation"].unique()[:driver_identifiers]
        return list(driver_identifiers)
    if isinstance(driver_identifiers, str):
        driver_identifiers = [driver_identifiers]

    extracted_drivers = []
    for driver_id in driver_identifiers:
        if isinstance(driver_id, (int, float)):
            driver_id = str(int(driver_id))
        extracted_drivers.append(session_data.get_driver(driver_id)["Abbreviation"])

    return extracted_drivers

@lru_cache(maxsize=256)
def get_race_session_info(
    session_year: int,
    event_identifier: int | str,
    session_type: str,
    driver_selection: Optional[tuple[str | int] | str | int] = None,
    use_teammate_ordering: bool = False,
) -> tuple[int, str, tuple[str]]:
    session_data = f.get_session(session_year, event_identifier, session_type)
    session_data.load(laps=False, telemetry=False, weather=False, messages=False)
    event_round_number = session_data.event["RoundNumber"]
    event_title = f"{session_data.event['EventName']} - {session_data.name}"

    if use_teammate_ordering:
        driver_selection = get_session_drivers(session_data, driver_selection, by="TeamName")
    else:
        driver_selection = get_session_drivers(session_data, driver_selection)

    return event_round_number, event_title, tuple(driver_selection)

def get_driver_display_color(driver_code: str) -> str:
    if p.DRIVER_TRANSLATE.get(driver_code, "NA") in p.DRIVER_COLORS:
        return p.DRIVER_COLORS[p.DRIVER_TRANSLATE[driver_code]]

    return "#2F4F4F"

def calculate_time_gap(
    target_driver: str, laps_df: Optional[pd.DataFrame] = None, update_global: bool = False, **kwargs
) -> pd.DataFrame:
    assert not (not update_global and laps_df is None), "laps_df must be provided if not editing in-place."

    if update_global:
        assert (
            "session_year" in kwargs and "session_type" in kwargs
        ), "Setting update_global=True requires specifying session_year and session_type."
        session_year, session_type = kwargs["session_year"], kwargs["session_type"]
        laps_df = JASKIRAT_DF_DICT[session_year][session_type]

    assert target_driver.upper() in laps_df["Driver"].unique(), "Driver not available."

    target_driver_laps = laps_df[laps_df["Driver"] == target_driver][["RoundNumber", "LapNumber", "Time"]]
    timing_column_name = f"{target_driver}Time"
    target_driver_laps = target_driver_laps.rename(columns={"Time": timing_column_name})

    target_driver_laps[timing_column_name] = target_driver_laps[timing_column_name].ffill()

    laps_df = laps_df.merge(
        target_driver_laps, how="left", on=["RoundNumber", "LapNumber"], validate="many_to_one"
    )
    laps_df[f"GapTo{target_driver}"] = (
        laps_df["Time"] - laps_df[timing_column_name]
    ).dt.total_seconds()
    laps_df = laps_df.drop(columns=timing_column_name)

    if update_global:
        JASKIRAT_DF_DICT[kwargs["session_year"]][kwargs["session_type"]] = laps_df

    return laps_df

def reorder_drivers_by_teammate_metric_gap(
    laps_data: pd.DataFrame, driver_list: tuple[str], metric: str
) -> tuple[str]:
    metric_median_values = laps_data.groupby("Driver")[metric].median(numeric_only=True)
    min_valid_laps = 5
    drivers_with_enough_laps = laps_data.groupby("Driver").size().loc[lambda x: x > min_valid_laps].index
    team_median_gaps = []

    for i in range(0, len(driver_list) - 1, 2):
        teammates = driver_list[i], driver_list[i + 1]

        if teammates[0] in drivers_with_enough_laps and teammates[1] in drivers_with_enough_laps:
            median_gap = abs(metric_median_values[teammates[0]] - metric_median_values[teammates[1]])
            team_median_gaps.append([teammates, median_gap])
        else:
            for driver in teammates:
                if driver in drivers_with_enough_laps:
                    team_median_gaps.append([tuple(driver), 0])
                else:
                    jaskirat_logger.warning(
                        "%s has less than %s laps of data and will not be plotted",
                        driver,
                        min_valid_laps
                    )

    team_median_gaps.sort(key=lambda x: x[1], reverse=True)

    remaining_driver = driver_list[-1:] if len(driver_list) % 2 == 1 else []

    driver_list = [driver for team in team_median_gaps for driver in team[0]]
    driver_list.extend(remaining_driver)

    return tuple(driver_list)

def _is_lap_under_sc(lap_data: pd.Series) -> bool:
    return "4" in lap_data.loc["TrackStatus"]

def _is_lap_under_vsc(lap_data: pd.Series) -> bool:
    return (("6" in lap_data.loc["TrackStatus"]) or ("7" in lap_data.loc["TrackStatus"])) and (
        "4" not in lap_data.loc["TrackStatus"]
    )

def _find_safety_car_periods(laps_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    sc_laps_nums = np.sort(laps_df[laps_df.apply(_is_lap_under_sc, axis=1)]["LapNumber"].unique())
    vsc_laps_nums = np.sort(laps_df[laps_df.apply(_is_lap_under_vsc, axis=1)]["LapNumber"].unique())

    return sc_laps_nums, vsc_laps_nums

def _add_shaded_sc_periods(sc_laps: np.ndarray, vsc_laps: np.ndarray):
    sc_laps = np.append(sc_laps, [-1])
    vsc_laps = np.append(vsc_laps, [-1])

    def plot_intervals(laps, label, shading=None):
        interval_start = 0
        interval_end = 1

        while interval_end < len(laps):
            if laps[interval_end] == laps[interval_end - 1] + 1:
                interval_end += 1
            else:
                if interval_end - interval_start > 1:
                    plt.axvspan(
                        xmin=laps[interval_start] - 1,
                        xmax=laps[interval_end - 1] - 1,
                        alpha=0.5,
                        color="orange",
                        label=label if interval_start == 0 else "_",
                        hatch=shading,
                    )
                else:
                    plt.axvspan(
                        xmin=laps[interval_start] - 1,
                        xmax=laps[interval_start],
                        alpha=0.5,
                        color="orange",
                        label=label if interval_start == 0 else "_",
                        hatch=shading,
                    )
                interval_start = interval_end
                interval_end += 1

    plot_intervals(sc_laps, "SC")
    plot_intervals(vsc_laps, "VSC", "-")

def _map_relative_to_absolute_compound(
    session_year: int, event_round: int, compound_names: Iterable[str]
) -> tuple[str]:
    compound_mapping = {"SOFT": 2, "MEDIUM": 1, "HARD": 0}
    if session_year == 2018:
        compound_mapping = {"SOFT": 0, "MEDIUM": 1, "HARD": 2}

    return_values = []

    for compound in compound_names:
        return_values.append(
            COMPOUND_SELECTION[str(session_year)][str(event_round)][compound_mapping[compound]]
        )

    return tuple(return_values)

def _validate_and_prepare_inputs(
    session_years: int | Iterable[int],
    event_identifiers: int | str | Iterable[str | int],
    session_types: str | Iterable[str],
    y_axis: str,
    compound_selections: Iterable[str],
    x_axis: str,
    percentage_cutoff: int | float,
    use_absolute_compounds: bool,
) -> tuple[list[f.events.Event], list[pd.DataFrame]]:
    compound_selections = [compound.upper() for compound in compound_selections]

    for compound in compound_selections:
        assert compound in {
            "SOFT",
            "MEDIUM",
            "HARD",
        }, f"requested compound {compound} is not valid"

    if x_axis not in {"LapNumber", "TyreLife"}:
        jaskirat_logger.warning(
            "Using %s as the x-axis is not recommended. (Recommended x: LapNumber, TyreLife)",
            x_axis,
        )

    if not use_absolute_compounds and len(event_identifiers) > 1:
        jaskirat_logger.warning(
            """
            Different events may use different compounds under the same name!
            e.g. SOFT may be any of C3 to C5 dependinging on the event
            """
        )

    if isinstance(session_years, (int, str)):
        session_years = [session_years]

    if isinstance(event_identifiers, (int, str)):
        event_identifiers = [event_identifiers]

    if isinstance(session_types, str):
        session_types = [session_types]

    assert (
        len(session_years) == len(event_identifiers) == len(session_types)
    ), f"Arguments {session_years}, {event_identifiers}, {session_types} have different lengths."

    event_data_objs = [f.get_event(session_years[i], event_identifiers[i]) for i in range(len(session_years))]

    laps_data_list = []

    for session_year, event_data, session_type in zip(session_years, event_data_objs, session_types):
        all_laps_df = JASKIRAT_DF_DICT[session_year][session_type]
        valid_laps_df = all_laps_df[
            (all_laps_df["RoundNumber"] == event_data["RoundNumber"])
            & (all_laps_df["IsValid"])
            & (all_laps_df["Compound"].isin(compound_selections))
            & (all_laps_df["PctFromFastest"] < percentage_cutoff)
        ]

        if y_axis in {"PctFromLapRep", "DeltaToLapRep"}:
            valid_laps_df = valid_laps_df[valid_laps_df["PctFromLapRep"] > -5]

        laps_data_list.append(valid_laps_df)

    return event_data_objs, laps_data_list

def visualize_driver_scatterplot(
    session_year: int,
    event_identifier: int | str,
    session_type: str = "R",
    driver_selection: Optional[Iterable[str | int] | str | int] = None,
    y_axis: str = "LapTime",
    percentage_cutoff: int | float = 10,
    use_absolute_compounds: bool = False,
    use_teammate_ordering: bool = False,
    laps_segment: Optional[list[int]] = None,
) -> Figure:
    plt.style.use("dark_background")
    title_font_params = {
        "fontsize": rcParams["axes.titlesize"],
        "fontweight": rcParams["axes.titleweight"],
        "color": rcParams["axes.titlecolor"],
        "verticalalignment": "baseline",
        "horizontalalignment": "center",
    }

    if not isinstance(driver_selection, (int, str)) and driver_selection is not None:
        driver_selection = tuple(driver_selection)

    event_round_number, event_title, driver_selection = get_race_session_info(
        session_year, event_identifier, session_type, driver_selection, use_teammate_ordering
    )
    all_laps_data = JASKIRAT_DF_DICT[session_year][session_type]
    driver_laps_data = all_laps_data[
        (all_laps_data["RoundNumber"] == event_round_number) & (all_laps_data["Driver"].isin(driver_selection))
    ]

    if use_teammate_ordering:
        driver_selection = reorder_drivers_by_teammate_metric_gap(driver_laps_data, driver_selection, y_axis)

    if laps_segment is not None:
        assert sorted(laps_segment) == list(range(laps_segment[0], laps_segment[-1] + 1))
        driver_laps_data = driver_laps_data[driver_laps_data["LapNumber"].isin(laps_segment)]

    max_columns = 4 if use_teammate_ordering else 5
    num_rows = ceil(len(driver_selection) / max_columns)
    num_cols = len(driver_selection) if len(driver_selection) < max_columns else max_columns
    fig, axes = plt.subplots(
        nrows=num_rows,
        ncols=num_cols,
        sharey=True,
        sharex=True,
        figsize=(5 * num_cols, 5 * num_rows),
    )

    plot_args = _get_plot_settings(session_year, use_absolute_compounds)

    if len(driver_selection) == 1:
        axes = np.array([axes])

    if y_axis in {"PctFromLapRep", "DeltaToLapRep"}:
        driver_laps_data = driver_laps_data[driver_laps_data["PctFromLapRep"] > -5]

    for index, driver in enumerate(driver_selection):
        row, col = divmod(index, max_columns)

        ax = axes[row][col] if num_rows > 1 else axes[col]

        single_driver_laps = driver_laps_data[driver_laps_data["Driver"] == driver]
        pit_in_laps_nums = single_driver_laps[single_driver_laps["PitInTime"].notna()]["LapNumber"].to_numpy()

        single_driver_laps = single_driver_laps[single_driver_laps["PctFromFastest"] < percentage_cutoff]

        if single_driver_laps.shape[0] < 5:
            jaskirat_logger.warning("%s HAS LESS THAN 5 LAPS ON RECORD FOR THIS EVENT", driver)

        sns.scatterplot(
            data=single_driver_laps,
            x="LapNumber",
            y=y_axis,
            ax=ax,
            hue=plot_args[0],
            palette=plot_args[1],
            hue_order=plot_args[3],
            style="FreshTyre",
            style_order=["True", "False", "Unknown"],
            markers=VISUAL_CONFIG["fresh"]["markers"],
            legend="auto" if index == num_cols - 1 else False,
        )
        ax.vlines(
            ymin=plt.yticks()[0][1],
            ymax=plt.yticks()[0][-2],
            x=pit_in_laps_nums,
            label="Pitstop",
            linestyle="dashed",
        )

        driver_color = get_driver_display_color(driver)
        title_font_params["color"] = driver_color
        ax.set_title(label=driver, fontdict=title_font_params, fontsize=12)

        ax.grid(color=driver_color, which="both", axis="both")
        sns.despine(left=True, bottom=True)

    fig.suptitle(t=f"{session_year} {event_title}", fontsize=20)
    axes.flatten()[num_cols - 1].legend(loc="best", fontsize=8, framealpha=0.5)

    return fig

def visualize_driver_lineplot(
    session_year: int,
    event_identifier: int | str,
    session_type: str = "R",
    driver_selection: Optional[Iterable[str | int] | str | int] = None,
    y_axis: str = "Position",
    percentage_cutoff: Optional[int | float] = None,
    grid_style: Optional[Literal["both", "x", "y"]] = None,
    laps_segment: Optional[list[int]] = None,
) -> Figure:
    plt.style.use("dark_background")

    if not isinstance(driver_selection, (int, str)) and driver_selection is not None:
        driver_selection = tuple(driver_selection)

    event_round_number, event_title, driver_selection = get_race_session_info(session_year, event_identifier, session_type, driver_selection)
    all_laps_data = JASKIRAT_DF_DICT[session_year][session_type]
    driver_laps_data = all_laps_data[
        (all_laps_data["RoundNumber"] == event_round_number) & (all_laps_data["Driver"].isin(driver_selection))
    ]

    if laps_segment is not None:
        assert sorted(laps_segment) == list(range(laps_segment[0], laps_segment[-1] + 1))
        driver_laps_data = driver_laps_data[driver_laps_data["LapNumber"].isin(laps_segment)]

    sc_laps_nums, vsc_laps_nums = _find_safety_car_periods(driver_laps_data)

    if percentage_cutoff is None:
        percentage_cutoff = 100 if y_axis == "Position" or y_axis.startswith("GapTo") else 10

    driver_laps_data = driver_laps_data[
        (driver_laps_data["RoundNumber"] == event_round_number)
        & (driver_laps_data["Driver"].isin(driver_selection))
        & (driver_laps_data["PctFromFastest"] < percentage_cutoff)
    ]

    num_laps = driver_laps_data["LapNumber"].nunique()
    fig, ax = plt.subplots(figsize=(ceil(num_laps * 0.25), 8))

    if y_axis == "Position":
        plt.yticks(range(2, 21, 2))

    if y_axis == "Position" or y_axis.startswith("GapTo"):
        ax.invert_yaxis()

    if len(driver_selection) > 10:
        ax.grid(which="major", axis="x")
    else:
        ax.grid(which="major", axis="both")

    for driver in driver_selection:
        single_driver_laps = driver_laps_data[driver_laps_data["Driver"] == driver]

        if single_driver_laps[y_axis].count() == 0:
            jaskirat_logger.warning("%s has no data entry for %s", driver, y_axis)
            continue

        driver_color = get_driver_display_color(driver)

        sns.lineplot(single_driver_laps, x="LapNumber", y=y_axis, ax=ax, color=driver_color, errorbar=None)
        last_lap = single_driver_laps["LapNumber"].max()
        last_pos = single_driver_laps[y_axis][single_driver_laps["LapNumber"] == last_lap].iloc[0]

        ax.annotate(
            xy=(last_lap + 1, last_pos + 0.25),
            text=driver,
            color=driver_color,
            fontsize=12,
        )
        sns.despine(left=True, bottom=True)

    _add_shaded_sc_periods(sc_laps_nums, vsc_laps_nums)

    if grid_style in {"both", "x", "y"}:
        plt.grid(axis=grid_style)
    else:
        plt.grid(visible=False)

    plt.legend(loc="lower right", fontsize=10)

    fig.suptitle(t=f"{session_year} {event_title}", fontsize=20)

    return fig

def visualize_driver_distributions(
    session_year: int,
    event_identifier: int | str,
    session_type: str = "R",
    driver_selection: Optional[Iterable[str | int] | str | int] = None,
    y_axis: str = "LapTime",
    percentage_cutoff: float | int = 10,
    show_swarmplot: bool = True,
    use_violinplot: bool = True,
    use_absolute_compounds: bool = False,
    use_teammate_ordering: bool = False,
) -> Figure:
    plt.style.use("dark_background")

    if not isinstance(driver_selection, (int, str)) and driver_selection is not None:
        driver_selection = tuple(driver_selection)

    event_round_number, event_title, driver_selection = get_race_session_info(
        session_year, event_identifier, session_type, driver_selection, use_teammate_ordering
    )

    all_laps_data = JASKIRAT_DF_DICT[session_year][session_type]
    valid_laps_data = all_laps_data[
        (all_laps_data["RoundNumber"] == event_round_number)
        & (all_laps_data["Driver"].isin(driver_selection))
        & (all_laps_data["PctFromFastest"] < percentage_cutoff)
    ]

    if use_teammate_ordering:
        driver_selection = reorder_drivers_by_teammate_metric_gap(valid_laps_data, driver_selection, y_axis)

    fig, ax = plt.subplots(figsize=(len(driver_selection) * 1.5, 10))
    plot_args = _get_plot_settings(session_year, use_absolute_compounds)

    driver_colors = [get_driver_display_color(driver) for driver in driver_selection]

    if use_violinplot:
        sns.violinplot(
            data=valid_laps_data,
            x="Driver",
            y=y_axis,
            inner=None,
            scale="area",
            palette=driver_colors,
            order=driver_selection,
        )
    else:
        sns.boxplot(
            data=valid_laps_data,
            x="Driver",
            y=y_axis,
            palette=driver_colors,
            order=driver_selection,
            whiskerprops={"color": "white"},
            boxprops={"edgecolor": "white"},
            medianprops={"color": "white"},
            capprops={"color": "white"},
            showfliers=False,
        )

    if show_swarmplot:
        sns.swarmplot(
            data=valid_laps_data,
            x="Driver",
            y=y_axis,
            hue=plot_args[0],
            palette=plot_args[1],
            order=driver_selection,
            linewidth=0,
            size=5,
        )

        handles, labels = ax.get_legend_handles_labels()
        order = _find_compound_display_order(labels)
        ax.legend(
            handles=[handles[idx] for idx in order],
            labels=[labels[idx] for idx in order],
            loc="best",
            title=plot_args[0],
            frameon=True,
            fontsize=10,
            framealpha=0.5,
        )

    ax.grid(visible=False)

    fig.suptitle(t=f"{session_year} {event_title}", fontsize=20)

    return fig

def visualize_strategy_summary(
    session_year: int,
    event_identifier: int | str,
    session_type: str = "R",
    driver_selection: Optional[Iterable[str | int]] = None,
    use_absolute_compounds: bool = False,
) -> Figure:
    if not isinstance(driver_selection, int) and driver_selection is not None:
        driver_selection = tuple(driver_selection)

    event_round_number, event_title, driver_selection = get_race_session_info(session_year, event_identifier, session_type, driver_selection)
    all_laps_data = JASKIRAT_DF_DICT[session_year][session_type]
    valid_laps_data = all_laps_data[
        (all_laps_data["RoundNumber"] == event_round_number) & (all_laps_data["Driver"].isin(driver_selection))
    ]

    fig, ax = plt.subplots(figsize=(5, len(driver_selection) // 3 + 1))
    plt.style.use("dark_background")

    driver_stints = (
        valid_laps_data[["Driver", "Stint", "Compound", "CompoundName", "FreshTyre", "LapNumber"]]
        .groupby(["Driver", "Stint", "Compound", "CompoundName", "FreshTyre"])
        .count()
        .reset_index()
    )
    driver_stints = driver_stints.rename(columns={"LapNumber": "StintLength"})
    driver_stints = driver_stints.sort_values(by=["Stint"])

    plot_args = _get_plot_settings(session_year, use_absolute_compounds)

    for driver in driver_selection:
        stints = driver_stints.loc[driver_stints["Driver"] == driver]

        previous_stint_end = 0
        for _, stint in stints.iterrows():
            plt.barh(
                [driver],
                stint["StintLength"],
                left=previous_stint_end,
                color=plot_args[1][stint[plot_args[0]]],
                edgecolor="black",
                fill=True,
                hatch=VISUAL_CONFIG["fresh"]["hatch"][stint["FreshTyre"]],
            )

            previous_stint_end += stint["StintLength"]

    _add_shaded_sc_periods(*_find_safety_car_periods(valid_laps_data))

    plt.title(f"{session_year} {event_title}", fontsize=16)
    plt.xlabel("Lap Number")
    plt.grid(False)

    handles, labels = ax.get_legend_handles_labels()
    if labels:
        deduplicated_labels_handles = dict(zip(labels, handles))
        plt.legend(
            handles=deduplicated_labels_handles.values(),
            labels=deduplicated_labels_handles.keys(),
            loc="lower right",
            fontsize=10,
        )

    ax.invert_yaxis()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    return fig


def visualize_compound_performance(
    session_years: int | Iterable[int],
    event_identifiers: int | str | Iterable[int | str],
    session_types: Optional[str | Iterable[str]] = None,
    y_axis: str = "LapTime",
    compound_selections: Iterable[str] = ["SOFT", "MEDIUM", "HARD"],
    x_axis: str = "TyreLife",
    percentage_cutoff: int | float = 10,
    use_absolute_compounds: bool = True,
) -> Figure:
    plt.style.use("dark_background")

    if isinstance(session_years, int):
        session_years = [session_years]

    event_data_objs, laps_data_list = _validate_and_prepare_inputs(
        session_years, event_identifiers, session_types, y_axis, compound_selections, x_axis, percentage_cutoff, use_absolute_compounds
    )

    fig, axes = plt.subplots(
        nrows=len(event_data
