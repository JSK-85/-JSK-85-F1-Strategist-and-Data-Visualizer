"""Make up-to-date visualizations for README."""
import logging
import shutil
import warnings
from pathlib import Path
import click
import fastf1 as f
import fastf1.plotting as p
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import f1_visualization.visualization as viz
from f1_visualization._consts import CURRENT_SEASON, NUM_ROUNDS, ROOT_PATH, SPRINT_ROUNDS
from f1_visualization.preprocess import get_last_round
logging.basicConfig(level=logging.INFO, format="%(levelname)s\t%(filename)s\t%(message)s")
logger = logging.getLogger(__name__)
DOC_VISUALS_PATH = ROOT_PATH / "Docs" / "visuals"
mpl.use("Agg")
sns.set_theme(rc={"figure.dpi": 300, "savefig.dpi": 300})
plt.style.use("dark_background")
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore")
def process_event_round(season: int, round_num: int, is_grand_prix: bool) -> int:
    if season > CURRENT_SEASON:
        raise ValueError(f"The latest season is {CURRENT_SEASON}.")
    if season < 2018:
        raise ValueError("Only 2018 and later seasons are available.")
    if round_num < 1 and round_num != -1:
        raise ValueError("Round number must be positive.")
    if season == CURRENT_SEASON:
        last_round_num = get_last_round(session_cutoff=5 if is_grand_prix else 3)
        if last_round_num == 0:
            raise ValueError(f"No session of the requested type in the {season} season yet.")
        if is_grand_prix:
            return min(round_num, last_round_num)
        if last_round_num <= round_num:
            return last_round_num
    if round_num > NUM_ROUNDS[season]:
        raise ValueError(f"{season} season only has {NUM_ROUNDS[season]} rounds.")
    if round_num == -1:
        round_num = NUM_ROUNDS[season]
    if is_grand_prix:
        return round_num
    try:
        round_num = max(
            sprint for sprint in SPRINT_ROUNDS[season] if sprint <= round_num
        )
    except KeyError as exc:
        raise ValueError(f"{season} season doesn't have any sprint race.") from exc
    except ValueError as exc:
        raise ValueError(
            f"{season} season only has these sprint races: {sorted(SPRINT_ROUNDS[season])}."
        ) from exc
    return round_num
@click.command()
@click.argument("season", nargs=1, default=CURRENT_SEASON, type=int)
@click.argument("round_number", nargs=1, default=-1, type=int)
@click.option(
    "--grand-prix/--sprint-race", "-g/-s", default=True, help="Default to grand prix."
)
@click.option("--update-readme", is_flag=True)
@click.option(
    "-r", "--reddit-machine", is_flag=True, help="Write plotted session name to text file."
)
def main(
    season: int, round_number: int, grand_prix: bool, update_readme: bool, reddit_machine: bool
):
    global DOC_VISUALS_PATH
    round_number = process_event_round(season, round_number, grand_prix)
    session_type = "R" if grand_prix else "S"
    session_data = f.get_session(season, round_number, session_type)
    session_data.load(telemetry=False, weather=False, messages=False)
    event_title = f"{session_data.event['EventName']} - {session_data.name}"
    destination_path = ROOT_PATH / "Visualizations" / f"{season}" / f"{event_title}"
    if destination_path.is_dir():
        if update_readme:
            shutil.copytree(destination_path, DOC_VISUALS_PATH, dirs_exist_ok=True)
            logger.info("Copied visualizations from %s to %s", destination_path, DOC_VISUALS_PATH)
            return
        overwrite_confirmation = input(
            (
                "WARNING:\n"
                f"{destination_path} may already contain the desired visualizations.\n"
                "Enter Y if you wish to overwrite them, otherwise, enter N: "
            )
        )
        if overwrite_confirmation.upper() != "Y":
            logger.info("Overwriting permission not given, aborting.")
            return
    else:
        Path.mkdir(destination_path, parents=True, exist_ok=True)
    logger.info("Visualizing %s", session_data)
    logger.info("Creating podium gap graph...")
    podium_finishers_list = viz.get_drivers(session_data, drivers=3)
    race_winner_driver = podium_finishers_list[0]
    viz.add_gap(race_winner_driver, modify_global=True, season=season, session_type=session_type)
    viz.driver_stats_lineplot(
        season=season,
        event=round_number,
        session_type=session_type,
        drivers=podium_finishers_list,
        y=f"GapTo{race_winner_driver}",
        grid="both",
    )
    plt.tight_layout()
    plt.savefig(destination_path / "podium_gap.png")
    logger.info("Creating lap time graph...")
    viz.driver_stats_scatterplot(
        season=season,
        event=round_number,
        session_type=session_type,
        drivers=10
    )
    plt.tight_layout()
    plt.savefig(destination_path / "laptime.png")
    logger.info("Creating strategy graph...")
    viz.strategy_barplot(
        season=season,
        event=round_number,
        session_type=session_type,
    )
    plt.tight_layout()
    plt.savefig(destination_path / "strategy.png")
    logger.info("Creating position change graph...")
    viz.driver_stats_lineplot(
        season=season,
        event=round_number,
        session_type=session_type,
    )
    plt.tight_layout()
    plt.savefig(destination_path / "position.png")
    logger.info("Creating teammate comparison boxplot...")
    viz.driver_stats_distplot(
        season=season,
        event=round_number,
        session_type=session_type,
        violin=False,
        swarm=False,
        teammate_comp=True,
    )
    plt.tight_layout()
    plt.savefig(destination_path / "teammate_box.png")
    logger.info("Creating teammate comp violin plot...")
    viz.driver_stats_distplot(
        season=season,
        event=round_number,
        session_type=session_type,
        teammate_comp=True,
        upper_bound=7,
    )
    plt.tight_layout()
    plt.savefig(destination_path / "teammate_violin.png")
    logger.info("Creating driver pace plot...")
    viz.driver_stats_distplot(
        season=season,
        event=round_number,
        session_type=session_type,
        upper_bound=7,
    )
    plt.tight_layout()
    plt.savefig(destination_path / "driver_pace.png")
    logger.info("Creating team pace comparison graph...")
if __name__ == "__main__":
   main()
