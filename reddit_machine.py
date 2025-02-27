import logging
import shutil
import time
import praw
from f1_visualization._consts import ROOT_PATH
logging.basicConfig(level=logging.INFO, format="%(levelname)s\t%(filename)s\t%(message)s")
logger = logging.getLogger(__name__)
IMAGES_PATH = ROOT_PATH / "Docs" / "visuals"
def main():
    reddit_client = praw.Reddit("armchair-strategist")
    subreddit_formula1 = reddit_client.subreddit("formula1")
    subreddit_f1technical = reddit_client.subreddit("f1technical")
    formula1_flairs = subreddit_formula1.flair.link_templates.user_selectable()
    f1technical_flairs = subreddit_f1technical.flair.link_templates.user_selectable()
    formula1_flair_id = next(
        flair for flair in formula1_flairs if "Statistics" in flair["flair_text"]
    )["flair_template_id"]
    f1technical_flair_id = next(
        flair for flair in f1technical_flairs if "Strategy" in flair["flair_text"]
    )["flair_template_id"]
    with open(ROOT_PATH / "tmp" / "event_name.txt", "r") as fin:
        event_title = fin.read().strip()
    dashboard_link = "Check out more at armchair-strategist.dev!"
    images_list = [
        {
            "image_path": IMAGES_PATH / "strategy.png",
            "caption": (
                "Tyre strategy recap. Stripped bar sections represent used tyre stints. "
                f"{dashboard_link}"
            ),
        },
        {
            "image_path": IMAGES_PATH / "podium_gap.png",
            "caption": f"Podium finishers' gaps to winners. {dashboard_link}",
        },
        {
            "image_path": IMAGES_PATH / "position.png",
            "caption": f"Race position history. {dashboard_link}",
        },
        {
            "image_path": IMAGES_PATH / "laptime.png",
            "caption": (
                "Point finishers' lap times. White vertical bars represent pitstops. "
                f"{dashboard_link}"
            ),
        },
        {
            "image_path": IMAGES_PATH / "team_pace.png",
            "caption": f"Team pace ranking. {dashboard_link}",
        },
        {
            "image_path": IMAGES_PATH / "teammate_violin.png",
            "caption": (
                "Driver pace ranking (teammates vs teammates). Largest gap on the left. "
                f"{dashboard_link}"
            ),
        },
        {
            "image_path": IMAGES_PATH / "driver_pace.png",
            "caption": (
                "Driver pace ranking (finishing order). Highest finisher on the left. "
                f"{dashboard_link}"
            ),
        },
    ]
    post_formula1 = subreddit_formula1.submit_gallery(
        title=f"{event_title} Strategy & Performance Recap",
        images=images_list,
        flair_id=formula1_flair_id,
    )
    post_formula1.reply(
        (
            "What other graphics do you want to see and "
            "how can these existing graphics be improved?"
        )
    )
    logger.info("Finished posting to r/formula1")
    time.sleep(5)
    post_f1technical = subreddit_f1technical.submit_gallery(
        title=f"{event_title} Strategy & Performance Recap",
        images=images_list,
        flair_id=f1technical_flair_id,
    )
    post_f1technical.reply(
        (
            "Check out the interactive version of these graphics and more "
            "at my [strategy dashboard](https://armchair-strategist.dev/)"
            "\n\n"
            "Please let me know if you have suggestions for improving these graphics "
            "or ideas for other graphics!"
        )
    )
    logger.info("Finished posting to r/f1technical")
    shutil.rmtree(ROOT_PATH / "tmp")
if __name__ == "__main__":
    main()
