import logging
import os
from pathlib import Path

from codetiming import Timer
from knockknock import slack_sender
import typer

UNET_ARCHNAME = "u-net"
SIMPLE_CONV_ARCHNAME = "simple-conv"
EXPERIMENT_NAME = os.getenv("WANDB_EXPERIMENT_NAME")
TAGS = {
    UNET_ARCHNAME: ["baseline", UNET_ARCHNAME],
    SIMPLE_CONV_ARCHNAME: ["baseline", SIMPLE_CONV_ARCHNAME, "debug"],
}

app = typer.Typer()

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(filename)s - %(asctime)s - %(message)s",
)
logger = logging.getLogger()
logger.setLevel("INFO")


@app.command()
@Timer(name="train", text="{name}: {minutes:.1f} minutes", logger=logging.info)
@slack_sender(webhook_url=os.getenv("KK_SLACK_WH_URL"), channel="general")
def main(
    workdir: Path,
):

    os.makedirs(workdir, exist_ok=True)

    # Create directories for experimental logs
    sample_dir = os.path.join(workdir, "samples")
    os.makedirs(sample_dir, exist_ok=True)

    logging.info(f"Starting {os.path.basename(__file__)}")

    logging.info(f"Finished {os.path.basename(__file__)}")


if __name__ == "__main__":
    app()
