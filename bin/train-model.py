import logging
import os
from pathlib import Path

from codetiming import Timer
from knockknock import slack_sender
import numpy as np
import torch
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import typer
import yaml

from ml_downscaling_emulator.unet import unet
from ml_downscaling_emulator.training import log_epoch, track_run, save_checkpoint
from ml_downscaling_emulator.training.dataset import get_dataset

UNET_ARCHNAME = "u-net"
SIMPLE_CONV_ARCHNAME = "simple-conv"
EXPERIMENT_NAME="ml-downscaling-emulator"
TAGS = {
    UNET_ARCHNAME: ["baseline", UNET_ARCHNAME],
    SIMPLE_CONV_ARCHNAME: ["baseline", SIMPLE_CONV_ARCHNAME, "debug"]
}

app = typer.Typer()

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
logger = logging.getLogger()
logger.setLevel('INFO')

@app.command()
@Timer(name="train", text="{name}: {minutes:.1f} minutes", logger=logging.info)
@slack_sender(webhook_url=os.getenv("KK_SLACK_WH_URL"), channel="general")
def main(
        workdir: Path,
        dataset: str = typer.Option(...),
        epochs: int = 200,
        learning_rate: float = 2e-4,
        batch_size: int = 64,
        snapshot_freq: int = 25,
        input_transform_key: str = "v1",
        target_transform_key: str = "v1",
    ):

    run_config = dict(
        dataset = dataset,
        input_transform_key = input_transform_key,
        target_transform_key = target_transform_key,
        batch_size = batch_size,
        epochs=epochs,
        architecture = "u-net",
        loss = "MSELoss",
        optimizer = "Adam",
        device = ('cuda' if torch.cuda.is_available() else 'cpu'),
    )

    run_name = workdir.name

    os.makedirs(workdir, exist_ok=True)

    gfile_stream = open(os.path.join(workdir, 'stdout.txt'), 'w')
    handler = logging.StreamHandler(gfile_stream)
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Create transform saving directory
    transform_dir = os.path.join(workdir, "transforms")
    os.makedirs(transform_dir, exist_ok=True)

    # Create directories for experimental logs
    sample_dir = os.path.join(workdir, "samples")
    os.makedirs(sample_dir, exist_ok=True)

    tb_dir = os.path.join(workdir, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)

    logging.info(f"Starting {os.path.basename(__file__)}")

    # Create checkpoints directory
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Intermediate checkpoints to resume training after pre-emption in cloud environments
    checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
    os.makedirs(os.path.dirname(checkpoint_meta_dir), exist_ok=True)

    device = torch.device(run_config["device"])
    logging.info(f'Using device {device}')


    # Build dataloaders
    train_dl, _, _ = get_dataset(dataset, dataset, input_transform_key, target_transform_key, transform_dir, batch_size=batch_size, split="train", evaluation=False)
    val_dl, _, _ = get_dataset(dataset, dataset, input_transform_key, target_transform_key, transform_dir, batch_size=batch_size, split="val", evaluation=False)

    # Setup model, loss and optimiser
    num_predictors, _, _ = train_dl.dataset[0][0].shape
    model = unet.UNet(num_predictors, 1).to(device=device)
    if run_config["loss"] == "MSELoss":
        criterion = torch.nn.MSELoss().to(device)
    else:
        raise NotImplementedError(
            f'Loss {run_config["loss"]} not supported yet!')

    if run_config["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        raise NotImplementedError(
            f'Optimizer {run_config["optimizer"]} not supported yet!')

    state = dict(optimizer=optimizer, model=model, step=0, epoch=0)

    initial_epoch = int(state['epoch'])
    step = state["step"]

    def loss_fn(model, batch, cond):
        return criterion(model(cond), batch)

    def optimize_fn(optimizer, params, step, lr, warmup=5000, grad_clip=1.):
        """Optimizes with warmup and gradient clipping (disabled if negative)."""
        if warmup > 0:
            for g in optimizer.param_groups:
                g['lr'] = lr * np.minimum(step / warmup, 1.0)
        if grad_clip >= 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
        optimizer.step()

    # Compute validation loss
    def eval_step_fn(state, batch, cond):
        """Running one step of training or evaluation.

        This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
        for faster execution.

        Args:
        state: A dictionary of training information, containing the score model, optimizer,
        EMA status, and number of optimization steps.
        batch: A mini-batch of training/evaluation data to model.
        cond: A mini-batch of conditioning inputs.

        Returns:
        loss: The average loss value of this state.
        """
        model = state['model']
        with torch.no_grad():
            loss = loss_fn(model, batch, cond)

        return loss

    def train_step_fn(state, batch, cond):
        """Running one step of training or evaluation.

        This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
        for faster execution.

        Args:
        state: A dictionary of training information, containing the score model, optimizer,
        EMA status, and number of optimization steps.
        batch: A mini-batch of training/evaluation data to model.
        cond: A mini-batch of conditioning inputs.

        Returns:
        loss: The average loss value of this state.
        """
        model = state['model']
        optimizer = state['optimizer']
        optimizer.zero_grad()
        loss = loss_fn(model, batch, cond)
        loss.backward()
        optimize_fn(optimizer, model.parameters(), step=state['step'], lr=learning_rate)
        state['step'] += 1

        return loss

    # save the config
    config_path = os.path.join(workdir, "config.yml")
    with open(config_path, 'w') as f:
        yaml.dump(run_config, f)

    with track_run(EXPERIMENT_NAME, run_name, run_config, TAGS[run_config['architecture']], tb_dir) as (wandb_run, tb_writer):
        # Fit model
        wandb_run.watch(model, criterion=criterion, log_freq=100)

        logging.info("Starting training loop at epoch %d." % (initial_epoch,))

        for epoch in range(initial_epoch, epochs+1):
            # Update model based on training data
            model.train()

            train_set_loss = 0.0
            with logging_redirect_tqdm():
                with tqdm(total=len(train_dl.dataset), desc=f'Epoch {epoch}', unit=' timesteps') as pbar:
                    for (cond_batch, x_batch) in train_dl:
                        cond_batch = cond_batch.to(device)
                        x_batch = x_batch.to(device)
                        ###################
                        # CURRENT VERSION #
                        ###################
                        # # Compute prediction and loss
                        # outputs_tensor = model(cond_batch)
                        # train_batch_loss = criterion(outputs_tensor, x_batch)
                        # train_set_loss += train_batch_loss.item()

                        # # Backpropagation
                        # optimizer.zero_grad()
                        # train_batch_loss.backward()
                        # optimizer.step()

                        #####################
                        # SCORE_SDE VERSION #
                        #####################
                        train_batch_loss = train_step_fn(state, x_batch, cond_batch)
                        train_set_loss += train_batch_loss.item()

                        #######
                        # END #
                        #######

                        # Log progress so far on epoch
                        pbar.update(cond_batch.shape[0])

                        step += 1
            train_set_loss = train_set_loss/len(train_dl)

            model.eval()
            val_set_loss = 0.0
            for val_cond_batch, val_x_batch in val_dl:
                # eval_cond_batch, eval_x_batch = next(iter(eval_ds))
                val_x_batch = val_x_batch.to(device)
                val_cond_batch = val_cond_batch.to(device)
                # eval_batch = eval_batch.permute(0, 3, 1, 2)
                val_batch_loss = eval_step_fn(state, val_x_batch, val_cond_batch)

                # Progress
                val_set_loss += val_batch_loss.item()
                val_set_loss = val_set_loss/len(val_dl)

            epoch_metrics = {"train/loss": train_set_loss, "val/loss": val_set_loss}
            log_epoch(epoch, epoch_metrics, wandb_run, tb_writer)
            # Checkpoint model
            if (epoch != 0 and epoch % snapshot_freq == 0) or epoch == epochs:
                checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch}.pth')
                save_checkpoint(checkpoint_path, state)
                logging.info(f"epoch: {epoch}, checkpoint saved to {checkpoint_path}")

    logging.info(f"Finished {os.path.basename(__file__)}")

if __name__ == "__main__":
    app()
