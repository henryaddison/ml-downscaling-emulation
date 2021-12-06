from contextlib import contextmanager
import logging

from torch.utils.data import DataLoader, TensorDataset

import wandb
import mlflow
from torch.utils.tensorboard import SummaryWriter
import torch
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import xarray as xr
from ml_downscaling_emulator.training.dataset import XRDataset

def train(train_dl, val_dl, model, criterion, optimizer, epochs, device):
    for epoch in range(epochs):
        # Update model based on training data
        epoch_train_loss = train_epoch(train_dl, model, criterion, optimizer, epoch, device)

        # Compute validation loss
        epoch_val_loss = val_epoch(val_dl, model, criterion, epoch, device)

        epoch_metrics = {"train/loss": epoch_train_loss, "val/loss": epoch_val_loss}
        yield epoch, epoch_metrics

def train_epoch(dataloader, model, criterion, optimizer, epoch, device):
    model.train()

    epoch_loss = 0.0
    with logging_redirect_tqdm():
        with tqdm(total=len(dataloader.dataset), desc=f'Epoch {epoch}', unit=' timesteps') as pbar:
            for (batch_X, batch_y) in dataloader:
                loss = train_on_batch(batch_X.to(device), batch_y.to(device), model, criterion, optimizer)
                epoch_loss += loss.item()

                # Log progress so far on epoch
                pbar.update(batch_X.shape[0])

    return epoch_loss

def val_epoch(dataloader, model, criterion, epoch, device):
    model.eval()

    epoch_val_loss = 0
    for batch_X, batch_y in dataloader:
        val_loss = val_on_batch(batch_X.to(device), batch_y.to(device), model, criterion)

        # Progress
        epoch_val_loss += val_loss.item()

    model.train()

    return epoch_val_loss

def train_on_batch(batch_X, batch_y, model, criterion, optimizer):
    # Compute prediction and loss
    outputs_tensor = model(batch_X)
    loss = criterion(outputs_tensor, batch_y)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss

def val_on_batch(batch_X, batch_y, model, criterion):
    with torch.no_grad():
        # Compute prediction and loss
        outputs_tensor = model(batch_X)
        loss = criterion(outputs_tensor, batch_y)

    return loss

def load_data(data_dirpath, batch_size):
    # train_set = XRDataset(xr.load_dataset(data_dirpath/'train.nc'), ['pr'])
    # val_set = XRDataset(xr.load_dataset(data_dirpath/'val.nc'), ['pr'])
    train_set = TensorDataset(torch.load(data_dirpath/'train_X.pt'), torch.load(data_dirpath/'train_y.pt'))
    val_set = TensorDataset(torch.load(data_dirpath/'val_X.pt'), torch.load(data_dirpath/'val_y.pt'))

    train_dl = DataLoader(train_set, batch_size=batch_size)
    val_dl = DataLoader(val_set, batch_size=batch_size)

    return train_dl, val_dl

def log_epoch(epoch, epoch_metrics, wandb_run, tb_writer):
    logging.info(f"Epoch {epoch}: Train Loss {epoch_metrics['train/loss']} Val Loss {epoch_metrics['val/loss']}")

    wandb_run.log(epoch_metrics)
    mlflow.log_metrics(epoch_metrics, step=epoch)
    for name, value in epoch_metrics.items():
        tb_writer.add_scalar(name, value, epoch)

def checkpoint_model(model, model_checkpoints_dir, epoch):
    model_checkpoint_path = model_checkpoints_dir / f"model-epoch{epoch}.pth"
    torch.save(model, model_checkpoint_path)
    logging.info(f"Epoch {epoch}: Saved model to {model_checkpoint_path}")

@contextmanager
def track_run(experiment_name, run_name, config, tags):
    with wandb.init(project=experiment_name, name=run_name, tags=tags, config=config) as wandb_run:

        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_name) :
            mlflow.set_tags({ key: True for key in tags})
            mlflow.log_params(config)

            with SummaryWriter() as tb_writer:
                yield wandb_run, tb_writer
