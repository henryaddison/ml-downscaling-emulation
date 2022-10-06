from contextlib import contextmanager
import logging
import os

from torch.utils.data import DataLoader, TensorDataset
import torch
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import xarray as xr
import yaml
from .dataset import CropT, Standardize, MinMax, ClipT, SqrtT, ComposeT, XRDataset

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

def get_variables(data_dirpath):
    with open(os.path.join(data_dirpath, 'ds-config.yml'), 'r') as f:
        ds_config = yaml.safe_load(f)
    variables = [ pred_meta["variable"] for pred_meta in ds_config["predictors"] ]
    target_variables = ["target_pr"]

    return variables, target_variables

def get_transform(data_dirpath):
    variables, target_variables = get_variables(data_dirpath)
    xr_data_train = xr.load_dataset(os.path.join(data_dirpath, 'train.nc'))

    transform = ComposeT([
        CropT(32),
        Standardize(variables),
        MinMax(variables)])
    target_transform = ComposeT([
        SqrtT(target_variables),
        ClipT(target_variables),
        MinMax(target_variables),
    ])
    xr_data_train = transform.fit_transform(xr_data_train)
    xr_data_train = target_transform.fit_transform(xr_data_train)

    return transform, target_transform, xr_data_train

def load_data(data_dirpath, batch_size, eval_split='val'):
    variables, target_variables = get_variables(data_dirpath)

    transform, target_transform, xr_data_train = get_transform(data_dirpath)

    xr_data_eval = xr.load_dataset(os.path.join(data_dirpath, f'{eval_split}.nc'))
    xr_data_eval = transform.transform(xr_data_eval)
    xr_data_eval = target_transform.transform(xr_data_eval)

    train_dataset = XRDataset(xr_data_train, variables)
    eval_dataset = XRDataset(xr_data_eval, variables)

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size)
    eval_data_loader = DataLoader(eval_dataset, batch_size=batch_size)

    return train_data_loader, eval_data_loader

def log_epoch(epoch, epoch_metrics, wandb_run, tb_writer):
    import mlflow
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
    import wandb
    import mlflow
    from torch.utils.tensorboard import SummaryWriter
    with wandb.init(project=experiment_name, name=run_name, tags=tags, config=config) as wandb_run:

        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_name) :
            mlflow.set_tags({ key: True for key in tags})
            mlflow.log_params(config)

            with SummaryWriter() as tb_writer:
                yield wandb_run, tb_writer
