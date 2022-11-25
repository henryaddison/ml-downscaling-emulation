from contextlib import contextmanager
import logging
import os

import torch

def log_epoch(epoch, epoch_metrics, wandb_run, tb_writer):
    import mlflow
    logging.info(f"Epoch {epoch}: Train Loss {epoch_metrics['train/loss']} Val Loss {epoch_metrics['val/loss']}")

    wandb_run.log(epoch_metrics)
    mlflow.log_metrics(epoch_metrics, step=epoch)
    for name, value in epoch_metrics.items():
        tb_writer.add_scalar(name, value, epoch)

def restore_checkpoint(ckpt_dir, state, device):
  if not os.path.exists(ckpt_dir):
    os.makedirs(os.path.dirname(ckpt_dir), exist_ok=True)
    logging.warning(f"No checkpoint found at {ckpt_dir}. "
                    f"Returned the same state as input")
    return state
  else:
    loaded_state = torch.load(ckpt_dir, map_location=device)
    state['optimizer'].load_state_dict(loaded_state['optimizer'])
    state['model'].load_state_dict(loaded_state['model'], strict=False)
    state['step'] = loaded_state['step']
    state['epoch'] = loaded_state['epoch']
    return state

def save_checkpoint(ckpt_dir, state):
  saved_state = {
    'optimizer': state['optimizer'].state_dict(),
    'model': state['model'].state_dict(),
    'step': state['step'],
    'epoch': state['epoch']
  }
  torch.save(saved_state, ckpt_dir)

@contextmanager
def track_run(experiment_name, run_name, config, tags, tb_dir):
    import wandb
    import mlflow
    from torch.utils.tensorboard import SummaryWriter
    with wandb.init(project=experiment_name, name=run_name, tags=tags, config=config) as wandb_run:

        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_name) :
            mlflow.set_tags({ key: True for key in tags})
            mlflow.log_params(config)

            with SummaryWriter(tb_dir) as tb_writer:
                yield wandb_run, tb_writer
