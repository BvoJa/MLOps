import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from pytorch_lightning.loggers import WandbLogger

from data import DataModule
from model import ColaModel

def main():
    cola_data = DataModule()
    cola_model = ColaModel()

    checkpoint_callback = ModelCheckpoint(
        dirpath="./models", monitor="val/loss", mode="min"
    )
    early_stopping_callback = EarlyStopping(
        monitor="val/loss", patience=3, verbose=True, mode="min"
    )

    wandb_logger = WandbLogger(project="MLOps_Basics")
    trainer = pl.Trainer(
        default_root_dir="logs",
        accelerator="gpu",
        devices=2,
        strategy="ddp",
        max_epochs=20,
        fast_dev_run=False,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
    )
    trainer.fit(cola_model, cola_data)

if __name__ == "__main__":
    main()