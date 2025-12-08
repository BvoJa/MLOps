import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from pytorch_lightning.loggers import WandbLogger
import pandas as pd

from data import DataModule
from model import ColaModel

class SamplesVisualizationLogger(pl.Callback):
    def __init__(self, datamodule):
        super().__init__()

        self.datamodule = datamodule

    def on_validation_end(self, trainer, pl_module):
        val_batch = next(iter(self.datamodule.val_dataloader()))
        sentences = val_batch["sentence"]

        outputs = pl_module(val_batch["input_ids"], val_batch["attention_mask"])
        preds = torch.argmax(outputs.logits, 1)
        labels = val_batch["label"]

        df = pd.DataFrame(
            {"Sentence": sentences, "Label": labels.numpy(), "Predicted": preds.numpy()}
        )

        wrong_df = df[df["Label"] != df["Predicted"]]
        trainer.logger.experiment.log(
            {
                "examples": wandb.Table(dataframe=wrong_df, allow_mixed_types=True),
                "global_step": trainer.global_step,
            }
        )

def main():
    cola_data = DataModule()
    cola_model = ColaModel()

    checkpoint_callback = ModelCheckpoint(
        dirpath="./models", 
        filename="best-checkpoint.ckpt",
        monitor="val/loss", 
        mode="min"
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
        callbacks=[checkpoint_callback, SamplesVisualizationLogger(cola_data) , early_stopping_callback],
    )
    trainer.fit(cola_model, cola_data)

if __name__ == "__main__":
    main()