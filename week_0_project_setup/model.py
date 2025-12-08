import torch 
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModel
from transformers import AutoModelForSequenceClassification
import torchmetrics

class ColaModel(pl.LightningModule):
    def __init__(self, model_name="google/bert_uncased_L-2_H-128_A-2", lr=1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.bert = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        )
        self.num_classes = 2

        self.train_accuracy_metric = torchmetrics.Accuracy(task="binary")
        self.val_accuracy_metric = torchmetrics.Accuracy(task="binary")
        self.f1_metric = torchmetrics.F1Score(task="binary", num_classes=self.num_classes)


    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        
        return outputs 

    def training_step(self, batch, batch_idx):
        outputs = self.forward(
            batch["input_ids"], batch["attention_mask"], labels=batch["label"]
        )
        preds = torch.argmax(outputs.logits, dim=1)
        train_acc = self.train_accuracy_metric(preds, batch["labels"])

        self.log("train/loss", outputs.loss, prog_bar=True, on_epoch=True)
        self.log("train/acc", train_acc, prog_bar=True, on_epoch=True)

        return outputs.loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(
            batch["input_ids"], batch["attention_mask"], labels=batch["label"]
        )
        preds = torch.argmax(outputs.logits, 1)

        val_acc = self.val_accuracy_metric(preds, batch["label"])
        f1 = self.f1_metric(preds, batch["label"])

        self.log("val/loss", outputs.loss, prog_bar=True, on_step=True)
        self.log("val/acc", val_acc, prog_bar=True, on_epoch=True)
        self.log("val/f1", f1, prog_bar=True, on_epoch=True)

        return {"labels": batch["label"], "logits": outputs.logits}
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])
    
    

        
    



        