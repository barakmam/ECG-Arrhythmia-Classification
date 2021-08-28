import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb

dropout = 0.1


class PaperNet(pl.LightningModule):
    def __init__(self, input_shape, num_classes, device, weight_decay=1.5e-06):
        super().__init__()

        self.dropout = dropout

        # log hyperparameters
        self.save_hyperparameters()
        self.automatic_optimization = True
        self.weight_decay = weight_decay

        self.features = nn.Sequential(
            #first layer
            nn.Conv2d(1, 8, 4),
            nn.ReLU(),
            nn.MaxPool2d(2),
            #second layer
            nn.Conv2d(8, 13, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            #third layer
            nn.Conv2d(13, 13, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(11700, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes),
            nn.Softmax(-1)
        ).to(device)


    def forward(self, x):
        x = self.features(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        loss = F.nll_loss(F.log_softmax(logits), y)  # , weight=self.loss_weights)

        # training metrics
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True)

        return loss  # {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        loss = F.nll_loss(F.log_softmax(logits), y)  # , weight=self.loss_weights)
        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss  # {'loss': loss}

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(F.log_softmax(logits), y)  # weight=self.loss_weights)

        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        return loss  # {'loss': loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), weight_decay=self.weight_decay)
        return optimizer



class ImagePredictionLogger(Callback):
    def __init__(self, val_samples, classes_names, num_samples=20):
        super().__init__()
        self.num_samples = num_samples
        self.val_imgs, self.val_labels = val_samples
        self.classes_names = classes_names

    def on_validation_epoch_end(self, trainer, pl_module):
        # Bring the tensors to CPU
        val_imgs = self.val_imgs.to(device=pl_module.device)
        val_labels = self.classes_names[self.val_labels]  # .to(device=pl_module.device)
        # Get model prediction
        logits = pl_module(val_imgs)
        preds = self.classes_names[torch.argmax(logits, -1).cpu()]
        # Log the images as wandb Image
        trainer.logger.experiment.log({
            "examples": [wandb.Image(x, caption=f"Pred:{pred}, Label:{y}")
                         for x, pred, y in zip(val_imgs[:self.num_samples],
                                               preds[:self.num_samples],
                                               val_labels[:self.num_samples])]
        })
