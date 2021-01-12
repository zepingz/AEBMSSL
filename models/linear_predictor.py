import copy
import numpy as np

import torch
from torch import nn
import torch.optim as optim
import pytorch_lightning as pl

from models.test_model import TestModel
from models.components.layers import ConvBlock


class LinearPredictor(nn.Module):
    def __init__(self, in_channels, out_channels, encoder):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.encoder = encoder.layers[:4]

        self.conv_layers = nn.Sequential(
            # ConvBlock(1, 64, 7, 2, 3),
            # ConvBlock(64, 64, 5, 2, 2),
            # ConvBlock(64, 64, 5, 2, 2),
            # ConvBlock(64, 64, 5, 2, 2),
            ConvBlock(64, 64, 5, 2, 2),
            ConvBlock(64, 64, 5, 2, 2),
        )
        self.fc = nn.Linear(self.in_channels, self.out_channels)

        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.encoder(x)
        x = self.conv_layers(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


class Predictor(pl.LightningModule):
    def __init__(self, data_module, weight_path, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.data_module = data_module

        encoder = TestModel.load_from_checkpoint(weight_path).frame_encoder
        self.predictor = LinearPredictor(64, 10, copy.deepcopy(encoder))

    def forward(self, batch):
        imgs = batch["target_frame"]
        lbls = batch["labels"].flatten()
        pred = self.predictor(imgs)
        return pred

    def configure_optimizers(self):
        model_optimizer = optim.Adam(
            self.parameters(),
            self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        # lambda1 = lambda epoch: 0.95 ** epoch
        # scheduler = optim.lr_scheduler.LambdaLR(model_optimizer, lr_lambda=lambda1)
        # scheduler = optim.lr_scheduler.StepLR(model_optimizer, step_size=5, gamma=0.1)
        scheduler = optim.lr_scheduler.MultiStepLR(model_optimizer, milestones=[5, 80], gamma=0.1)

        return [model_optimizer] , [scheduler]

    def train_dataloader(self):
        return self.data_module.linpred_dataloader(self.hparams.batch_size)

    def val_dataloader(self):
        return self.data_module.val_dataloader(self.hparams.batch_size)

    def xent_loss(self, predicted, target):
        return nn.CrossEntropyLoss()(predicted, target)

    def training_step(self, batch, batch_idx):
        pred = self.forward(batch)
        lbls = batch['labels'].flatten()
        loss = self.xent_loss(pred, lbls)

        if self.logger is not None:
            self.logger.log_metrics({"train_loss": loss.item()})

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        pred = self.forward(batch)
        lbls = batch['labels'].flatten()
        loss = self.xent_loss(pred, lbls)

        correct = torch.sum(torch.argmax(pred, dim=1) == lbls).item()
        result_dic = {"loss": loss, "count": len(lbls), "correct": correct}
        return result_dic

    def validation_epoch_end(self, outputs):
        if self.current_epoch == 9:
            for param in self.predictor.encoder.parameters():
                param.requires_grad = True

        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        total = np.sum([x["count"] for x in outputs])
        correct = np.sum([x["correct"] for x in outputs])
        tensorboard_logs = {
            "val_loss": avg_loss,
            "acc": correct / total,
        }

        return {
            "val_loss": avg_loss,
            "log": tensorboard_logs,
            "progress_bar": tensorboard_logs,
        }
