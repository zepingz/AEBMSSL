from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import pytorch_lightning as pl

import models.components


class Merger(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 64)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class DeterminsticEBM(pl.LightningModule):
    def __init__(self, data_module, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.data_module = data_module

        # init model modules
        self.frame_encoder = models.components.encoder_dict[
            self.hparams.encoder_type
        ]()
        self.frame_decoder = models.components.decoder_dict[
            self.hparams.decoder_type
        ]()
        self.merger = Merger()

        self.lambdas = dict(
            target_prediction=self.hparams.lambda_target_prediction,
            decoding_error=self.hparams.lambda_decoding_error,
        )

    def forward(self, conditional_frames):
        bs, seq_len, c, h, w = conditional_frames.size()
        conditional_frames = conditional_frames.view(bs * seq_len, c, h, w)

        encoded_frames = self.frame_encoder(conditional_frames)
        decoded_frames = self.frame_decoder(encoded_frames)

        encoded_frames = encoded_frames.view(bs, seq_len, -1)
        merged = self.merger(encoded_frames)
        predicted_frame = self.frame_decoder(merged)

        decoded_frames = decoded_frames.view(bs, seq_len, c, h, w)

        return predicted_frame, decoded_frames

    def reconstruction_error(self, predicted, target):
        return F.mse_loss(predicted, target)

    def training_step(self, batch, batch_idx):
        conditional_frames = batch["conditional_frames"]
        target_frame = batch["target_frame"]

        predicted_frame, decoded_frames = self.forward(conditional_frames)

        energies = {}
        energies["target_prediction"] = self.reconstruction_error(
            predicted_frame, target_frame
        )
        energies["decoding_error"] = self.reconstruction_error(
            decoded_frames, conditional_frames
        )
        energies["total"] = sum(
            [energies[k] * self.lambdas[k] for k in energies]
        )

        if self.logger is not None:
            self.logger.log_metrics(energies)

        return dict(loss=energies["total"], progress_bar=energies)

    def validation_step(self, batch, batch_idx):
        conditional_frames = batch["conditional_frames"]
        target_frame = batch["target_frame"]

        predicted_frame, decoded_frames = self.forward(conditional_frames)

        energies = {}
        energies["target_prediction"] = self.reconstruction_error(
            predicted_frame, target_frame
        )
        energies["decoding_error"] = self.reconstruction_error(
            decoded_frames, conditional_frames
        )
        energies["total"] = sum(
            [energies[k] * self.lambdas[k] for k in energies]
        )

        result_dic = {"loss": energies["total"], "free_energies": energies}

        # Include images for logging
        if batch_idx == 0:
            result_dic["pred_target_imgs"] = (
                predicted_frame.detach().cpu().numpy()[:10]
            )
            result_dic["pred_imgs"] = (
                decoded_frames.detach().cpu().numpy()[:10, -1]
            )
            if self.current_epoch == 0:
                result_dic["target_imgs"] = (
                    target_frame.detach().cpu().numpy()[:10]
                )
                result_dic["imgs"] = (
                    conditional_frames.detach().cpu().numpy()[:10, -1]
                )

        return result_dic

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()

        key_list = ["target_prediction", "decoding_error"]
        tensorboard_logs = {
            "val_"
            + key: torch.stack([x["loss_dict"][key] for x in outputs]).mean()
            for key in key_list
        }
        tensorboard_logs["val_loss"] = avg_loss

        if self.logger is not None:
            # Log images
            if self.current_epoch == 0:
                target_imgs = outputs[0]["target_imgs"]
                for i in range(len(target_imgs)):
                    self.logger.experiment.log_image(
                        target_imgs[i, 0],
                        name=f"target_img{str(i+1).zfill(2)}_gt",
                    )

                imgs = outputs[0]["imgs"]
                for i in range(len(imgs)):
                    self.logger.experiment.log_image(
                        imgs[i, 0], name=f"img{str(i+1).zfill(2)}_gt"
                    )

            pred_target_imgs = outputs[0]["pred_target_imgs"]
            for i in range(len(pred_target_imgs)):
                img_name = (
                    f"target_img{str(i+1).zfill(2)}_pred_"
                    f"epoch{str(self.current_epoch+1).zfill(3)}"
                )
                self.logger.experiment.log_image(
                    pred_target_imgs[i, 0], name=img_name
                )

            pred_imgs = outputs[0]["pred_imgs"]
            for i in range(len(pred_imgs)):
                img_name = (
                    f"img{str(i+1).zfill(2)}_pred_"
                    f"epoch{str(self.current_epoch+1).zfill(3)}"
                )
                self.logger.experiment.log_image(
                    pred_imgs[i, 0], name=img_name
                )

        return {
            "val_loss": avg_loss,
            "log": tensorboard_logs,
            "progress_bar": tensorboard_logs,
        }

    def configure_optimizers(self):
        model_optimiser = optim.Adam(self.parameters(), self.hparams.lr,)
        return model_optimiser

    def train_dataloader(self):
        return self.data_module.train_dataloader(self.hparams.batch_size)

    def val_dataloader(self):
        return self.data_module.val_dataloader(self.hparams.batch_size)

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        parser = ArgumentParser(parents=[parent_parser])

        parser.add_argument(
            "--batch_size", default=512, type=int, metavar="N", help="",
        )
        parser.add_argument(
            "--lr",
            default=0.1,
            type=float,
            metavar="LR",
            help="initial learning rate",
        )

        parser.add_argument(
            "--lambda_target_prediction", default=1.0, type=float
        )
        parser.add_argument("--lambda_decoding_error", default=1.0, type=float)

        return parser
