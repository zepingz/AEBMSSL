import copy
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl

import models


class TestModel(pl.LightningModule):
    def __init__(self, data_module, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.data_module = data_module

        self.frame_encoder = models.components.encoder_dict[
            self.hparams.encoder_type
        ]()
        self.frame_decoder = models.components.decoder_dict[
            self.hparams.decoder_type
        ]()
        self.hidden_predictor = models.components.hidden_predictor_dict[self.hparams.hidden_predictor_type](
            self.frame_encoder._embedding_size, self.hparams)

        self.lambdas = dict(
            target_prediction=self.hparams.lambda_target_prediction,
        )

    def forward(self, batch):
        conditional_frames = batch["conditional_frames"]
        ptp = batch["PTP"]

        bs, seq_len, c, h, w = conditional_frames.shape

        conditional_frames = conditional_frames.view(bs * seq_len, c, h, w)
        encoded_frames = self.frame_encoder(conditional_frames)

        # predict target frame
        # encoded_frames = encoded_frames.view(bs, seq_len, -1)
        encoded_frames = encoded_frames.view(bs, seq_len, 2, 2, 64)
        predicted_hidden = self.hidden_predictor(encoded_frames, ptp, latent)

        predicted_frames = self.frame_decoder(predicted_hidden)

        return predicted_frames

    def sample_latent(self, predicted_hidden):
        bs = predicted_hidden.shape[0]
        latent = torch.randn(
            bs, self.hparams.latent_size).to(predicted_hidden.device)
        latent.requires_grad = True
        return latent

    def optimize_latent(
        self,
        latent,
        ptp,
        encoded_frames,
        target_frame,
    ):
        with torch.enable_grad():
            latent_optimizer = optim.LBFGS((latent,))

            def closure():
                # merged = self.latent_merger(
                #     predicted_hidden.detach(), latent
                # )
                merged = self.hidden_predictor(encoded_frames.detach(), ptp, latent)
                predicted_frames = self.frame_decoder(merged)
                energies = self.compute_energies(
                    predicted_frames,
                    target_frame,
                    latent=latent,
                )
                latent_optimizer.zero_grad()
                energies["total"].backward()
                return energies["total"]

            latent_optimizer.step(closure)

    def configure_optimizers(self):
        model_optimizer = optim.Adam(
            self.parameters(),
            self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return model_optimizer

    def train_dataloader(self):
        return self.data_module.train_dataloader(self.hparams.batch_size)

    def val_dataloader(self):
        return self.data_module.val_dataloader(self.hparams.batch_size)

    def linpred_dataloader(self):
        return self.data_module.linpred_dataloader(self.hparams.batch_size)

    def reconstruction_error(self, predicted, target):
        return F.mse_loss(predicted, target)

    def latent_regularizer(self, latent):
        return latent.norm(2, dim=-1).mean()

    def training_step(self, batch, batch_idx):
        loss, free_energies = self._compute_objective(batch)

        if self.logger is not None:
            self.logger.log_metrics(free_energies)
            # self.logger.log_metrics({
            #     "latent_l2norm": self.latent_regularizer(latent)
            # })

        # zero grad because inference may accumulate gradient
        self.zero_grad()

        return {
            "loss": loss,
            "progress_bar": free_energies,
        }

    def validation_step(self, batch, batch_idx):
        loss, free_energies = self._compute_objective(batch)

        # energies.update({
        #     "latent_l2norm": self.latent_regularizer(latent)
        # })

        result_dic = {
            "loss": loss, "free_energies": free_energies
        }

        return result_dic

    def validation_epoch_end(self, outputs):
        tensorboard_logs = {}

        if not self.hparams.no_eval:
            eval_acc = self.encoder_eval()
            tensorboard_logs.update({"eval_acc": eval_acc})

        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs["val_loss"] = avg_loss

        key_list = [
            "target_prediction",
            # "latent_l2norm"
        ]
        tensorboard_logs.update({
            "val_"
            + key: torch.stack(
                [x["free_energies"][key] for x in outputs]
            ).mean()
            for key in key_list
        })

        return {
            "val_loss": avg_loss,
            "log": tensorboard_logs,
            "progress_bar": tensorboard_logs,
        }

    def _compute_objective(self, batch):
        conditional_frames = batch["conditional_frames"]
        ptp = batch["PTP"][:, :self.hparams.ptp_size] # DEBUG
        target_frame = batch["target_frame"]

        bs, seq_len, c, h, w = conditional_frames.shape

        if self.hparams.no_latent:
            latent = 0
        else:
            latent = self._compute_optimal_latent(
                conditional_frames, ptp, target_frame
            )

        conditional_frames = conditional_frames.view(bs * seq_len, c, h, w)
        encoded_frames = self.frame_encoder(conditional_frames)
        encoded_frames = encoded_frames.view(bs, seq_len, -1)
        # encoded_frames = encoded_frames.view(bs, seq_len, 2, 2, 64)

        # encoded_target = self.frame_encoder(target_frame)

        merged = self.hidden_predictor(encoded_frames, ptp, latent)
        predicted_frame = self.frame_decoder(merged)

        free_energies = self.compute_energies(
            predicted_frame,
            target_frame,
            latent=latent,
        )
        objective = free_energies["total"]
        return objective, free_energies

    def _compute_optimal_latent(self, conditional_frames, ptp, target_frame):
        bs, seq_len, c, h, w = conditional_frames.shape

        # predict latent
        # encoded_target = self.frame_encoder(target_frame)

        conditional_frames = conditional_frames.view(bs * seq_len, c, h, w)
        encoded_frames = self.frame_encoder(conditional_frames)
        encoded_frames = encoded_frames.view(bs, seq_len, -1)
        # encoded_frames = encoded_frames.view(bs, seq_len, 2, 2, 64)

        latent = self.sample_latent(encoded_frames)
        self.optimize_latent(
            latent,
            ptp,
            encoded_frames,
            target_frame,
        )
        return latent

    def compute_energies(
        self,
        predicted_frames,
        target_frame,
        latent=None
    ):
        energies = {}
        energies["target_prediction"] = self.reconstruction_error(
            predicted_frames, target_frame
        )

        # if latent is not None:
        #     energies["latent_regularizer"] = self.latent_regularizer(latent)

        energies["total"] = sum(
            [energies[k] * self.lambdas[k] for k in energies]
        )

        return energies

    def encoder_eval(self):
        train_loader = self.linpred_dataloader()
        val_loader = self.val_dataloader()

        linear_predictor = models.LinearPredictor(
            64, 10, copy.deepcopy(self.frame_encoder)
        ).cuda()
        linear_optimizer = optim.Adam(
            linear_predictor.parameters(), lr=self.hparams.linpred_lr
        )

        with torch.enable_grad():
            for epoch in range(self.hparams.linpred_epochs):
                for batch in train_loader:
                    imgs = batch["target_frame"].cuda()
                    lbls = batch["labels"].flatten().cuda()
                    pred = linear_predictor.forward(imgs)
                    loss = nn.CrossEntropyLoss()(pred, lbls)

                    linear_optimizer.zero_grad()
                    loss.backward()
                    linear_optimizer.step()

        linear_predictor.eval()
        correct = 0
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["target_frame"].cuda()
                lbls = batch["labels"].flatten().cuda()
                pred = linear_predictor.forward(imgs)

                correct += (torch.argmax(pred, dim=1) == lbls).sum().item()

        return correct / len(val_loader.dataset)

    @staticmethod
    def add_model_specific_args(parser):
        # models
        parser.add_argument(
            "--encoder_type", default="test", help="Encoder Type",
        )
        parser.add_argument(
            "--decoder_type", default="test", help="Decoder Type",
        )
        parser.add_argument(
            "--hidden_predictor_type",
            default="transformer1",
            help="Hidden predictor Type",
        )

        parser.add_argument(
            "--hidden_predictor_layers",
            default=6,
            type=int,
            help="Number of layers in hidden predictor",
        )
        parser.add_argument(
            "--hidden_predictor_dim_feedforward",
            default=512,
            type=int,
            help="Number of FFN dimensionality in hidden predictor",
        )
        parser.add_argument(
            "--hidden_predictor_nhead",
            default=8,
            type=int,
            help="Number of attention head in hidden predictor",
        )

        # Data
        parser.add_argument(
            "--batch_size", default=512, type=int, metavar="N", help="",
        )

        parser.add_argument(
            "--lr",
            default=3e-4,
            type=float,
            metavar="LR",
            help="initial learning rate",
        )
        parser.add_argument(
            "--weight_decay", default=0.0, type=float,
        )
        parser.add_argument(
            "--lambda_latent_regularizer", default=1.0, type=float
        )
        # parser.add_argument("--lambda_latent_error", default=1.0, type=float)
        parser.add_argument(
            "--lambda_target_prediction", default=1.0, type=float
        )
        # parser.add_argument(
        #     "--lambda_decoding_error", default=1.0, type=float
        # )
        # parser.add_argument(
        #     "--lambda_cosine_distance", default=1.0, type=float
        # )

        # Latent
        parser.add_argument("--no_latent", action="store_true")
        parser.add_argument("--latent_size", type=int, default=2)
        parser.add_argument(
            "--start_latent_epoch", default=0, type=int,
        )


        # Encoder evaluation
        parser.add_argument("--no_eval", action="store_true")
        parser.add_argument(
            "--linpred_epochs", type=int, default=1,
        )
        parser.add_argument(
            "--linpred_batch_size", default=512, type=int,
        )
        parser.add_argument(
            "--linpred_lr", default=0.1, type=float,
        )

        parser.add_argument(
            "--ptp_size", default=5, type=int,
        )

        return parser
