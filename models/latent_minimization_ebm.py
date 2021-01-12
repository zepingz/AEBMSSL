import copy
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl

import models


class LatentMinimizationEBM(pl.LightningModule):
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
        self.latent_merger = models.components.latent_merger_dict[
            self.hparams.latent_merger_type
        ](self.hparams.latent_size)
        self.latent_predictor = models.components.latent_predictor_dict[
            self.hparams.latent_predictor_type
        ](self.hparams.latent_size)
        self.hidden_predictor = models.components.hidden_predictor_dict[
            self.hparams.hidden_predictor_type
        ](self.hparams.ptp_size)

        self.lambdas = dict(
            latent_regularizer=self.hparams.lambda_latent_regularizer,
            latent_error=self.hparams.lambda_latent_error,
            target_prediction=self.hparams.lambda_target_prediction,
            cosine_distance=self.hparams.lambda_cosine_distance,
            # hidden_error=self.hparams.lambda_hidden_error,
        )

    def get_latent(self, inputs):
        device = inputs["conditional_frames"].device
        latent = torch.randn(
            self.hparams.batch_size, self.hparams.latent_size
        ).to(device)
        return latent

    def forward(self, batch):
        conditional_frames = batch["conditional_frames"]
        ptp = batch["PTP"]

        bs, seq_len, c, h, w = conditional_frames.shape

        latent = self.get_latent()

        conditional_frames = conditional_frames.view(bs * seq_len, c, h, w)
        encoded_frames = self.frame_encoder(conditional_frames)
        encoded_frames = encoded_frames.view(bs, seq_len, -1)

        predicted_hidden = self.hidden_predictor(encoded_frames, ptp)
        merged = self.latent_merger(predicted_hidden, latent)
        predicted_frame = self.frame_decoder(merged)

        return predicted_frame

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

    def latent_regularizer(self, latent):
        device = latent.device
        if self.hparams.latent_regularizer_type == "l1norm":
            return latent.norm(1, dim=-1).mean()
        if self.hparams.latent_regularizer_type == "l2norm":
            return latent.norm(2, dim=-1).mean()
        if self.hparams.latent_regularizer_type == "l2norm_with_noise":
            return (latent + torch.randn(*latent.shape, device=device)).norm(2, dim=-1).mean()

    def latent_error(self, predicted, target):
        return F.mse_loss(predicted, target)

    def cosine_distance(self, encoded_frames, merged):
        delta_1 = encoded_frames[:, -1] - encoded_frames[:, -2]
        delta_2 = merged - encoded_frames[:, -1]
        return 1 - F.cosine_similarity(delta_1, delta_2).mean()

    def reconstruction_error(self, predicted, target):
        return F.mse_loss(predicted, target)

    def training_step(self, batch, batch_idx):
        loss, energies = self._compute_objective(batch)

        if self.logger is not None:
            self.logger.log_metrics(energies)

        # zero grad because inference may accumulate gradient
        self.zero_grad()

        return dict(loss=loss, progress_bar=energies)

    def validation_step(self, batch, batch_idx):
        result_dic = {}
        if batch_idx == 0 and self.logger is not None:
            loss, energies, pred_imgs = self._compute_objective(
                batch, return_pred_imgs=True
            )

            # Include images for logging later
            result_dic["pred_imgs"] = pred_imgs.detach().cpu().numpy()
            if self.current_epoch == 0:
                result_dic["imgs"] = (
                    batch["target_frame"].detach().cpu().numpy()
                )
        else:
            loss, energies = self._compute_objective(batch)

        result_dic.update(
            {"loss": energies["total"], "free_energies": energies}
        )

        return result_dic

    def validation_epoch_end(self, outputs):
        eval_acc = self.encoder_eval()
        tensorboard_logs = {"eval_acc": eval_acc}

        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()

        key_list = [
            "latent_regularizer",
            "target_prediction",
            "latent_error",
            "cosine_distance",
        ]
        tensorboard_logs.update(
            {
                "val_"
                + key: torch.stack(
                    [x["free_energies"][key] for x in outputs]
                ).mean()
                for key in key_list
            }
        )
        tensorboard_logs["val_loss"] = avg_loss

        # Log images
        if self.logger is not None:
            if self.current_epoch == 0:
                imgs = outputs[0]["imgs"][:10]
                for i in range(len(imgs)):
                    self.logger.experiment.log_image(
                        imgs[i, 0], name=f"img{str(i+1).zfill(2)}_gt"
                    )

            pred_imgs = outputs[0]["pred_imgs"][:10]
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

    def encoder_eval(self):
        train_loader = self.train_dataloader()
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

    def sample_latent(self, predicted_latent):
        batch_size = predicted_latent.shape[0]
        latent = predicted_latent.detach() + torch.randn(
            batch_size, self.hparams.latent_size
        ).to(predicted_latent.device)
        latent.requires_grad = True
        return latent

    def compute_energies(
        self,
        predicted_frame,
        target_frame,
        encoded_frames,
        merged,
        encoded_target,
        predicted_latent=None,
        latent=None,
    ):
        energies = {}
        energies["target_prediction"] = self.reconstruction_error(
            predicted_frame, target_frame,
        )

        if predicted_latent is not None:
            energies["latent_regularizer"] = self.latent_regularizer(latent)
            energies["latent_error"] = self.latent_error(predicted_latent, latent)
        else:
            energies["latent_regularizer"] = torch.zeros(1)
            energies["latent_error"] = torch.zeros(1)

        energies["cosine_distance"] = self.cosine_distance(
            encoded_frames, merged
        )

        # energies["hidden_error"] = self.reconstruction_error(
        #     merged, encoded_target
        # )

        energies["total"] = sum(
            [energies[k] * self.lambdas[k] for k in energies]
        )

        return energies

    def _compute_optimal_latent(self, conditional_frames, ptp, target_frame):
        bs, seq_len, c, h, w = conditional_frames.shape

        # predict latent
        encoded_target = self.frame_encoder(target_frame)

        conditional_frames = conditional_frames.view(bs * seq_len, c, h, w)
        encoded_frames = self.frame_encoder(conditional_frames)
        encoded_frames = encoded_frames.view(bs, seq_len, -1)

        predicted_hidden = self.hidden_predictor(encoded_frames, ptp)

        predicted_latent = self.latent_predictor(
            encoded_target, predicted_hidden
        )
        latent = self.sample_latent(predicted_latent)
        self.optimize_latent(
            latent,
            predicted_latent,
            predicted_hidden,
            encoded_frames,
            conditional_frames,
            target_frame,
            encoded_target,
        )
        return latent, predicted_latent

    def optimize_latent(
        self,
        latent,
        predicted_latent,
        predicted_hidden,
        encoded_frames,
        conditional_frames,
        target_frame,
        encoded_target,
    ):

        # This is required because lightning does no-grad for evaluation.
        with torch.enable_grad():
            if self.hparams.latent_optimizer == "lbfgs":
                latent_optimizer = optim.LBFGS(
                    (latent,),
                    max_iter=self.hparams.latent_steps,
                    line_search_fn="strong_wolfe"
                    if self.hparams.use_strong_wolfe
                    else None,
                )

                def closure():
                    merged = self.latent_merger(
                        predicted_hidden.detach(), latent
                    )
                    predicted_frame = self.frame_decoder(merged)
                    energies = self.compute_energies(
                        predicted_frame,
                        target_frame.detach(),
                        encoded_frames.detach(),
                        merged,  # .detach(),
                        encoded_target.detach(),
                        predicted_latent=predicted_latent.detach(),
                        latent=latent,
                    )
                    latent_optimizer.zero_grad()
                    energies["total"].backward()
                    return energies["total"]

                latent_optimizer.step(closure)

            if self.hparams.latent_optimizer == "gd":
                latent_optimizer = optim.SGD((latent,), self.hparams.latent_lr)
                for _ in range(self.hparams.latent_steps):
                    merged = self.latent_merger(
                        predicted_hidden.detach(), latent
                    )
                    predicted_frame = self.frame_decoder(merged)
                    energies = self.compute_energies(
                        predicted_frame,
                        target_frame.detach(),
                        encoded_frames.detach(),
                        merged,  # .detach(),
                        encoded_target.detach(),
                        predicted_latent=predicted_latent.detach(),
                        latent=latent,
                    )
                    latent_optimizer.zero_grad()
                    energies["total"].backward()
                    latent_optimizer.step()

    def _compute_objective(self, batch, return_pred_imgs=False):
        conditional_frames = batch["conditional_frames"]
        ptp = batch["PTP"][:, :self.hparams.ptp_size] # DEBUG
        # ptp = batch["PTP"][:, :2]
        target_frame = batch["target_frame"]

        bs, seq_len, c, h, w = conditional_frames.shape

        dropout = random.random() < self.hparams.dropout_rate
        if not dropout and self.current_epoch >= self.hparams.no_latent_epoch:
            latent, predicted_latent = self._compute_optimal_latent(
                conditional_frames, ptp, target_frame
            )

        conditional_frames = conditional_frames.view(bs * seq_len, c, h, w)
        encoded_frames = self.frame_encoder(conditional_frames)
        encoded_frames = encoded_frames.view(bs, seq_len, -1)

        encoded_target = self.frame_encoder(target_frame)

        if not dropout and self.current_epoch >= self.hparams.no_latent_epoch:
            predicted_hidden = self.hidden_predictor(encoded_frames, ptp)
            merged = self.latent_merger(predicted_hidden, latent)
            predicted_frame = self.frame_decoder(merged)

            free_energies = self.compute_energies(
                predicted_frame,
                target_frame,
                encoded_frames,
                merged,
                encoded_target,
                predicted_latent=predicted_latent,
                latent=latent,
            )

        else:
            merged = self.hidden_predictor(encoded_frames, ptp)
            predicted_frame = self.frame_decoder(merged)

            free_energies = self.compute_energies(
                predicted_frame,
                target_frame,
                encoded_frames,
                merged,
                encoded_target,
            )

        objective = free_energies["total"]

        if return_pred_imgs:
            return objective, free_energies, predicted_frame
        else:
            return objective, free_energies

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
            "--latent_merger_type",
            default="new_linearize",
            help="Latent merger Type",
        )
        parser.add_argument(
            "--latent_predictor_type",
            default="new_linearize",
            help="Latent predictor Type",
        )
        parser.add_argument(
            "--hidden_predictor_type",
            default="new_linearize",
            help="Hidden predictor Type",
        )

        # Data
        parser.add_argument(
            "--batch_size", default=256, type=int, metavar="N", help="",
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
        parser.add_argument("--lambda_latent_error", default=1.0, type=float)
        parser.add_argument(
            "--lambda_target_prediction", default=1.0, type=float
        )
        parser.add_argument(
            "--lambda_cosine_distance", default=1.0, type=float
        )
        # parser.add_argument(
        #     "--lambda_hidden_error", default=1.0, type=float
        # )

        parser.add_argument("--latent_steps", default=10, type=int)
        parser.add_argument("--latent_lr", default=0.1, type=float)
        parser.add_argument("--latent_size", type=int, default=2)
        parser.add_argument(
            "--latent_optimizer",
            choices=["lbfgs", "gd"],
            default="lbfgs",
            help="Latent optimizer type",
        )
        parser.add_argument(
            "--use_strong_wolfe",
            action="store_true",
            help="Whether to use strong wolfe in lbfgs",
        )

        parser.add_argument(
            "--dropout_rate", default=0.0, type=float,
        )

        parser.add_argument(
            "--latent_regularizer_type", default="l2norm", type=str,
        )

        # Encoder evaluation
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
            "--ptp_size", type=int, default=5,
        )

        parser.add_argument(
            "--no_latent_epoch", type=int, default=0,
        )

        return parser
