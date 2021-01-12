"""Abstract classes heirarchy for different ways of using energy-based models
"""
from argparse import ArgumentParser

import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl

import models


class BaseEBM(pl.LightningModule):
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
        self.latent_merger = models.components.latent_merger_dict[
            self.hparams.latent_merger_type
        ](self.hparams.latent_size)
        self.latent_predictor = models.components.latent_predictor_dict[
            self.hparams.latent_predictor_type
        ](self.hparams.latent_size)
        self.hidden_predictor = models.components.hidden_predictor_dict[
            self.hparams.hidden_predictor_type
        ]()

        self.lambdas = dict(
            latent_regularizer=self.hparams.lambda_latent_regularizer,
            latent_error=self.hparams.lambda_latent_error,
            target_prediction=self.hparams.lambda_target_prediction,
            decoding_error=self.hparams.lambda_decoding_error,
            cosine_distance=self.hparams.lambda_cosine_distance,
        )

    def predict_hidden(self, inputs):
        raise NotImplementedError

    def decode(self, hidden):
        raise NotImplementedError

    def merge(self, hidden, latent):
        raise NotImplementedError

    def get_latent(self, inputs):
        raise NotImplementedError

    def forward(self, inputs):
        hidden = self.predict_hidden(inputs)
        latent = self.get_latent(inputs)
        merged = self.merge(hidden, latent)
        prediction = self.decode(merged)
        return prediction

    def energy(inputs, targets, latent):
        raise NotImplementedError

    def estimate_free_energy_gradient(inputs, targets):
        raise NotImplementedError

    def reconstruction_error(self, predicted, target):
        return F.mse_loss(predicted, target)

    def compute_energies(
        self,
        decoded_frames,
        conditional_frames,
        predicted_frame,
        target_frame,
    ):
        energies = {}
        energies["target_prediction"] = self.reconstruction_error(
            predicted_frame, target_frame,
        )
        energies["decoding_error"] = self.reconstruction_error(
            decoded_frames, conditional_frames
        )
        return energies

    def configure_optimizers(self):
        model_optimizer = optim.Adam(self.parameters(), self.hparams.lr,)
        return model_optimizer

    def train_dataloader(self):
        return self.data_module.train_dataloader(self.hparams.batch_size)

    def val_dataloader(self):
        return self.data_module.val_dataloader(self.hparams.batch_size)

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        parser = ArgumentParser(parents=[parent_parser])

        # models
        parser.add_argument(
            "--encoder_type",
            choices=["dummy", "resnet", "test"],
            default="test",
            help="Encoder Type",
        )
        parser.add_argument(
            "--decoder_type",
            choices=["dummy", "resnet", "test"],
            default="test",
            help="Decoder Type",
        )
        parser.add_argument(
            "--latent_merger_type",
            choices=[
                "dummy",
                "resnet",
                "test",
                "linearize1",
                "linearize2",
                "linearize3",
            ],
            default="test",
            help="Latent merger Type",
        )
        parser.add_argument(
            "--latent_predictor_type",
            choices=["dummy", "resnet", "test", "linearize1", "linearize2"],
            default="test",
            help="Latent predictor Type",
        )
        parser.add_argument(
            "--hidden_predictor_type",
            choices=[
                "dummy",
                "resnet",
                "test1",
                "test2",
                "test3",
                "test4",
                "linearize",
            ],
            default="test1",
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
            "--lambda_latent_regularizer", default=1.0, type=float
        )
        parser.add_argument("--lambda_latent_error", default=1.0, type=float)
        parser.add_argument(
            "--lambda_target_prediction", default=1.0, type=float
        )
        parser.add_argument("--lambda_decoding_error", default=1.0, type=float)
        parser.add_argument(
            "--lambda_cosine_distance", default=1.0, type=float
        )

        return parser


class LatentMCMCSamplingEBM(BaseEBM):
    def _sample_MCMC(self, unnormalised_density):
        pass

    def estimate_free_energy_gradient(self, inputs, targets):
        def unnormalised_density(latent):
            return self.energy(inputs, targets, latent)

        for i in range(self.number_of_samples):
            latent = self._sample_MCMC(inputs, targets)
            energy = self.energy(inputs, targets, latent)
            energy.backward()


class NumericalIntegrationEBM(BaseEBM):
    def get_latents_distribution(inputs, targets):
        raise NotImplementedError

    def estimate_free_energy_gradient(self, inputs, targets):
        for probability, latent in self.get_latents_distribution(
            inputs, targets
        ):
            energy = self.energy(inputs, targets, latent)
            energy.backward(probability)


class VariationalSamplingEBM(BaseEBM):
    def _approximate_posterior_sample(self, inputs, targets):
        raise NotImplementedError

    def estimate_free_energy_gradient(self, inputs, targets):
        for i in range(self.number_of_samples):
            latent = self._approximate_posterior_sample(inputs, targets)
            energy = self.energy(inputs, targets, latent)
            energy.backward()
