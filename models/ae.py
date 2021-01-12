from argparse import ArgumentParser

import torch
from torch import optim
import torch.nn.functional as F
import pytorch_lightning as pl

import models.components


class Autoencoder(pl.LightningModule):
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

    def forward(self, x):
        x = self.frame_encoder(x)
        x = self.frame_decoder(x)
        return x

    def reconstruction_error(self, predicted, target):
        return F.mse_loss(predicted, target)

    def training_step(self, batch, batch_idx):
        imgs = batch["target_frame"]
        pred_imgs = self.forward(imgs)
        loss = self.reconstruction_error(pred_imgs, imgs)

        self.logger.log_metrics({"decoding_error": loss})
        return dict(loss=loss)

    def validation_step(self, batch, batch_idx):
        imgs = batch["target_frame"]
        pred_imgs = self.forward(imgs)
        loss = self.reconstruction_error(pred_imgs, imgs)

        result_dic = {
            "loss": loss,
        }
        if batch_idx == 0:
            result_dic["pred_imgs"] = pred_imgs.detach().cpu().numpy()
            if self.current_epoch == 0:
                result_dic["imgs"] = imgs.detach().cpu().numpy()

        return result_dic

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x.get("loss") for x in outputs]).mean()

        tensorboard_logs = {"val_loss": avg_loss}

        # Log images
        if not self.hparams.debug:
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
            "--batch_size", default=256, type=int, metavar="N", help="",
        )
        parser.add_argument(
            "--lr",
            default=3e-4,
            type=float,
            metavar="LR",
            help="initial learning rate",
        )
        return parser


def get_args():
    # global args
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--save_path", dest="default_root_dir", default="./output"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="seed for initializing training. ",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/scratch/zz2332/WaymoDataset/",
        help="Where to load waymo data",
    )
    parser.add_argument(
        "--mnist_data_root",
        type=str,
        default="/scratch/zz2332/MNIST/",
        help="Where to load mnist data",
    )
    parser.add_argument(
        "--num_workers", type=int, default=16, help="Number of workers"
    )

    # data
    parser.add_argument(
        "--num_conditional_frames",
        type=int,
        default=2,
        help="Number of input conditional frames",
    )
    parser.add_argument(
        "--dataset",
        choices=["dummy", "waymo", "moving_mnist"],
        default="moving_mnist",
        help="Dataset Type",
    )
    parser.add_argument(
        "--subset", action="store_true", help="Whether to use subset or not",
    )

    # models
    parser.add_argument(
        "--encoder",
        choices=["dummy", "resnet", "test"],
        default="test",
        help="Encoder Type",
    )
    parser.add_argument(
        "--decoder",
        choices=["dummy", "resnet", "test"],
        default="test",
        help="Decoder Type",
    )

    parser.add_argument(
        "--debug", action="store_true",
    )

    # trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # model args
    parser = Autoencoder.add_model_specific_args(parser)
    return parser.parse_args()
