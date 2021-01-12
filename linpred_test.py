import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import CometLogger

from data.moving_mnist import MovingMNISTDataModule


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ebm_num", type=str)
    parser.add_argument("--epoch_num", default=99, type=int)
    parser.add_argument("--linpred_size", default=500, type=int, choices=[10, 50, 100, 200, 500])
    parser.add_argument("--lr", default=0.003, type=float)
    parser.add_argument("--weight_decay", default=0, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--seed", default=45, type=int)
    parser = pl.Trainer.add_argparse_args(parser)
    # args = parser.parse_args([
    #     "--ebm_num", "17_1",
    #     "--epoch_num", "99",
    #     "--linpred_size", "500",
    #     "--max_epochs", "30",
    #     "--gpus", "1",
    #     "--lr", "0.003",
    #     "--weight_decay", "0",
    #     "--batch_size", "32",
    #     "--seed", "42",
    # ])
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    data_module = MovingMNISTDataModule(
        '../MNIST',
        2,
        num_digits=1,
        ptp_type="2",
        determinstic=False,
        train_dataset_size=10,
        val_dataset_size=20000,
        linpred_dataset_size=args.linpred_size,
        num_workers=12,
    )

    api_key = os.environ.get("COMET_API_KEY")
    project_name = "moving-mnist-linpred" # os.environ.get("COMET_PROJECT_NAME")
    workspace = os.environ.get("COMET_WORKSPACE")
    if api_key is not None and workspace is not None:
        comet_logger = CometLogger(
            api_key=api_key, project_name=project_name, workspace=workspace, save_dir="./output"
        )
        old_experiment_name = comet_logger.experiment.get_key()
        new_experiment_name = f"ebm_{args.ebm_num}_{args.linpred_size}"
        comet_logger.experiment.set_name(new_experiment_name)
    else:
        comet_logger = None

    predictor = Predictor(data_module, f"output/ebm/ebm_{args.ebm_num}/checkpoints/epoch={args.epoch_num}.ckpt", **vars(args))
    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=comet_logger,
        num_sanity_val_steps=0,
    )
    trainer.fit(predictor)
