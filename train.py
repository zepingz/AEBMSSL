import os
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import models
import data


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

    # Model
    parser.add_argument(
        "--model", type=str, default="test",
    )

    # Data
    parser.add_argument(
        "--dataset",
        choices=["dummy", "waymo", "moving_mnist", "carla"],
        default="moving_mnist",
        help="Dataset Type",
    )
    parser.add_argument(
        "--num_conditional_frames",
        type=int,
        default=2,
        help="Number of input conditional frames",
    )
    parser.add_argument(
        "--waymo_data_root",
        type=str,
        default="/scratch/zz2332/WaymoDataset/",
        help="Where to load waymo data",
    )
    parser.add_argument(
        "--waymo_subset",
        action="store_true",
        help="Whether to use subset of Waymo or not",
    )
    parser.add_argument(
        "--mnist_data_root",
        type=str,
        default="../MNIST/",
        help="Where to load mnist data",
    )
    parser.add_argument(
        "--mnist_determinstic",
        action="store_true",
        help="Whether to use determinstic moving mnist or not",
    )
    parser.add_argument(
        "--mnist_train_dataset_size",
        type=int,
        default=180000,
        help="Size of mnist training dataset"
    )
    parser.add_argument(
        "--mnist_val_dataset_size",
        type=int,
        default=20000,
        help="Size of mnist validation dataset"
    )
    parser.add_argument(
        "--mnist_linpred_dataset_size",
        type=int,
        default=10000,
        help="Size of mnist linpred dataset"
    )
    parser.add_argument(
        "--carla_data_root",
        type=str,
        default="../carla/",
        help="Where to load carla data",
    )
    parser.add_argument(
        "--carla_mode",
        type=str,
        default="bev",
        help="The carla mode should be ego or bev",
    )
    parser.add_argument(
        "--num_workers", type=int, default=16, help="Number of workers"
    )

    parser.add_argument("--not_log", action="store_true")
    parser.add_argument("--counter", type=int, default=None)

    # trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # model args
    # TODO: change this hacky approach
    # parser = models.LatentMinimizationEBM.add_model_specific_args(parser)
    parser = models.TestModel.add_model_specific_args(parser)

    return parser.parse_args()


if __name__ == "__main__":
    # get args
    args = get_args()

    # seed
    # deterministic is slow... so we disable it for now
    pl.seed_everything(args.seed)

    # init data module
    if args.dataset == "dummy":
        data_module = data.DummyDataModule()
    elif args.dataset == "waymo":
        data_module = data.UnlabeledDataModule(
            args.waymo_data_root,
            args.num_conditional_frames,
            subset=args.waymo_subset,
            num_workers=args.num_workers,
        )
    elif args.dataset == "moving_mnist":
        data_module = data.MovingMNISTDataModule(
            args.mnist_data_root,
            args.num_conditional_frames,
            num_digits=1,
            ptp_type="2",
            determinstic=args.mnist_determinstic,
            angle_range=(-30, 30),
            scale_range=(0.8, 1.2),
            shear_range=(-20, 20),
            train_dataset_size=args.mnist_train_dataset_size,
            val_dataset_size=args.mnist_val_dataset_size,
            linpred_dataset_size=args.mnist_linpred_dataset_size,
            num_workers=args.num_workers,
        )
    elif args.dataset == "carla":
        data_module = data.CarlaDataModule(
            args.carla_data_root,
            num_conditional_frames=args.num_conditional_frames,
            mode=args.carla_mode,
            height=256,
            width=256,
            num_workers=args.num_workers,
        )

    # init model
    model = models.model_dict[args.model](data_module, **vars(args))

    api_key = os.environ.get("COMET_API_KEY")
    project_name = os.environ.get("COMET_PROJECT_NAME")
    workspace = os.environ.get("COMET_WORKSPACE")
    do_log = (
        not args.not_log
        and api_key is not None
        and project_name is not None
        and workspace is not None
    )
    if do_log:
        # init comet logger
        comet_logger = CometLogger(
            api_key=api_key, project_name=project_name, workspace=workspace, save_dir=args.default_root_dir
        )
        old_experiment_name = comet_logger.experiment.get_key()
        new_experiment_name = f"ebm_{args.counter}"
        comet_logger.experiment.set_name(new_experiment_name)
    else:
        comet_logger = None

    # init checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        verbose=True, save_top_k=-1, monitor="val_loss", mode="min"
    )

    # init trainer. all flags are available via CLI (--gpus, --max_epochs)
    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=comet_logger,
        num_sanity_val_steps=0,
        checkpoint_callback=checkpoint_callback,
    )

    # train!
    trainer.fit(model)

    if do_log:
        # rename folder
        folder_path = os.path.join(args.default_root_dir, project_name)
        os.rename(
            os.path.join(folder_path, old_experiment_name),
            os.path.join(folder_path, new_experiment_name),
        )
