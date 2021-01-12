import os
import numpy as np
from glob import glob
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class CarlaDataset(Dataset):
    def __init__(
        self,
        data_root,
        train=True,
        num_conditional_frames=2,
        mode="ego",
        transform=None
    ):
        if mode == "ego":
            data_root = os.path.join(data_root, "CarlaEgoViewSample")
        elif mode == "bev":
            data_root = os.path.join(data_root, "CarlaTopDownViewSample")
        else:
            raise Exception(f"mode should be 'ego' or 'bev' not '{mode}'")

        self.num_conditional_frames = num_conditional_frames
        self.transform = transform

        self.img_paths = sorted(glob(os.path.join(data_root, "frames", "*.jpg")))
        self.pose_paths = [os.path.join(
            data_root, "pose", f"{os.path.basename(path).split('.')[0]}.npy") for path in self.img_paths]

        split_num = int(0.8 * len(self.img_paths))
        if train:
            self.img_paths = self.img_paths[:split_num]
            self.pose_paths = self.pose_paths[:split_num]
        else:
            self.img_paths = self.img_paths[split_num:]
            self.pose_paths = self.pose_paths[split_num:]

    def __getitem__(self, index):
        conditional_frames = []

        # Conditional frames
        for i in range(self.num_conditional_frames):
            conditional_frames.append(self.transform(Image.open(self.img_paths[index+i])))
        conditional_frames = torch.stack(conditional_frames)

        # Target frame
        target_frame = self.transform(Image.open(self.img_paths[index+self.num_conditional_frames]))

        # Pose
        m1 = np.load(self.pose_paths[index+self.num_conditional_frames-1])
        m2 = np.load(self.pose_paths[index+self.num_conditional_frames])
        ptp = torch.from_numpy(np.linalg.inv(m2) @ m1).reshape(-1)

        batch = {
            "conditional_frames": conditional_frames,
            "PTP": ptp,
            "target_frame": target_frame
        }

        return batch

    def __len__(self):
        return len(self.img_paths) - self.num_conditional_frames


class CarlaDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root,
        num_conditional_frames=2,
        mode="ego",
        height=60,
        width=80,
        num_workers=16,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.data_root = data_root
        self.num_conditional_frames = num_conditional_frames
        self.mode = mode
        self.height = height
        self.width = width
        self.num_workers = num_workers

    def train_dataloader(self, batch_size=32, transform=None):
        if transform is None:
            transform = self._default_transform()

        dataset = CarlaDataset(
            self.data_root,
            train=True,
            num_conditional_frames=self.num_conditional_frames,
            mode=self.mode,
            transform=transform)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            # drop_last=True,
            # pin_memory=True,
            # worker_init_fn=set_worker_random_seed,
        )

        return loader

    def val_dataloader(self, batch_size=32, transform=None):
        if transform is None:
            transform = self._default_transform()

        dataset = CarlaDataset(
            self.data_root,
            train=False,
            num_conditional_frames=self.num_conditional_frames,
            mode=self.mode,
            transform=transform)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            # drop_last=True,
            # pin_memory=True,
            # worker_init_fn=set_worker_random_seed,
        )

        return loader

    def _default_transform(self):
        transform = transforms.Compose([
            transforms.Resize([self.height, self.width]),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ])
        return transform
