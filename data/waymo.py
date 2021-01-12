import os
import h5py
import json
from enum import Enum

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from pytorch_lightning import LightningDataModule


class CameraName(Enum):
    UNKNOWN = 0
    FRONT = 1
    FRONT_LEFT = 2
    FRONT_RIGHT = 3
    SIDE_LEFT = 4
    SIDE_RIGHT = 5


class LaserName(Enum):
    UNKNOWN = 0
    TOP = 1
    FRONT = 2
    SIDE_LEFT = 3
    SIDE_RIGHT = 4
    REAR = 5


class RollingShutterReadOutDirection(Enum):
    UNKNOWN = 0
    TOP_TO_BOTTOM = 1
    LEFT_TO_RIGHT = 2
    BOTTOM_TO_TOP = 3
    RIGHT_TO_LEFT = 4
    GLOBAL_SHUTTER = 5


class ObjectType(Enum):
    UNKNOWN = 0
    VEHICLE = 1
    PEDESTRIAN = 2
    SIGN = 3
    CYCLIST = 4


def collate_fn(batch):
    imgs = []
    targets = []
    for data in batch:
        lbls = data["label_type"]
        boxes = data["label"]
        if len(lbls) == 0:
            continue

        imgs.append(data["frame"])
        targets.append({"labels": lbls, "boxes": boxes})

    data_dict = {"frame": torch.stack(imgs), "target": targets}
    return data_dict


class WaymoDataset(Dataset):
    def __init__(
        self,
        data_root,
        transform,
        train=True,
        subset=False,
        all_camera_view=False,
    ):
        """
        Args:
            data_root (string): Where to load the data
            num_conditional_frames (int): Number of conditional frames
            transform (Transform): The function to preprocess the image
            train (bool): Whether ot use training dataset or test dataset
            subset (bool): Whether to use subset or not
            all_camera_view (bool): Whether to include all camera view
        """

        self.data_root = data_root
        self.transform = transform
        self.train = train
        self.subset = subset
        self.all_camera_view = all_camera_view

        # Load json file
        split = "training" if self.train else "validation"
        with open(os.path.join(self.data_root, f"{split}.json"), "r") as f:
            self.meta_dict = json.load(f)
        self.data_root = os.path.join(self.data_root, split)

    def initialize_indices(self, num_conditional_frames):
        episode_names = sorted(self.meta_dict.keys())
        if self.subset:
            episode_names = episode_names[:20]

        self.indices = []
        for episode_name in episode_names:
            length = self.meta_dict[episode_name]["length"]
            if self.all_camera_view:
                self.indices += [
                    {"episode_name": episode_name, "frame_idx": str(frame_idx)}
                    for frame_idx in range(length - num_conditional_frames)
                ]
            else:
                self.indices += [
                    {
                        "episode_name": episode_name,
                        "camera_idx": str(camera_idx),
                        "frame_idx": str(frame_idx),
                    }
                    for camera_idx in range(1, 6)
                    for frame_idx in range(length - num_conditional_frames)
                ]

    def get_index_info(self, index):
        info = self.indices[index]
        if self.all_camera_view:
            return info["episode_name"], info["frame_idx"]
        else:
            return info["episode_name"], info["frame_idx"], info["camera_idx"]

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.indices)


class UnlabeledDataset(WaymoDataset):
    def __init__(
        self,
        data_root,
        transform,
        num_conditional_frames,
        train=True,
        subset=False,
        all_camera_view=False,
    ):
        super().__init__(data_root, transform, train, subset, all_camera_view)

        self.num_conditional_frames = num_conditional_frames

        # Initialize indices
        self.initialize_indices(self.num_conditional_frames)

    def __getitem__(self, index):
        info = self.get_index_info(index)
        episode_name = info[0]
        frame_idx = info[1]
        if not self.all_camera_view:
            camera_idx = info[2]
        frame_idx = int(frame_idx)

        file = h5py.File(
            os.path.join(self.data_root, f"{episode_name}.hdf5"), "r"
        )

        if self.all_camera_view:
            conditional_frames = []
            target_frame = []
            for camera_idx in range(1, 6):
                conditional_frames_, target_frame_ = self.get_data(
                    file, frame_idx, str(camera_idx)
                )
                conditional_frames.append(conditional_frames_)
                target_frame.append(target_frame_)
            conditional_frames = torch.stack(conditional_frames, dim=1)
            target_frame = torch.stack(target_frame, dim=0)
        else:
            conditional_frames, target_frame = self.get_data(
                file, frame_idx, camera_idx
            )

        # Load PTP
        PTP = torch.from_numpy(
            file[str(frame_idx + self.num_conditional_frames)]["PTP"][:]
        ).float()

        batch = dict(
            conditional_frames=conditional_frames,
            PTP=PTP,
            target_frame=target_frame,
        )

        return batch

    def get_data(self, file, frame_idx, camera_idx):
        # Load conditional frames
        conditional_frames = torch.stack(
            [
                self.transform(
                    file[str(frame_idx + i)][camera_idx]["image"][:]
                )
                for i in range(self.num_conditional_frames)
            ]
        )

        # Load target frame
        target_frame = self.transform(
            file[str(frame_idx + self.num_conditional_frames)][camera_idx][
                "image"
            ][:]
        )

        return conditional_frames, target_frame


class LabeledDataset(WaymoDataset):
    def __init__(
        self,
        data_root,
        transform,
        train=True,
        subset=False,
        all_camera_view=False,
    ):
        super().__init__(data_root, transform, train, subset, all_camera_view)

        # Initialize indices
        self.initialize_indices(0)

    def __getitem__(self, index):
        info = self.get_index_info(index)
        episode_name = info[0]
        frame_idx = info[1]
        if not self.all_camera_view:
            camera_idx = info[2]

        file = h5py.File(
            os.path.join(self.data_root, f"{episode_name}.hdf5"), "r"
        )

        if self.all_camera_view:
            frame = []
            label = []
            label_type = []
            for camera_idx in range(1, 6):
                frame_, label_, label_type_ = self.get_data(
                    file[frame_idx], str(camera_idx)
                )
                frame.append(frame_)
                label.append(label_)
                label_type.append(label_type_)
            frame = torch.stack(frame, dim=0)
        else:
            frame, label, label_type = self.get_data(
                file[frame_idx], camera_idx
            )

        batch = dict(frame=frame, label=label, label_type=label_type)
        return batch

    def get_data(self, frame_group, camera_idx):
        # Load frame
        frame = self.transform(frame_group[camera_idx]["image"][:])

        # Load label
        label = torch.from_numpy(frame_group[camera_idx]["label"][:]).float()

        # Load label type
        label_type = torch.from_numpy(frame_group[camera_idx]["type"][:])

        return frame, label, label_type


class WaymoDataModule(LightningDataModule):
    def __init__(
        self,
        data_root,
        subset=False,
        all_camera_view=False,
        num_workers=16,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.data_root = data_root
        self.subset = subset
        self.all_camera_view = all_camera_view
        self.num_workers = num_workers

    def train_dataloader(self, batch_size=32, transform=None):
        raise NotImplementedError

    def val_dataloader(self, batch_size=32, transform=None):
        raise NotImplementedError

    def _default_transform(self):
        waymo_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.3133904, 0.34293564, 0.37411802],
                    std=[0.20420429, 0.21793098, 0.25517774],
                ),
            ]
        )
        return waymo_transform


class UnlabeledDataModule(WaymoDataModule):
    def __init__(
        self,
        data_root,
        num_conditional_frames,
        subset=False,
        all_camera_view=False,
        num_workers=16,
        *args,
        **kwargs,
    ):
        super().__init__(
            data_root, subset, all_camera_view, num_workers, *args, **kwargs
        )
        self.num_conditional_frames = num_conditional_frames

    def train_dataloader(self, batch_size=32, transform=None):
        if transform is None:
            transform = self._default_transform()

        dataset = UnlabeledDataset(
            self.data_root,
            transform,
            self.num_conditional_frames,
            train=True,
            subset=self.subset,
            all_camera_view=self.all_camera_view,
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader

    def val_dataloader(self, batch_size=32, transform=None):
        if transform is None:
            transform = self._default_transform()

        dataset = UnlabeledDataset(
            self.data_root,
            transform,
            self.num_conditional_frames,
            train=False,
            subset=self.subset,
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader


class LabeledDataModule(WaymoDataModule):
    def __init__(
        self,
        data_root,
        subset=False,
        all_camera_view=False,
        num_workers=16,
        *args,
        **kwargs,
    ):
        super().__init__(
            data_root, subset, all_camera_view, num_workers, *args, **kwargs
        )

    def train_dataloader(self, batch_size=32, transform=None):
        if transform is None:
            transform = self._default_transform()

        dataset = LabeledDataset(
            self.data_root,
            transform,
            train=True,
            subset=self.subset,
            all_camera_view=self.all_camera_view,
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        return loader

    def val_dataloader(self, batch_size=32, transform=None):
        if transform is None:
            transform = self._default_transform()

        dataset = LabeledDataset(
            self.data_root, transform, train=False, subset=self.subset
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        return loader
