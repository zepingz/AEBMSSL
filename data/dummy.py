import torch
from torch.utils.data import Dataset, DataLoader

from pytorch_lightning import LightningDataModule


class DummyDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.batch = dict(
            conditional_frames=torch.randn(2, 10),
            PTP=torch.randn(4, 4),
            target_frame=torch.randn(10),
        )

    def __len__(self):
        return 100

    def __getitem__(self, i):
        return self.batch


class DummyDataModule(LightningDataModule):
    def __init__(self, num_workers=12, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_workers = num_workers

    def train_dataloader(self, batch_size=32, transforms=None):
        dataset = DummyDataset()
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return loader

    def val_dataloader(self, batch_size=32, transforms=None):
        dataset = DummyDataset()
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return loader
