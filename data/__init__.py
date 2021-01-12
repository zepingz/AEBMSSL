from data.dummy import DummyDataModule
from data.waymo import UnlabeledDataModule, LabeledDataModule
from data.moving_mnist import MovingMNISTDataModule
from data.carla import CarlaDataModule

__all__ = [
    "DummyDataModule",
    "UnlabeledDataModule",
    "LabeledDataModule",
    "MovingMNISTDataModule",
    "CarlaDataModule"
]

data_module_dict = {
    "dummy": DummyDataModule,
    "waymo": UnlabeledDataModule,
    "moving_mnist": MovingMNISTDataModule,
    "carla": CarlaDataModule,
}
