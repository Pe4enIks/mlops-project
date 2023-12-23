import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class CatsDogsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_data_path: str,
        val_data_path: str,
        test_data_path: str,
        train_transform: transforms.Compose,
        test_transform: transforms.Compose,
        train_batch_size: int,
        val_batch_size: int,
        test_batch_size: int,
        num_workers: int,
    ) -> None:
        super().__init__()

        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.test_data_path = test_data_path

        self.train_transform = train_transform
        self.test_transform = test_transform

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size

        self.num_workers = num_workers

    def setup(self, stage: str) -> None:
        if stage == "fit":
            train_dataset = datasets.ImageFolder(
                self.train_data_path, transform=self.train_transform
            )

            val_dataset = datasets.ImageFolder(
                self.val_data_path, transform=self.train_transform
            )

            self.train_dataset = train_dataset
            self.val_dataset = val_dataset

        if stage == "test":
            test_dataset = datasets.ImageFolder(
                self.test_data_path, transform=self.test_transform
            )
            self.test_dataset = test_dataset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
