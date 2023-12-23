from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn, optim
from torchmetrics.functional import accuracy, auroc, precision, recall
from torchvision import models


class ClassificationModel(pl.LightningModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.example_input_array = torch.rand((1, 3, cfg.transform.h, cfg.transform.w))
        self.cfg = cfg
        if "resnet" in cfg.model:
            model = getattr(models, cfg.model)(weights="DEFAULT")
            model.fc = nn.Linear(model.fc.in_features, 2)
            self.model = model
        else:
            raise ValueError(f"{cfg.model} is not available")

    def forward(self, x: Any) -> Any:
        return self.model(x)

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def _metrics(self, y_hat: Any, y: Any) -> list[float]:
        acc = accuracy(
            y_hat, y, task="multiclass", num_classes=2, average="weighted"
        ).item()

        prec = precision(
            y_hat, y, task="multiclass", num_classes=2, average="weighted"
        ).item()

        rec = recall(
            y_hat, y, task="multiclass", num_classes=2, average="weighted"
        ).item()

        roc = auroc(
            y_hat, y, task="multiclass", num_classes=2, average="weighted"
        ).item()

        return [acc, prec, rec, roc]

    def _log_metrics(self, metrics: list[tuple[str, float]]) -> None:
        for metric in metrics:
            metric_name, metric_val = metric
            self.log(
                metric_name,
                metric_val,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        x, y = batch
        y_hat = self.model(x)

        loss = F.cross_entropy(y_hat, y)
        metrics = self._metrics(y_hat, y)

        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        self._log_metrics(
            [
                ("val_accuracy", metrics[0]),
                ("val_precision", metrics[1]),
                ("val_recall", metrics[2]),
                ("val_roc_auc", metrics[3]),
            ]
        )

        return loss

    def test_step(self, batch: Any, batch_idx: int) -> None:
        x, y = batch
        y_hat = self.model(x)

        metrics = self._metrics(y_hat, y)

        self._log_metrics(
            [
                ("test_accuracy", metrics[0]),
                ("test_precision", metrics[1]),
                ("test_recall", metrics[2]),
                ("test_roc_auc", metrics[3]),
            ]
        )

    def configure_optimizers(self) -> Any:
        optimizer = getattr(optim, self.cfg.optimizer.name)(
            self.model.parameters(), **self.cfg.optimizer.params
        )
        return optimizer
