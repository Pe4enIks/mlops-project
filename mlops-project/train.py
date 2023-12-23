import os
import subprocess
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from dataset import CatsDogsDataModule
from model import ClassificationModel
from omegaconf import DictConfig, open_dict
from torchvision import transforms


def get_git_revision_hash() -> str:
    encoded_hash = subprocess.check_output(["git", "rev-parse", "HEAD"])
    decoded_hash = encoded_hash.decode("ascii").strip()
    return decoded_hash


def configure_transforms(
    cfg: DictConfig,
) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((int(1.5 * cfg.transform.h), int(1.5 * cfg.transform.w))),
            transforms.RandomCrop((cfg.transform.h, cfg.transform.w)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(25),
            transforms.ToTensor(),
            transforms.Normalize(cfg.transform.mean, cfg.transform.std),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize((cfg.transform.h, cfg.transform.w)),
            transforms.ToTensor(),
            transforms.Normalize(cfg.transform.mean, cfg.transform.std),
        ]
    )

    return train_transform, test_transform


@hydra.main(
    config_path=str(Path(__file__).parent / "configs"),
    config_name="main",
    version_base="1.2",
)
def main(cfg: DictConfig):
    expected_workdir = Path(__file__).parent
    os.system("dvc pull")

    onnx_save_path = expected_workdir / cfg.train.onnx_save_path
    os.makedirs(str(onnx_save_path), exist_ok=True)
    onnx_save_path = onnx_save_path / f"{cfg.train.experiment_name}.onnx"

    pl.seed_everything(cfg.seed)
    torch.set_float32_matmul_precision("medium")

    train_transform, test_transform = configure_transforms(cfg)

    dm = CatsDogsDataModule(
        train_data_path=expected_workdir / cfg.train.dataset.train.path,
        val_data_path=expected_workdir / cfg.train.dataset.val.path,
        test_data_path=expected_workdir / cfg.train.dataset.test.path,
        train_transform=train_transform,
        test_transform=test_transform,
        train_batch_size=cfg.train.dataset.train.batch_size,
        val_batch_size=cfg.train.dataset.val.batch_size,
        test_batch_size=cfg.train.dataset.test.batch_size,
        num_workers=cfg.num_workers,
    )

    model = ClassificationModel(cfg)

    logger = pl.loggers.MLFlowLogger(
        experiment_name="cats_dogs",
        run_name=cfg.train.experiment_name,
        tracking_uri=cfg.mlflow_server,
    )

    with open_dict(cfg):
        cfg.commit_hash = get_git_revision_hash()
    logger.log_hyperparams(cfg)

    every_n_train_steps = cfg.train.callbacks.model_ckpt.every_n_train_steps
    callbacks = [
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        pl.callbacks.RichModelSummary(
            max_depth=cfg.train.callbacks.model_summary.max_depth
        ),
        pl.callbacks.ModelCheckpoint(
            dirpath=cfg.train.callbacks.model_ckpt.ckpt_path,
            filename=cfg.train.experiment_name,
            monitor="val_loss",
            save_top_k=cfg.train.callbacks.model_ckpt.save_top_k,
            every_n_train_steps=every_n_train_steps,
        ),
    ]

    trainer = pl.Trainer(
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        precision=cfg.precision,
        max_steps=cfg.train.steps,
        log_every_n_steps=cfg.train.loggers.log_every_n_steps,
        enable_checkpointing=True,
        enable_model_summary=True,
        enable_progress_bar=True,
        logger=logger,
        callbacks=callbacks,
    )

    trainer.fit(model=model, datamodule=dm)
    trainer.test(model=model, datamodule=dm)

    model.to_onnx(onnx_save_path, export_params=True)


if __name__ == "__main__":
    main()
