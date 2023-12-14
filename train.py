import logging
import os

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn, optim
from torch.utils.data import DataLoader
from torchmetrics.classification import (MulticlassAccuracy, MulticlassAUROC,
                                         MulticlassPrecision, MulticlassRecall)
from torchvision import datasets, models, transforms

from utils import seed_everything

logger = logging.getLogger(__name__)


def train(
        model,
        optimizer,
        loss_fn,
        train_dataloader,
        val_dataloader,
        device,
        epochs,
        metrics,
        logger
):
    metric_names = metrics.keys()
    for epoch in range(epochs):
        train_loss, val_loss = [], []
        train_metrics = val_metrics = {mn: [] for mn in metric_names}

        model.train()
        for batch in train_dataloader:
            img, label = batch

            img = img.to(device)
            label = label.to(device)

            logits = model.forward(img)
            loss = loss_fn(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += [loss.item()]

            for metric_name in metric_names:
                train_metrics[metric_name] += [
                    metrics[metric_name](logits, label).item()
                ]

        model.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                img, label = batch

                img = img.to(device)
                label = label.to(device)

                logits = model.forward(img)
                loss = loss_fn(logits, label)

                val_loss += [loss.item()]

                for metric_name in metric_names:
                    val_metrics[metric_name] += [
                        metrics[metric_name](logits, label).item()
                    ]

        train_metric_str, val_metric_str = '', ''
        for metric_name in metric_names:
            train_metric = np.mean(train_metrics[metric_name])
            train_metric_str += f' train {metric_name}: {train_metric:.3f}'
            val_metric = np.mean(val_metrics[metric_name])
            val_metric_str += f' val {metric_name}: {val_metric:.3f}'

        train_loss_epoch = np.mean(train_loss)
        val_loss_epoch = np.mean(val_loss)

        logger.info(
            f'EPOCH: {epoch+1} '
            f'train loss: {train_loss_epoch:.3f} '
            f'{train_metric_str[1:]} '
            f'val loss: {val_loss_epoch:.3f} '
            f'{val_metric_str[1:]} '
        )


def test(
        model,
        dataloader,
        device,
        metrics,
        logger
):
    metric_names = metrics.keys()
    test_metrics = {mn: [] for mn in metric_names}

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            img, label = batch

            img = img.to(device)
            label = label.to(device)

            logits = model.forward(img)

            for metric_name in metric_names:
                test_metrics[metric_name] += [
                    metrics[metric_name](logits, label).item()
                ]

    test_metric_str = ''
    for metric_name in metric_names:
        test_metric = np.mean(test_metrics[metric_name])
        test_metric_str += f' {metric_name}: {test_metric:.3f}'

    logger.info(f'{test_metric_str[1:]}')


@hydra.main(config_path='configs', config_name='main', version_base='1.2')
def main(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg)
    expected_workdir = os.path.dirname(__file__)

    model_name = cfg_dict['model']
    seed = cfg_dict['seed']
    device = cfg_dict['device']
    num_workers = cfg_dict['num_workers']
    epochs = cfg_dict['train']['epochs']
    train_data_path = cfg_dict['train']['dataset']['train']['path']
    val_data_path = cfg_dict['train']['dataset']['val']['path']
    test_data_path = cfg_dict['train']['dataset']['test']['path']
    train_batch_size = cfg_dict['train']['dataset']['train']['batch_size']
    val_batch_size = cfg_dict['train']['dataset']['val']['batch_size']
    test_batch_size = cfg_dict['train']['dataset']['test']['batch_size']
    h, w = cfg_dict['transform']['h'], cfg_dict['transform']['w']
    mean = cfg_dict['transform']['mean']
    std = cfg_dict['transform']['std']
    optimizer_name = cfg_dict['optimizer']['name']
    save_path = cfg_dict['train']['save_path']

    save_path = os.path.join(expected_workdir, save_path)
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, f'{model_name}.onnx')

    seed_everything(seed)
    device = torch.device(device)

    train_transform = transforms.Compose([
        transforms.Resize((int(1.5*h), int(1.5*w))),
        transforms.RandomCrop((h, w)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_dataset = datasets.ImageFolder(
        os.path.join(expected_workdir, train_data_path),
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        os.path.join(expected_workdir, val_data_path),
        transform=train_transform
    )

    test_dataset = datasets.ImageFolder(
        os.path.join(expected_workdir, test_data_path),
        transform=test_transform
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    # если не делать shuffle, то будут батчи из 0 или 1 = неверная метрика
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    # если не делать shuffle, то будут батчи из 0 или 1 = неверная метрика
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    if 'resnet' in model_name:
        model = getattr(models, model_name)(weights='DEFAULT')
        model.fc = nn.Linear(model.fc.in_features, 2)
        model = model.to(device)
    else:
        raise ValueError(f'{model_name} is not available')

    loss_fn = nn.CrossEntropyLoss()
    loss_fn = loss_fn.to(device)

    optimizer = getattr(optim, optimizer_name)(
        model.parameters(),
        **cfg_dict['optimizer']['params']
    )

    acc = MulticlassAccuracy(average='weighted', num_classes=2).to(device)
    prc = MulticlassPrecision(average='weighted', num_classes=2).to(device)
    rec = MulticlassRecall(average='weighted', num_classes=2).to(device)
    auroc = MulticlassAUROC(average='weighted', num_classes=2).to(device)

    metrics = {
        'accuracy': acc,
        'precision': prc,
        'recall': rec,
        'roc-auc': auroc
    }

    train(
        model,
        optimizer,
        loss_fn,
        train_dataloader,
        val_dataloader,
        device,
        epochs,
        metrics,
        logger
    )

    test(
        model,
        test_dataloader,
        device,
        metrics,
        logger
    )

    dummy_input = torch.rand((1, 3, 96, 96), dtype=torch.float32).to(device)
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['image'],
        output_names=['label'],
        dynamic_axes={
            'image': {0: 'batch_size'},
            'label': {0: 'batch_size'}
        }
    )


if __name__ == '__main__':
    main()
