# MLOps Project

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/Pe4enIks/mlops-project/main.svg)](https://results.pre-commit.ci/latest/github/Pe4enIks/mlops-project/main)

## Постановка задачи
Дан набор изображений котов и собак, необходимо разработать систему классифицирующую объект на картинке на два класса - кот, собака.

## Датасет
Датасет состоит из трех частей - обучающей, валидационной и тестовой.
Обучающая часть - 11250 изображений, 5625 котов и 5625 собак.
Валидационная часть - 2500 изображений, 1250 котов и 1250 собак.
Тестовая часть - 12494 изображений, 6251 котов и 6243 собак.

## Модель
Поддерживаются все модели семейства ResNet из модуля torchvision.models.
Последний fc слой заменяется новым, который имеет 2 выхода.
Производится дообучение модели с использованием весов данной модели, обученной на датасете ImageNet.

## Файлы
- train.py - обучение модели, сохранение весов в двух форматах: pickle и onnx, измерение и логирование метрик.
- infer.py - инференс модели с использованием onnxruntime.

## Инструменты
- pytorch lightning.
- poetry.
- pre-commit.
- dvc.
- git.
- hydra.
- mlflow.
