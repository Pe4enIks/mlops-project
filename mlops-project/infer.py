import logging
import os
from pathlib import Path

import hydra
import numpy as np
import onnxruntime
import pytorch_lightning as pl
from omegaconf import DictConfig
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)


@hydra.main(
    config_path=str(Path(__file__).parent / "configs"),
    config_name="main",
    version_base="1.2",
)
def main(cfg: DictConfig):
    expected_workdir = Path(__file__).parent

    os.system("dvc pull")
    pl.seed_everything(cfg.seed)

    transform = transforms.Compose(
        [
            transforms.Resize((cfg.transform.h, cfg.transform.w)),
            transforms.ToTensor(),
            transforms.Normalize(cfg.transform.mean, cfg.transform.std),
        ]
    )

    ort_session = onnxruntime.InferenceSession(
        expected_workdir / cfg.infer.ckpt, providers=["CPUExecutionProvider"]
    )

    img = Image.open(str(expected_workdir / cfg.infer.img)).convert("RGB")
    img = transform(img)
    img = img.unsqueeze(0).numpy().astype(np.float32)

    inputs = {"input.1": img}
    outputs = ort_session.run(None, inputs)

    class_mapping = {0: "cat", 1: "dog"}

    pred_class = class_mapping[np.argmax(outputs[0], axis=1)[0]]

    logger.info(
        f"ckpt: {cfg.infer.ckpt} "
        f"img: {cfg.infer.img} "
        f"predicted class: {pred_class}"
    )


if __name__ == "__main__":
    main()
