import logging
import os

import hydra
import numpy as np
import onnxruntime
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torchvision import transforms

from utils import seed_everything

logger = logging.getLogger(__name__)


@hydra.main(config_path='configs', config_name='main', version_base='1.2')
def main(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg)
    expected_workdir = os.path.dirname(__file__)

    seed = cfg_dict['seed']

    h, w = cfg_dict['transform']['h'], cfg_dict['transform']['w']
    mean = cfg_dict['transform']['mean']
    std = cfg_dict['transform']['std']

    ckpt_path = cfg_dict['infer']['ckpt']
    img_path = cfg_dict['infer']['img']

    seed_everything(seed)

    transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    ort_session = onnxruntime.InferenceSession(
        os.path.join(expected_workdir, ckpt_path),
        providers=['CPUExecutionProvider']
    )

    img = Image.open(os.path.join(expected_workdir, img_path)).convert('RGB')
    img = transform(img)
    img = img.unsqueeze(0).numpy().astype(np.float32)

    inputs = {'image': img}
    outputs = ort_session.run(None, inputs)

    class_mapping = {
        0: 'cat',
        1: 'dog'
    }

    pred_class = class_mapping[np.argmax(outputs[0], axis=1)[0]]

    logger.info(
        f'ckpt: {ckpt_path} '
        f'img: {img_path} '
        f'predicted class: {pred_class}'
    )


if __name__ == '__main__':
    main()
