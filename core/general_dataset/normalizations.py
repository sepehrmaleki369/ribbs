import numpy as np
from core.general_dataset.logger import logger


def normalize_image(image: np.ndarray) -> np.ndarray:
    return image / 255.0 if image.max() > 1.0 else image
