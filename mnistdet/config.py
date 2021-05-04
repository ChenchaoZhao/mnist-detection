from omegaconf import OmegaConf

__all__ = ["DEFAULT_CONFIG"]
default_config_string = """
DEFAULT:
    canvas: [256, 256]
    margin: [50, 50]
    stride: [10, 10]
    min_size: 16
    max_size: 64
    keep_aspect: True
    rotation: 5
    bgd_contrast: 1.0
    bgd_bridgtness: 1.0
    fgd_contrast: 1.0
    fgd_bridgtness: 1.8
    blur_radius: 1.7
"""

DEFAULT_CONFIG = OmegaConf.create(default_config_string).DEFAULT
