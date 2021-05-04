from typing import *
from PIL import Image
import torch
import torchvision
import numpy as np
from omegaconf import OmegaConf
from .simulator import generate_detection_image_with_annotation


__all__ = ["DetectionDatasetBase", "DetectionWithCenterMask"]


class DetectionDatasetBase(torch.utils.data.Dataset):
    """
    Params:
        data_src: torch image Dataset
        avg_num: float, mean number of objects per image
        max_iter: int
        poisson: bool, if True randomly sample the number of objects
        config: simulator config
    Output:
        image: Tensor of the image, shape (C, H, W)
        boxes: Tensor, shape (N, 4)
        label: Tensor, shape (N,)
    """

    REPR_INDENT = 2

    def __init__(
        self,
        data_src: Sequence[Tuple[Image.Image, int]],
        avg_num: Union[float, int],
        max_iter: int = 1000,
        poisson: bool = True,
        config: Optional[OmegaConf] = None,
    ) -> None:

        self.max_iter = max_iter
        self.data = data_src
        self.poisson = poisson
        self.num = avg_num
        self.config = config

    def __getitem__(self, idx: int) -> Dict[str, Any]:

        num = np.random.poisson(self.num) if self.poisson else self.num
        num = max(1, num)
        img, box, lab = generate_detection_image_with_annotation(
            self.data, num, self.config
        )

        W, H = img.size

        box = torch.tensor(box).float()
        box[:, 0::2] /= W
        box[:, 1::2] /= H
        lab = torch.tensor(lab).long()

        return {"image": img, "bbox": box, "label": lab}

    def __repr__(self) -> str:
        out = []
        out.append(self.__class__.__name__)
        for k, v in self.__dict__.items():
            out.append(self.REPR_INDENT * " " + f"{k}: {v}")

        return "\n".join(out)

    def __len__(self) -> int:
        return self.max_iter


class DetectionWithCenterMask(DetectionDatasetBase):
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        outputs = super().__getitem__(idx)
        image = outputs["image"]
        bbox = outputs["bbox"]
        label = outputs["label"]
        W, H = image.size
        center = 0.5 * (bbox[:, :2] + bbox[:, 2:])
        center[:, 0] *= W
        center[:, 1] *= H
        center = center.numpy().astype(int)

        mask = np.ones((W, H), dtype=np.uint8) * 255
        for xy, z in zip(center, label):
            mask[xy[1], xy[0]] = z.item()

        mask = Image.fromarray(mask)

        outputs["mask"] = mask

        return outputs
