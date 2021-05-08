from enum import Enum
from typing import *
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as TF


__all__ = ["CollateMode", "CollateFunctionBase", "collate_boxes", "collate_pil_images"]


class CollateMode(Enum):

    MOSAIC = "mosaic"
    TOLIST = "tolist"


def collate_pil_images(
    images: Union[Sequence[Image.Image], Sequence[torch.Tensor]],
    rows: int,
    columns: int,
    atom_size: Optional[Tuple[int, int]] = None,
    mode: str = "RGB",
    to_tensor: bool = False,
) -> Union[Image.Image, torch.Tensor]:

    if atom_size is None:
        atom_size = images[0].size
    else:
        atom_size = tuple(atom_size)

    pil_images = []
    for img in images:
        if isinstance(img, torch.Tensor):
            pil_img = TF.to_pil_img(img)
        elif isinstance(img, Image.Image):
            pil_img = img
        else:
            raise TypeError(
                f"Type of image {type(img)} is not in (`PIL.Image.Image`, `torch.Tensor`)"
            )
        assert pil_img.size == atom_size
        pil_images.append(pil_img.convert(mode))

    w, h = atom_size

    assert rows * columns == len(images)

    W, H = w * columns, h * rows

    img = Image.new(mode, (W, H))
    idx = 0
    for rdx in range(rows):
        for cdx in range(columns):

            img.paste(images[idx], (w * cdx, h * rdx))
            idx += 1

    return TF.to_tensor(img) if to_tensor else img


def collate_boxes(
    boxes: List[torch.Tensor],  # boxes in relative coordinates, float tensors
    rows: int,
    columns: int,
) -> torch.Tensor:

    idx = 0
    for rdx in range(rows):
        for cdx in range(columns):
            boxes[idx][:, 0::2] += cdx
            boxes[idx][:, 1::2] += rdx
            idx += 1

    return torch.cat(boxes)


class CollateFunctionBase(Callable):
    """CollateFunctionBase:
    factory class of collate functions

    params:
        keys: sequence of keywords
        mode: collation mode
        mosaic_row_col: number of rows and number of columns of mosaic collation

    example usage:

        collate_fn = CollateFunctionBase(keys=['image', 'bbox', 'label'], mode='tolist')

        or

        collate_fn = CollateFunctionBase(keys=['image', 'mask', 'bbox', 'label'], mode='mosaic', mosaic_row_col=(3, 3))
    """

    REPR_INDENT = 2
    KEYS = {"image", "mask", "bbox", "label"}

    def __init__(
        self,
        keys: Sequence[str],
        mode: Union[CollateMode, str],
        mosaic_row_col: Optional[Tuple[int, int]] = None,
        to_tensor: bool = True,
    ) -> None:

        for k in keys:
            assert k in self.KEYS
        self.keys = keys
        if isinstance(mode, str):
            mode = CollateMode(mode)
        assert isinstance(mode, CollateMode)
        self.mode = mode
        if mode == CollateMode.MOSAIC:
            assert mosaic_row_col is not None
        self._row_col = mosaic_row_col
        self._to_tensor = to_tensor
        self.batch_size = mosaic_row_col[0] * mosaic_row_col[1]

    def __call__(self, batch: Dict[str, Tuple[Any]]) -> Tuple[Any]:

        outputs = {k: [] for k in self.keys}
        for item in batch:
            for k, v in item.items():
                outputs[k].append(v)

        if self.mode == CollateMode.TOLIST:
            return tuple([outputs[k] for k in self.keys])

        elif self.mode == CollateMode.MOSAIC:
            row, col = self._row_col
            for k in self.keys:
                if k == "image":
                    outputs[k] = collate_pil_images(
                        outputs[k],
                        rows=row,
                        columns=col,
                        mode="RGB",
                        to_tensor=self._to_tensor,
                    )
                elif k == "mask":
                    pil_mask = collate_pil_images(
                        outputs[k], rows=row, columns=col, mode="L", to_tensor=False
                    )
                    if self._to_tensor:
                        m = np.array(pil_mask)
                        outputs[k] = torch.tensor(m).long()
                    else:
                        outputs[k] = pil_mask
                elif k == "bbox":
                    outputs[k] = collate_boxes(outputs[k], rows=row, columns=col)
                elif k == "label":
                    outputs[k] = torch.cat(outputs[k], 0)
                else:
                    raise NotImplementedError(f"Keyword: {k} is not implemented.")

            return tuple([outputs[k] for k in self.keys])
        else:
            raise NotImplementedError(f"Collate mode: {self.mode} is not implemented.")

    def __repr__(self) -> str:
        out = []
        out.append(self.__class__.__name__)
        for k, v in self.__dict__.items():
            out.append(self.REPR_INDENT * " " + f"{k}: {v}")

        return "\n".join(out)
