from enum import Enum
from typing import *
from PIL import Image
import torch
import torchvision.transforms.functional as TF


class CollateMode(Enum):

    MOSAIC = "mosaic"
    TOLIST = "tolist"


def collate_pil_images(
    images: List[Image.Image],
    rows: int,
    columns: int,
    atom_size: Optional[Tuple[int, int]] = None,
    mode: str = "RGB",
):

    if atom_size is None:
        atom_size = images[0].size

    w, h = atom_size

    assert rows * columns == len(images)

    W, H = w * columns, h * rows

    img = PIL.Image.new(mode, (W, H))
    idx = 0
    for rdx in range(rows):
        for cdx in range(columns):

            img.paste(images[idx], (w * cdx, h * rdx))
            idx += 1

    return img


def collate_boxes(
    boxes: List[torch.Tensor], rows: int, columns: int, atom_size: Tuple[int, int]
):

    idx = 0
    for rdx in range(rows):
        for cdx in range(columns):
            boxes[idx][:, 0::2] += cdx * atom_size[0]
            boxes[idx][:, 1::2] += rdx * atom_size[1]
            idx += 1

    return torch.cat(boxes)


class CollateFunctionBase(Callable):

    REPR_INDENT = 2
    KEYS = {"image", "mask", "bbox", "label"}

    def __init__(
        self,
        keys: Union[Set[str], Sequence[str]],
        mode: Union[CollateMode, str],
        mosaic_row_col: Optional[Tuple[int, int]] = None,
    ) -> None:

        for k in keys:
            assert k in self.KEYS
        self.keys = keys if isinstance(keys, set) else set(keys)
        if isinstance(mode, str):
            mode = CollateMode[mode]
        assert isinstance(mode, CollateMode)
        self.mode = mode
        if mode == CollateMode.MOSAIC:
            assert mosaic_row_col is not None
        self._row_col = mosaic_row_col

    def __call__(self, batch: Dict[str, Tuple[Any]]) -> Tuple[Any]:

        if self.mode == CollateMode.TOLIST:
            outputs = {k: [] for k in self.keys}
            for k, v in batch.items():
                outputs[k].append(v)
            return tuple([v for v in outputs.values()])

        elif self.mode == CollateMode.MOSAIC:
            outputs = {}
            for k in self.keys:
                if k == "image" or k == "mask":
                    outputs[k] = collate_pil_images(batch[k])
                elif k == "bbox":
                    outputs[k] = collate_boxes(batch[k])
                elif k == "label":
                    outputs[k] = torch.stack(batch[k], 0)
                else:
                    raise NotImplementedError(f"Keyword: {k} is not implemented.")

        else:
            raise NotImplementedError(f"Collate mode: {self.mode} is not implemented.")

    def __repr__(self):
        out = []
        out.append(self.__class__.__name__)
        for k, v in self.__class__.__dict__.items():
            out.append(self.REPR_INDENT * " " + f"{k}: {v}")

        return "\n".join(out)
