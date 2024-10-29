import json
import cv2
import pathlib

from nodeflow.node import Variable
from typing import Any, Optional
from collections import namedtuple

TTV = ["train", "test", "valid"]
image_file_ext = {".jpg", ".jpeg", ".png"}
label_file_ext = {".txt"}


metaForYolo = namedtuple("metaForYolo", ["anns", "imgs"])

class PathVariable(Variable):
    def __init__(self, value: pathlib.Path):
        assert isinstance(value, pathlib.Path)
        assert value.exists()
        super().__init__(value.resolve())

    def __truediv__(self, other: str) -> pathlib.Path:
        return self.value / other

class YOLO_Dataset(Variable):
    def __init__(
            self,
            path: Optional[PathVariable],
            datayaml: Any,
            anns: dict[str, Any],
            imgs: dict[str, Any],
        ):
        """
            provide either a path to a dataset or objects representing a dataset
        """
        super().__init__(value=path)
        self.datayaml = datayaml
        self.anns     = anns
        self.imgs     = imgs



class COCO_Dataset(Variable):
    def __init__(
            self,
            path: Optional[PathVariable],
            anns: dict,
            imgs: dict,
    ):
        """
            provide either a path to a dataset or objects representing a dataset
        """
        super().__init__(value=path)
        self.anns = anns
        self.imgs = imgs

class JSON_Dataset(Variable):
    ...


__all__ = [
    "PathVariable",

    "YOLO_Dataset",
    "COCO_Dataset",
    "JSON_Dataset",
]
