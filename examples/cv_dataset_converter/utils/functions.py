from nodeflow.node import Function
from examples.cv_dataset_converter.utils.variables import Path, YOLO_Dataset, COCO_Dataset


class YOLO_Reader(Function):
    def compute(self, path_to_dataset: Path) -> YOLO_Dataset:
        return YOLO_Dataset(path=path_to_dataset)

class COCO_Reader(Function):
    def compute(self, path_to_dataset: Path) -> COCO_Dataset:
        return COCO_Dataset(path=path_to_dataset)

__all__ = [
    "YOLO_Reader",
    "COCO_Reader"
]
