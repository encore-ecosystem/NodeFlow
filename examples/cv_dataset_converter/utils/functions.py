from nodeflow.node import Function
from variables import Path, YOLO_Dataset


class YOLO_Reader(Function):
    def compute(self, path_to_dataset: Path) -> YOLO_Dataset:
        return YOLO_Dataset(value=path_to_dataset)


__all__ = [
    "YOLO_Reader"
]
