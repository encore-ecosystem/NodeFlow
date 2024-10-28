from nodeflow.node import Variable
import pathlib


class Path(Variable):
    def __init__(self, value: pathlib.Path):
        assert isinstance(value, pathlib.Path)
        assert value.exists()
        super().__init__(value)


class YOLO_Dataset(Variable):
    ...

class COCO_Dataset(Variable):
    ...

class JSON_Dataset(Variable):
    ...




__all__ = [
    "Path",

    "YOLO_Dataset",
    "COCO_Dataset",
    "JSON_Dataset"
]