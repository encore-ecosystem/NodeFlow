from nodeflow.adapters import Adapter
from variables import YOLO_Dataset, COCO_Dataset


class YOLO2COCO_Adapter(Adapter):
    def convert(self, variable: YOLO_Dataset) -> COCO_Dataset:
        raise NotImplementedError

    def is_loses_information(self) -> bool:
        return False


class COCO2YOLO_Adapter(Adapter):
    def convert(self, variable: COCO_Dataset) -> YOLO_Dataset:
        raise NotImplementedError

    def is_loses_information(self) -> bool:
        return False


__all__ = [
    'YOLO2COCO_Adapter',
    'COCO2YOLO_Adapter',
]