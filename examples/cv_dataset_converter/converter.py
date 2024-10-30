from examples.cv_dataset_converter.utils.adapters import *
from examples.cv_dataset_converter.utils.functions import *
from examples.cv_dataset_converter.utils.variables import *

from nodeflow import Converter
import pathlib

ADAPTER_LIST = [
    COCO2YOLO_Adapter(),
    YOLO2COCO_Adapter(),
]


def main():
    converter = Converter(ADAPTER_LIST)

    coco_dataset = converter.convert(
        variable=YOLO_Reader().compute(
            path_to_dataset=PathVariable(
                value=pathlib.Path().resolve() / 'toy_datasets' / 'YOLO'
            )
        ),
        to_type=COCO_Dataset
    )
    COCO_Writer().compute(
        coco_dataset=coco_dataset,
        target_path=PathVariable(
            value=pathlib.Path().resolve() / 'toy_datasets' / 'COCO'
        )
    )

    yolo_dataset = converter.convert(
        variable=COCO_Reader().compute(
            path_to_dataset=PathVariable(
                value=pathlib.Path().resolve() / 'toy_datasets' / 'COCO'
            )
        ),
        to_type=YOLO_Dataset
    )
    YOLO_Writer().compute(
        yolo_dataset=yolo_dataset,
        target_path=PathVariable(
            value=pathlib.Path().resolve() / 'toy_datasets' / 'YOLO'
        )
    )


if __name__ == '__main__':
    main()
