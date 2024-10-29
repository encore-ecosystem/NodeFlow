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
            path_to_dataset=Path(
                value=pathlib.Path().resolve() / 'toy_datasets' / 'YOLO'
            )
        ),
        to_type=COCO_Dataset
    )

    print(f"coco {coco_dataset.anns}")

    yolo_dataset = converter.convert(
        variable=COCO_Reader().compute(
            path_to_dataset=Path(
                value=pathlib.Path().resolve() / 'toy_datasets' / 'COCO'
            )
        ),
        to_type=YOLO_Dataset
    )

    print(f"coco {yolo_dataset.anns}")


if __name__ == '__main__':
    main()
