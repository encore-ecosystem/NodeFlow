from examples.cv_dataset_converter.utils.coco_dataset import COCO_Dataset
from examples.cv_dataset_converter.utils.yolo_dataset import YOLO_Dataset

from shapely.geometry.polygon import Polygon

from nodeflow import Adapter
from PIL import Image
from tqdm import tqdm

import numpy as np


class YOLO2COCO_Adapter(Adapter):
    def compute(self, variable: YOLO_Dataset) -> COCO_Dataset:
        assert isinstance(variable, YOLO_Dataset)

        cat_id_to_name_mapping = [
            {"id": idx, "name": cat_name}
            for idx, cat_name in enumerate(variable.data_yaml["names"])
        ]

        coco_anns = {}
        for split in variable.anns:
            coco_anns[split] = {
                'images': [],
                'annotations': [],
                'categories': cat_id_to_name_mapping
            }
            image_id, annotation_id = 0, 0
            for im_name, im_path in tqdm(variable.images[split].items()):
                image = np.array(Image.open(im_path))
                height, width, _ = image.shape
                image_info = {
                    "id"        : image_id,
                    "file_name" : im_name,
                    "width"     : width,
                    "height"    : height,
                }
                coco_anns[split]["images"].append(image_info)

                for txt_label, line in variable.anns[split].items():
                    mapped_line = list(map(float, line[0].split()))
                    if len(mapped_line) == 5:
                        class_id, x_center, y_center, width, height = mapped_line

                        x_min = int((x_center - width / 2) * image_info["width"])
                        y_min = int((y_center - height / 2) * image_info["height"])
                        bbox_width = int(width * image_info["width"])
                        bbox_height = int(height * image_info["height"])

                        annotation = {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": int(class_id),
                            "bbox": [x_min, y_min, bbox_width, bbox_height],
                            "area": bbox_width * bbox_height,
                            "iscrowd": 0,
                        }
                        coco_anns[split]["annotations"].append(annotation)
                        annotation_id += 1
                    elif len(mapped_line) > 5:
                        class_id = mapped_line[0]
                        converted_segmentation = []
                        for i, coord in enumerate(mapped_line[1:]):
                            converted_segmentation.append(
                                coord * width) if i % 2 == 1 else converted_segmentation.append(coord * height)

                        annotation = {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": int(class_id),
                            "segmentation": [converted_segmentation],
                            "area": Polygon(zip(converted_segmentation[0::2], converted_segmentation[1::2])).area,
                            "iscrowd": 0,
                        }
                        coco_anns[split]["annotations"].append(annotation)
                        annotation_id += 1

                image_id += 1

        return COCO_Dataset(anns=coco_anns, images=variable.images, path=None)

    def is_loses_information(self) -> bool:
        return True


__all__ = [
    'YOLO2COCO_Adapter'
]
