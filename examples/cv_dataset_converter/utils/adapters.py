from nodeflow.adapters import Adapter
from examples.cv_dataset_converter.utils.variables import YOLO_Dataset, COCO_Dataset, PathVariable
from tqdm import tqdm
from shapely.geometry import Polygon

import yaml


class YOLO2COCO_Adapter(Adapter):
    def convert(self, variable: YOLO_Dataset) -> COCO_Dataset:
        assert isinstance(variable, YOLO_Dataset)

        cat_id_to_name_mapping = [
            {"id": idx, "name": cat_name}
            for idx, cat_name in enumerate(variable.datayaml["names"])
        ]

        coco_anns = {}
        for split in variable.anns:
            coco_anns[split] = {
                'images': [],
                'annotations': [],
                'categories': cat_id_to_name_mapping
            }
            image_id, annotation_id = 0, 0
            for im_name, im_data in tqdm(variable.imgs[split].items()):
                height, width, _ = im_data.shape
                image_info = {
                    "id": image_id,
                    "file_name": im_name,
                    "width": width,
                    "height": height,
                }
                coco_anns[split]["images"].append(image_info)

                for line in variable.anns[split][im_name]:
                    mapped_line = list(map(float, line.split()))
                    if len(mapped_line) == 5:
                        class_id, x_center, y_center, width, height = map(float, line.split())

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

        return COCO_Dataset(anns=coco_anns, imgs=variable.imgs, path=None)

    def is_loses_information(self) -> bool:
        return False


class COCO2YOLO_Adapter(Adapter):
    def convert(self, variable: COCO_Dataset) -> YOLO_Dataset:
        # Output path
        output_base_directory = variable.value.value.parent / "results" / "COCO"

        yolo_anns = {}
        for split in ["train", "valid", "test"]:
            yolo_anns[split] = {}
            category_mapping = {cat["id"]: cat["name"] for cat in variable.anns[split]["categories"]}
            category_id_mapping = {cat["name"]: cat["id"] for cat in variable.anns[split]["categories"]}

            for image in tqdm(variable.anns[split]["images"]):
                image_id = image["id"]
                im_name = image["file_name"]

                for annotation in variable.anns[split]["annotations"]:
                    category_id = category_id_mapping[category_mapping[annotation["category_id"]]]
                    if annotation["image_id"] == image_id:
                        if annotation.get("bbox"):
                            x_center = (annotation["bbox"][0] + annotation["bbox"][2] / 2) / image["width"]
                            y_center = (annotation["bbox"][1] + annotation["bbox"][3] / 2) / image["height"]
                            width = annotation["bbox"][2] / image["width"]
                            height = annotation["bbox"][3] / image["height"]
                            if yolo_anns[split].get(im_name):
                                yolo_anns[split][im_name] += f"{category_id} {x_center} {y_center} {width} {height}\n"
                            else:
                                yolo_anns[split][im_name] = f"{category_id} {x_center} {y_center} {width} {height}\n"

                        elif annotation.get("segmentation"):
                            im_width, im_height = image["width"], image["height"]
                            converted_segmentation = []
                            for i, coord in enumerate(annotation.get("segmentation")[0]):
                                converted_segmentation.append(
                                    coord / im_width) if i % 2 == 0 else converted_segmentation.append(coord / im_height)
                            if yolo_anns[split].get(im_name):
                                yolo_anns[split][im_name] += f"{category_id} {str(converted_segmentation)}\n"
                            else:
                                yolo_anns[split][im_name] = f"{category_id} {str(converted_segmentation)}\n"


        yaml_file = f"path: {str(output_base_directory)}\n"
        yaml_file += 'train: ../train\n'
        yaml_file += 'val: ../valid\n'
        yaml_file += 'test: ../test\n'
        yaml_file += f'nc: {len(category_mapping)}\n'
        yaml_file += f'names: {list(category_mapping.values())}\n'

        yaml_file = yaml.load(yaml_file, Loader=yaml.SafeLoader)

        return YOLO_Dataset(datayaml=yaml_file, anns=yolo_anns, imgs=variable.imgs, path=None)

    def is_loses_information(self) -> bool:
        return False


__all__ = [
    'YOLO2COCO_Adapter',
    'COCO2YOLO_Adapter',
]
