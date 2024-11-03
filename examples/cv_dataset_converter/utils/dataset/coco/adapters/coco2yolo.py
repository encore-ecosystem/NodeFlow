from examples.cv_dataset_converter.utils import YOLO_Dataset, COCO_Dataset
from nodeflow import Adapter
from tqdm import tqdm

import yaml


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

        return YOLO_Dataset(data_yaml=yaml_file, anns=yolo_anns, images=variable.images, path=None)

    def is_loses_information(self) -> bool:
        return False


__all__ = [
    'COCO2YOLO_Adapter'
]