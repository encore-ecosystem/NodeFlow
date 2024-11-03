from examples.cv_dataset_converter.utils.coco_dataset import COCO_Dataset
from nodeflow.builtin import PathVariable

import json


def coco_reader(path_to_dataset: PathVariable) -> COCO_Dataset:
    annotations, images = {}, {}
    for split in ["train", "test", "valid"]:
        annotations[split], images[split] = {}, {}

        shrinkage_superclass_mapping = {}
        all_categories_mapping       = {}
        assert (path_to_dataset / split / "_annotations.json").exists()
        with open(path_to_dataset / split / "_annotations.json", "r") as f:
            data = json.load(f)
            for category in data["categories"]:
                all_categories_mapping[category["id"]] = category["name"]
                shrinkage_superclass_mapping[category["name"]] = shrinkage_superclass_mapping.get(category["name"], []) + [category["id"]]

            for category in data["annotations"]:
                category['category_id'] = min(shrinkage_superclass_mapping[all_categories_mapping[category['category_id']]])

            data['categories'] = [
                {
                    'id'            : min(shrinkage_superclass_mapping[category]),
                    'name'          : category,
                    'supercategory' : 'none'
                }
                for category in shrinkage_superclass_mapping
            ]
            annotations[split] = data

        images_directory = path_to_dataset / split
        for image_file_path in images_directory.iterdir():
            if image_file_path.suffix in ['.jpg', '.jpeg', '.png']:
                images[split][image_file_path.stem] = image_file_path / image_file_path

    return COCO_Dataset(path=path_to_dataset, anns=annotations, images=images)


__all__ = [
    'coco_reader',
]
