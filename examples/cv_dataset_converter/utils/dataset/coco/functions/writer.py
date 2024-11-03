from examples.cv_dataset_converter.utils.shared import PathVariable, Result
from examples.cv_dataset_converter.utils import COCO_Dataset

import shutil
import json


def coco_writer(coco_dataset: COCO_Dataset, target_path: PathVariable) -> Result:
    root = target_path.value

    for split in ["train", "test", "valid"]:
        (root / split).mkdir(parents=True, exist_ok=True)

        for image_path in coco_dataset.images[split].values():
            shutil.copy(
                src = image_path,
                dst = root / split / image_path.name
            )

        with open(root / split / '_annotations.json', 'w') as json_file:
            json.dump(coco_dataset.anns[split], json_file, indent=4)

    return Result(True)


__all__ = [
    'coco_writer'
]
