from examples.cv_dataset_converter.utils.yolo_dataset import YOLO_Dataset
from nodeflow.builtin import PathVariable, Result

import shutil
import yaml


def yolo_writer(yolo_dataset: YOLO_Dataset, target_path: PathVariable) -> Result:
    root = target_path.value

    # create directories
    for split in ["train", "test", "valid"]:
        (root / split / "labels").mkdir(parents=True, exist_ok=True)
        (root / split / "images").mkdir(parents=True, exist_ok=True)

        for image_path in yolo_dataset.images[split].values():
            shutil.copy(
                src = image_path,
                dst = root / split / 'images' / image_path.name
            )

            with open(root / split / 'labels' / f"{image_path.stem}.txt", 'w') as label_file:
                label_file.write(yolo_dataset.anns[split][image_path.stem])

    with open(root / 'data.yaml', "w") as yaml_file:
        data_yaml = {
            'train': yolo_dataset.data_yaml['train'],
            'val'  : yolo_dataset.data_yaml['val'],
            'test' : yolo_dataset.data_yaml['test'],
            'nc'   : yolo_dataset.data_yaml['nc'],
            'names': yolo_dataset.data_yaml['names'],
        }
        yaml.dump(data_yaml, yaml_file)

    return Result(True)


__all__ = [
    'yolo_writer'
]
