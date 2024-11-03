from examples.cv_dataset_converter.utils.yolo_dataset import YOLO_Dataset
from nodeflow.builtin import PathVariable

import yaml



def yolo_reader(path_to_dataset: PathVariable) -> YOLO_Dataset:
    # <deprecated>
    # files = list(path_to_dataset.value.glob("*yaml"))
    # # print(path_to_dataset.value)
    # assert len(files) == 1, "Could not find .yaml file"
    # data_yaml_path = files[0]

    data_yaml_path = path_to_dataset / 'data.yaml'
    assert data_yaml_path.exists(), "Could not find .yaml file"

    with open(data_yaml_path, "r") as data_yaml_file:
        data_yaml = yaml.load(data_yaml_file, Loader=yaml.SafeLoader)

    annotations, images = {}, {}
    for split in ["train", "test", "valid"]:
        annotations[split], images[split] = {}, {}

        for label_file_path in (path_to_dataset / split / "labels").glob("*.txt"):
            with open(label_file_path, "r") as label_file:
                annotations[split][label_file_path.name] = label_file.readlines()

        images_directory = path_to_dataset / split / "images"
        for image_file_path in images_directory.iterdir():
            if image_file_path.suffix in ['.jpg', '.jpeg', '.png']:
                images[split][image_file_path.stem] = images_directory / image_file_path

    return YOLO_Dataset(
        path      = path_to_dataset,
        data_yaml = data_yaml,
        anns      = annotations,
        images    = images,
    )


__all__ = [
    'yolo_reader'
]
