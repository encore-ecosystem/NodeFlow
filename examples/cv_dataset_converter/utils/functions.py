from examples.cv_dataset_converter.utils.variables import PathVariable, YOLO_Dataset, COCO_Dataset
from nodeflow import Node
from nodeflow.node import Function
import shutil
import yaml
import json


class YOLO_Reader(Function):
    def compute(self, path_to_dataset: PathVariable) -> YOLO_Dataset:
        # validate path
        assert isinstance(path_to_dataset, PathVariable)

        # load .yaml file
        # <deprecated>
        # files = list(path_to_dataset.value.glob("*yaml"))
        # # print(path_to_dataset.value)
        # assert len(files) == 1, "Could not find .yaml file"
        # datayaml_path = files[0]

        datayaml_path = path_to_dataset / 'data.yaml'
        assert datayaml_path.exists(), "Could not find .yaml file"

        with open(datayaml_path, "r") as datayaml_file:
            datayaml = yaml.load(datayaml_file, Loader=yaml.SafeLoader)

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
            path     = path_to_dataset,
            datayaml = datayaml,
            anns     = annotations,
            imgs     = images,
        )

class COCO_Reader(Function):
    def compute(self, path_to_dataset: PathVariable) -> COCO_Dataset:
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

        return COCO_Dataset(path=path_to_dataset, anns=annotations, imgs=images)

class YOLO_Writer(Function):
    def compute(self, yolo_dataset: YOLO_Dataset, target_path: PathVariable) -> Node:
        root = target_path.value

        # create directories
        for split in ["train", "test", "valid"]:
            (root / split / "labels").mkdir(parents=True, exist_ok=True)
            (root / split / "images").mkdir(parents=True, exist_ok=True)

            for image_path in yolo_dataset.imgs[split].values():
                shutil.copy(
                    src = image_path,
                    dst = root / split / 'images' / image_path.name
                )

                with open(root / split / 'labels' / f"{image_path.stem}.txt", 'w') as label_file:
                    label_file.write(yolo_dataset.anns[split][image_path.name])

        with open(root / 'data.yaml', "w") as yaml_file:
            datayaml = {
                'train': yolo_dataset.datayaml['train'],
                'val'  : yolo_dataset.datayaml['val'],
                'test' : yolo_dataset.datayaml['test'],
                'nc'   : yolo_dataset.datayaml['nc'],
                'names': yolo_dataset.datayaml['names'],
            }
            yaml.dump(datayaml, yaml_file)


class COCO_Writer(Function):
    def compute(self, coco_dataset: COCO_Dataset, target_path: PathVariable) -> Node:
        root = target_path.value

        for split in ["train", "test", "valid"]:
            (root / split).mkdir(parents=True, exist_ok=True)

            for image_path in coco_dataset.imgs[split].values():
                shutil.copy(
                    src = image_path,
                    dst = root / split / image_path.name
                )

            with open(root / split / '_annotations.json', 'w') as json_file:
                json.dump(coco_dataset.anns[split], json_file, indent=4)


__all__ = [
    "YOLO_Reader",
    "COCO_Reader",
    "YOLO_Writer",
    "COCO_Writer",
]

