import json

import cv2
import yaml
import pathlib

from nodeflow.node import Variable
from typing import Any, Optional


TTV = ["train", "test", "valid"]
image_file_ext = {".jpg", ".jpeg", ".png"}
label_file_ext = {".txt"}


class Path(Variable):
    def __init__(self, value: pathlib.Path):
        assert isinstance(value, pathlib.Path)
        assert value.exists()
        super().__init__(value)


class YOLO_Dataset(Variable):
    def __init__(self,
                 path: Optional[Path] = None,
                 datayaml: Optional[Any] = None,
                 anns: Optional[dict[str, Any]] = None,
                 imgs: Optional[dict[str, Any]] = None):
        '''
            provide either a path to a dataset or objects representing a dataset
        '''
        if path is not None:
            assert isinstance(path, Path)
            super().__init__(path)

            self.from_folder(path)
        else:
            self.from_objects(datayaml, anns, imgs)

    def from_objects(self, datayaml: Path = None,
                 anns: dict[str, Any] = None,
                 imgs: dict[str, Any] = None):
        self.datayaml = datayaml
        self.anns = anns
        self.imgs = imgs

    def from_folder(self, path: Path):
        # validate path
        assert isinstance(path, Path)
        super().__init__(path)
        self.dataset_path = path

        # load .yaml file
        files = list(path.value.glob("*yaml"))
        print(path.value)
        assert len(files) == 1, "Could not find .yaml file"
        datayaml_path = files[0]

        with open(datayaml_path, "r") as datayaml_file:
            self.datayaml = yaml.load(datayaml_file, Loader=yaml.SafeLoader)

        self.anns = {split: {} for split in TTV}
        self.imgs = {split: {} for split in TTV}
        for split in TTV:
            for label_file_path in list((self.dataset_path.value / split / "labels").glob("*.txt")):
                if label_file_path.suffix in label_file_ext:
                    with open(label_file_path, "r") as label_file:
                        self.anns[split][label_file_path.stem] = label_file.readlines()

            for image_file_path in (self.dataset_path.value / split / "images").iterdir():
                if image_file_path.suffix in image_file_ext:
                    self.imgs[split][image_file_path.stem] = cv2.imread(image_file_path)

        cat_names = self.datayaml["names"]

        self.START_IDX = 0
        self.cat_id_to_name_mapping = [{"id": idx, "name": cat_name} for idx, cat_name in
                                       enumerate(cat_names, start=self.START_IDX)]

class COCO_Dataset(Variable):
    def __init__(self, path: Optional[Path] = None,
                 anns: Optional[dict[dict, Any]] = None,
                 imgs: Optional[dict[dict, Any]] = None):
        '''
            provide either a path to a dataset or objects representing a dataset
        '''

        if path is not None:
            assert isinstance(path, Path)
            super().__init__(path)

            self.from_folder(path)
        else:
            self.from_objects(anns, imgs)

    def from_objects(self, anns: dict[dict, Any], imgs: dict[dict, Any]):
        self.anns = anns    #TODO: validate annotations
        self.imgs = imgs

    def from_folder(self, path: Path):
        assert isinstance(path, Path)
        super().__init__(path)
        self.dataset_path = path

        self.anns = {split: {} for split in TTV}
        self.imgs = {split: {} for split in TTV}
        for split in TTV:
            assert (path.value / split / "_annotations.json").exists()
            with open(self.dataset_path.value / split / "_annotations.json", "r", encoding="utf-8") as f:
                self.anns[split] = json.load(f)

            for image_file_path in (self.dataset_path.value / split).iterdir():
                if image_file_path.suffix in image_file_ext:
                    self.imgs[split][image_file_path.stem] = cv2.imread(image_file_path)

class JSON_Dataset(Variable):
    ...


__all__ = [
    "Path",

    "YOLO_Dataset",
    "COCO_Dataset",
    "JSON_Dataset"
]
