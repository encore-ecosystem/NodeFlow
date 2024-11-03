from nodeflow import Variable
from nodeflow.builtin.variables import PathVariable
from typing import Optional


class COCO_Dataset(Variable):
    def __init__(
            self,
            path: Optional[PathVariable],
            anns: dict,
            images: dict,
    ):
        """
            provide either a path to a dataset or objects representing a dataset
        """
        super().__init__(value=path)
        self.anns   = anns
        self.images = images


__all__ = [
    'COCO_Dataset',
]
