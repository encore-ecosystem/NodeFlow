from nodeflow import Variable
from nodeflow.builtin.variables import PathVariable
from typing import Optional, Any


class YOLO_Dataset(Variable):
    def __init__(
            self,
            path: Optional[PathVariable],
            data_yaml: Any,
            anns: dict[str, Any],
            images: dict[str, Any],
        ):
        super().__init__(value=path)
        self.data_yaml = data_yaml
        self.anns      = anns
        self.images    = images


__all__ = [
    'YOLO_Dataset',
]
