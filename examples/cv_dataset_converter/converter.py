# The datasets in directory ./input_datasets were obtained from the:
# https://universe.roboflow.com/mnist-bvalq/coco-ki11n
# by user: MNIST
# Please, read README files in datasets directory
import shutil
from enum import Enum
from pathlib import Path

from examples.cv_dataset_converter.utils.coco2yolo import COCO2YOLO_Adapter
from examples.cv_dataset_converter.utils.coco_reader import coco_reader
from examples.cv_dataset_converter.utils.coco_writer import coco_writer
from examples.cv_dataset_converter.utils.yolo2coco import YOLO2COCO_Adapter
from examples.cv_dataset_converter.utils.yolo_reader import yolo_reader
from examples.cv_dataset_converter.utils.yolo_writer import yolo_writer
from nodeflow import func2node, Dispenser
from nodeflow.builtin import PathVariable

CONSOLE_LEN : int = 64


class ConvertingPipeline(str, Enum):
    YOLO2COCO = 'y2c'
    COCO2YOLO = 'c2y'


def main():
    while True:
        print(f"{"[ NodeFlow example for dataset converting ]":=^{CONSOLE_LEN}}")
        print("1) YOLO2COCO")
        print("2) COCO2YOLO")
        print("0) Exit")

        menu = input("Choose converting pipeline: ")
        if not menu.isnumeric():
            print("Invalid input")
            continue

        menu = int(menu)
        match menu:
            case 0:
                print("Bye!")
                break
            case 1:
                print("YOLO2COCO")
                pipeline = ConvertingPipeline.YOLO2COCO
            case 2:
                print("COCO2YOLO")
                pipeline = ConvertingPipeline.COCO2YOLO
            case _:
                print("Invalid input")
                continue

        input_folder  = Path("./input_datasets").resolve()
        output_folder = Path("./output_datasets").resolve()

        assert input_folder.exists()
        output_folder.mkdir(exist_ok=True)

        match pipeline:
            case ConvertingPipeline.YOLO2COCO:
                input_folder  /= "YOLO"
                output_folder /= "COCO"

                assert input_folder.exists(), "Please, download datasets"
                output_folder.exists() and shutil.rmtree(output_folder)
                output_folder.mkdir()

                status = Dispenser(
                    coco_dataset = PathVariable(input_folder) >> func2node(yolo_reader) >> YOLO2COCO_Adapter(),
                    target_path  = PathVariable(output_folder),
                ) >> func2node(coco_writer)

            case ConvertingPipeline.COCO2YOLO:
                input_folder  /= "COCO"
                output_folder /= "YOLO"

                assert input_folder.exists(), "Please, download datasets"
                output_folder.exists() and shutil.rmtree(output_folder)
                output_folder.mkdir()

                status = Dispenser(
                    yolo_dataset = PathVariable(input_folder) >> func2node(coco_reader) >> COCO2YOLO_Adapter(),
                    target_path  = PathVariable(output_folder),
                ) >> func2node(yolo_writer)

            case _:
                print("Something went wrong... Maybe you don't add new converting pipeline")
                continue

        print(f"Status: {status.value}")


if __name__ == '__main__':
    main()
