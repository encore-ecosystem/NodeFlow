# The datasets in directory ./input_datasets were obtained from the:
# https://universe.roboflow.com/mnist-bvalq/coco-ki11n
# by user: MNIST
# Please, read README files in datasets directory

from examples.cv_dataset_converter.utils.dataset.coco.adapters.coco2yolo import COCO2YOLO_Adapter
from utils import *
from pathlib import Path
from enum import Enum

from nodeflow import func2node, Dispenser


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

        input_folder = Path(input("Enter Input folder: ")).resolve()
        assert input_folder.exists(), "Input folder does not exist"

        output_folder = Path(input("Enter Output folder: ")).resolve()
        assert output_folder.exists(), "Output folder does not exist"

        match pipeline:
            case ConvertingPipeline.YOLO2COCO:
                Dispenser({
                    'yolo_dataset': PathVariable(input_folder) >> func2node(coco_reader) >> COCO2YOLO_Adapter(),
                    'target_path' : PathVariable(output_folder)
                }) >> func2node(yolo_writer)

            case ConvertingPipeline.COCO2YOLO:
                Dispenser({
                    'yolo_dataset': PathVariable(input_folder) >> func2node(yolo_reader) >> YOLO2COCO_Adapter(),
                    'target_path': PathVariable(output_folder)
                }) >> func2node(coco_writer)

            case _:
                print("Something went wrong... Maybe you don't add new converting pipeline")


if __name__ == '__main__':
    main()
