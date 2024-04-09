import json
import time
import requests
import numpy as np
import logging
import layoutparser as lp
from pdf2image import convert_from_bytes
from typing import Dict, List, Tuple

# from paddleocr import PPStructure
from enum import Enum
import easyocr
from PIL import Image
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor,wait,FIRST_COMPLETED
import ray
import cProfile

logger = logging.getLogger("ray.serve")


class Label(str, Enum):
    EXTRA = "extra"
    TITLE = "title"
    TEXT = "text"
    FORMULA = "formula"
    TABLE = "table"
    LIST = "list"
    FIGURE = "figure"

class Layoutinfer:
    def __init__(self) -> None:
        model_path: str = "/root/Rayserver/model/model_final.pth"
        config_path: str = "/root/Rayserver/model/config.yaml"
        extra_config: List = []
        label_map: Dict[int, str] = {
            0: "extra",
            1: "title",
            2: "text",
            3: "formula",
            4: "table",
            5: "figure",
            6: "list",
        }
        self.model = lp.Detectron2LayoutModel(
            config_path, model_path, extra_config=extra_config, label_map=label_map
        )

    def detect(self, image):
        result = self.model.detect(image)
        return result
ocr = easyocr.Reader(["en"], gpu=True)
def perform_ocr(image):
        return ocr.readtext_batched(image, n_height=800, n_width=600, batch_size=30)

def split_list_into_sublists(lst, sublist_size):
    """
    Split a list into sublists of specified size.

    Args:
    lst (list): The list to split.
    sublist_size (int): The size of each sublist.

    Returns:
    list: A list of sublists.
    """
    return [lst[i:i + sublist_size] for i in range(0, len(lst), sublist_size)]
def process_pdf(url_json):
    link = url_json.get("link")
    slug = url_json.get("slug")
    pdf = requests.get(link).content
    pdf = convert_from_bytes(pdf, thread_count=10)
    model = Layoutinfer()
    all_cropped_images = []
    for page in pdf:
        layout_prediction = model.detect(page)
        cropped_images = []
        for block in layout_prediction:
            cropped_page = page.crop(
                (block.block.x_1, block.block.y_1, block.block.x_2, block.block.y_2))
            if block.type != Label.EXTRA.value and block.type in (Label.TEXT.value, Label.LIST.value, Label.TITLE.value):
                cropped_images.append(np.array(cropped_page))
        if len(cropped_images) != 0:
            all_cropped_images.extend(cropped_images)
    # Join cropped images into a single image
    start_time = time.time()
    all_cropped_images = split_list_into_sublists(all_cropped_images,sublist_size=8)
    with ThreadPoolExecutor(max_workers=7) as executor:
        futures = []
        for image in all_cropped_images:
            # Submit the OCR task to the thread pool
            future = executor.submit(perform_ocr, image)
            futures.append(future)
        return futures

        # Wait for all tasks to complete
        # for future in futures:
        #     ocr_results = future.result()
        #     print(ocr_results)
            # Process OCR results here...
        # Print OCR results
        # print(ocr_results)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"processed :{slug} with {len(pdf)} pages in {elapsed_time} seconds.")
    return


def main():
    future_store = list()
    for i in range(5):
        future_store.append(process_pdf({"slug": "sediment-extractor", "link": "https://blr1.vultrobjects.com/patents/202211077695/documents/3-9f1ada133e37520494b70f0282bdfe28.pdf"}))
    for future in future_store:
        for in_f in future:
            print(in_f.result())

if __name__ == "__main__":
    main()