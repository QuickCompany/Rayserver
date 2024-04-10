import io
import threading
import ray
import os
# import boto3
import time
import requests
import numpy as np
import logging
import layoutparser as lp
# import pytesseract
from pdf2image import convert_from_bytes
from typing import Dict, List, Tuple

# from paddleocr import PPStructure
from enum import Enum
from uuid import uuid4
from dotenv import load_dotenv
from urllib.parse import quote_plus
from bs4 import BeautifulSoup
from ray import serve
from starlette.requests import Request
from copy import deepcopy
from ray.util.actor_pool import ActorPool
from ray.util.queue import Queue
# import easyocr
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import chain
from paddleocr import PaddleOCR, PPStructure

logger = logging.getLogger("ray.serve")


class Label(str, Enum):
    EXTRA = "extra"
    TITLE = "title"
    TEXT = "text"
    FORMULA = "formula"
    TABLE = "table"
    LIST = "list"
    FIGURE = "figure"



# def load_easyocr():
#     reader = easyocr.Reader(lang_list=["en"],gpu=True)
#     return reader

def load_layout():
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
    layout_model = lp.Detectron2LayoutModel(
        config_path, model_path, extra_config=extra_config, label_map=label_map
    )
    return layout_model

# model = load_easyocr()
# # layout_model = load_layout()
# # layout_model_ref = ray.put(layout_model)
# model_ref = ray.put(model)

@ray.remote(num_gpus=0.1, concurrency_groups={"io": 2, "compute": 10})
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

    @ray.method(concurrency_group="compute")
    async def detect(self, image):
        result = self.model.detect(image)
        return result

# @ray.remote(num_gpus=0.5)
# class EasyOcr:
#     def __init__(self) -> None:
#         self.model = load_easyocr()
#     async def process_image(self,image_data):
#         try:
#             # text = model.readtext(np.array(image_data),detail=0,paragraph=True)
#             text = self.model.readtext_batched(image_data,n_height=800,n_width=600)
#             return text
#         except Exception as e:
#             logger.info(e)

@ray.remote(num_gpus=0.5)
class PaddleOcr:
    def __init__(self) -> None:
        self.model = PPStructure(lang='en', show_log=True, ocr=True)
    async def process_image(self,image_data):
        try:
            # text = model.readtext(np.array(image_data),detail=0,paragraph=True)
            # text = self.model.readtext_batched(image_data,n_height=800,n_width=600)
            text = self.model(image_data)
            return text
        except Exception as e:
            logger.info(e)

@ray.remote(num_gpus=0.5,num_cpus=0)
def process_image(model_ref,image_data):
    try:
        # text = model.readtext(np.array(image_data),detail=0,paragraph=True)
        text = model_ref.readtext_batched(image_data,n_height=800,n_width=600)
        return text
    except Exception as e:
        logger.info(e)

@ray.remote(num_gpus=0.1)
def detect_image(layout_model,image):
    prediction = layout_model.detect(image)
    return prediction

# Function to split list into batches
def split_into_batches(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i+batch_size]

@ray.remote(num_cpus=1)
def get_pdf_text(req: Tuple):
        page, layout_predicted = req
        remaining_list = []
        for idx, block in enumerate(layout_predicted):
            cropped_page = page.crop(
                (block.block.x_1, block.block.y_1, block.block.x_2, block.block.y_2))
            if block.type != Label.EXTRA.value and block.type in (Label.TEXT.value, Label.LIST.value, Label.TITLE.value):
                remaining_list.append(np.array(cropped_page))
        return remaining_list




url_json = {"slug":"sediment-extractor","link":"https://blr1.vultrobjects.com/patents/202211077651/documents/3-6b53b815709400005c34b69b4ead8a79.pdf"}
link = url_json.get("link")
slug = url_json.get("slug")
pdf = requests.get(link).content
pdf = convert_from_bytes(pdf, thread_count=10)
layout_model = Layoutinfer.remote()
start_time = time.time()
cor_1 =list(ray.get([layout_model.detect.remote(i) for i in pdf]))
logger.info(cor_1)
page_pred = list(zip(pdf,cor_1))

ray.kill(layout_model)
paddocr_model = PaddleOcr.remote()
# easyocr_model = EasyOcr.remote()
# easyocr_model_2 = EasyOcr.remote()

results = list(chain.from_iterable(list(ray.get([get_pdf_text.remote(i) for i in page_pred]))))

print(len(results))
result_in_batch = split_into_batches(results,20)


# pool = ActorPool([easyocr_model])
pool = ActorPool([PaddleOcr.remote() for i in range(1)])
t1 = time.time()

# res = ray.get([process_image.remote(model_ref,i) for i in result_in_batch])
print(f"time take:{time.time()-t1}")


for result in pool.map(lambda a,v:a.process_image.remote(v),results):
    print(result)
# print(f"time take:{time.time()-t1}")
# app: serve.Application = LayoutRequest.bind()
# serve.run(name="newapp", target=app,
#           route_prefix="/", host="0.0.0.0", port=8000)
