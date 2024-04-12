import io
import threading
import paddle.vision
import ray
import os
# import boto3
import time
import ray.serve
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
import paddle
from ray import serve
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AutoImageProcessor,TableTransformerModel
from PIL import Image
import torch
logger = logging.getLogger("ray.serve")


class Label(str, Enum):
    EXTRA = "extra"
    TITLE = "title"
    TEXT = "text"
    FORMULA = "formula"
    TABLE = "table"
    LIST = "list"
    FIGURE = "figure"

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


@ray.remote(num_gpus=0.5)
class TransformerOcrprocessor:
    def __init__(self) -> None:
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
        self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
        self.model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed').to(self.device) 
    def process_image(self,images):
        try:
            # image = Image.fromarray(image)
            image_list = [Image.fromarray(image) for image in images]
            # logger.info(image)
            pixel_values = self.processor(images=image_list, return_tensors="pt").pixel_values.to(self.device)
            # logger.info(self.device)
            # logger.info(pixel_values)
            generated_ids = self.model.generate(pixel_values)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            logger.info(generated_text)
            return generated_text
        except Exception as e:
            logger.info(e)
@ray.remote(num_gpus=0.5)
class TransformerTableProcessor:
    def __init__(self) -> None:
        self.device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
        self.image_processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")
        self.model = TableTransformerModel.from_pretrained("microsoft/table-transformer-detection").to(self.device)
    def process_table(self,images):
        image_list = [Image.fromarray(image) for image in images]
        inputs = self.image_processor(images=image_list, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        return outputs

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
        logger.info(result)
        return result



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
        table_list = []
        for idx, block in enumerate(layout_predicted):
            cropped_page = page.crop(
                (block.block.x_1, block.block.y_1, block.block.x_2, block.block.y_2))
            if block.type != Label.EXTRA.value and block.type in (Label.TEXT.value, Label.LIST.value, Label.TITLE.value):
                remaining_list.append(np.array(cropped_page))
            elif block.type==Label.TABLE.value:
                table_list.append(cropped_page)
        return {
            "textlist":remaining_list,
            "tablelist":table_list
        }

@ray.remote
class ProcessActor:
    def __init__(self) -> None:
        self.layout_model = Layoutinfer.remote()
        # self.pool = ActorPool([PaddleOcr.remote() for i in range(1)])
        self.pool = ActorPool([TransformerOcrprocessor.remote()])
    def del_model(self):
        ray.kill(self.layout_model)
    def acquire_model(self):
        self.layout_model  = Layoutinfer.remote()
    def process_url(self):
        url_json = {"slug":"sediment-extractor","link":"https://blr1.vultrobjects.com/patents/202211077651/documents/3-6b53b815709400005c34b69b4ead8a79.pdf"}
        link = url_json.get("link")
        slug = url_json.get("slug")
        pdf = requests.get(link).content
        pdf = convert_from_bytes(pdf, thread_count=20)
        start_time = time.time()
        cor_1 =list(ray.get([self.layout_model.detect.remote(i) for i in pdf]))
        logger.info(cor_1)
        self.del_model()
        page_pred = list(zip(pdf,cor_1))
        results = list(ray.get([get_pdf_text.remote(i) for i in page_pred]))

        remaining_list = list(chain.from_iterable([res.get("textlist") for res in results if res.get("textlist")  is not None]))
        table_list = list(chain.from_iterable([res.get("tablelist") for res in results if res.get("tablelist") is not None]))
        logger.info(table_list)
        result_in_batch = split_into_batches(remaining_list,50)
        t1 = time.time()
        for result in self.pool.map(lambda a,v:a.process_image.remote(v),result_in_batch):
            logger.info(result)
            # print("pass")
        logger.info(f"time take:{time.time()-t1}")
        self.acquire_model()





# p = ProcessActor()
# for i in range(1):
#     p.process_url()
@serve.deployment(num_replicas=1)
class MainActorServe:
    def __init__(self) -> None:
        self.process_actor = ProcessActor.remote()
    def __call__(self, request:Request):
        if request.url.path == "/predict":
            self.process_actor.process_url.remote()
            return "200" 



app: serve.Application = MainActorServe.bind()
serve.run(name="newapp", target=app,
          route_prefix="/", host="0.0.0.0", port=8000)
