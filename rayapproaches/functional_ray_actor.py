import datetime
import json
import threading
import ray
import gc
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
from ray import serve
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AutoImageProcessor,TableTransformerModel
from PIL import Image
import torch
from collections import namedtuple

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

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

@ray.remote(num_gpus=0.1,num_cpus=0.1)
class TransformerTableProcessor:
    def __init__(self) -> None:
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
        logger.info(self.device)
        self.image_processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")
        self.table_model = TableTransformerModel.from_pretrained("microsoft/table-transformer-detection")
        self.table_model.to(self.device)
    async def process_table(self,images):
        image_list = [Image.fromarray(image) for image in images]
        inputs = self.image_processor(images=image_list, return_tensors="pt").to(self.device)
        outputs = self.table_model(**inputs)
        results = {"last_hidden_state":outputs["last_hidden_state"].to("cpu"),
                   "encoder_last_hidden_state": outputs["encoder_last_hidden_state"].to("cpu")
                   }
        return results

@ray.remote(num_gpus=0.1,num_cpus=0.1)
class TransformerOcrprocessor:
    def __init__(self) -> None:
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
        logger.info(self.device)
        self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
        self.model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
        self.model.to(self.device)
        # self.image_processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")
        # self.table_model = TableTransformerModel.from_pretrained("microsoft/table-transformer-detection")
        # self.table_model.to(self.device)
    async def process_image(self,images):
        try:
            # image = Image.fromarray(image)
            t1 = time.perf_counter()
            image_list = [Image.fromarray(image) for image in images]
            # logger.info(image)
            pixel_values = self.processor(images=image_list, return_tensors="pt").pixel_values.to(self.device)
            # logger.info(self.device)
            # logger.info(pixel_values)
            generated_ids = self.model.generate(pixel_values)
            del pixel_values
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            del generated_ids
            torch.cuda.empty_cache()
            logger.info(generated_text)
            logger.info(f"processed batch in: {time.perf_counter() - t1}")
            return generated_text
        except Exception as e:
            logger.info(e)
    # async def process_table(self,images):
    #     image_list = [Image.fromarray(image) for image in images]
    #     inputs = self.image_processor(images=image_list, return_tensors="pt").to(self.device)
    #     outputs = self.table_model(**inputs)
    #     results = {"last_hidden_state":outputs["last_hidden_state"].to("cpu"),
    #                "encoder_last_hidden_state": outputs["encoder_last_hidden_state"].to("cpu")
    #                }
    #     return results


@ray.remote(num_gpus=0.1,num_cpus=0.1)
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
        logger.info(result)
        return result



# @ray.remote(num_cpus=0.5)
# def get_pdf_text(req: Tuple):
#         page, layout_predicted = req
#         remaining_list = []
#         table_list = []
#         for idx, block in enumerate(layout_predicted):
#             cropped_page = page.crop(
#                 (block.block.x_1, block.block.y_1, block.block.x_2, block.block.y_2))
#             if block.type != Label.EXTRA.value and block.type in (Label.TEXT.value, Label.LIST.value, Label.TITLE.value):
#                 remaining_list.append(np.array(cropped_page))
#             elif block.type==Label.TABLE.value:
#                 table_list.append(np.array(cropped_page))
#         return {
#             "textlist":remaining_list,
#             "tablelist":table_list
#         }

@ray.remote(num_cpus=0.1)
class ProcessActor:
    def __init__(self) -> None:
        self.layout_model = Layoutinfer.remote()
        self.layout_pool = ActorPool([self.layout_model,Layoutinfer.remote()])
        # self.pool = ActorPool([PaddleOcr.remote() for i in range(1)])
        self.ocr = TransformerOcrprocessor.remote()
        self.pool = ActorPool([self.ocr])
        self.table_ocr = TransformerTableProcessor.remote()
        # self.tableprocessorpool = ActorPool([TransformerTableProcessor.remote()])
    def del_model(self):
        ray.kill(self.layout_model)
    def acquire_model(self):
        self.layout_model  = Layoutinfer.remote()
    def post_to_api(self,data):
        try:
            url = "https://www.quickcompany.in/api/v1/patents"
            res = requests.post(url=url,json=data)
            logger.info(f"res is:{res.text}")
            if res.status_code == 200:
                return {
                    "message":"processed"
                }
            return None
        except Exception as e:
            logger.debug(e)
    def get_pdf_text(self,req: Tuple):
        page, layout_predicted = req
        remaining_list = []
        table_list = []
        for idx, block in enumerate(layout_predicted):
            cropped_page = page.crop(
                (block.block.x_1, block.block.y_1, block.block.x_2, block.block.y_2))
            if block.type != Label.EXTRA.value and block.type in (Label.TEXT.value, Label.LIST.value, Label.TITLE.value):
                remaining_list.append(np.array(cropped_page))
            elif block.type==Label.TABLE.value:
                table_list.append(np.array(cropped_page))
        return {
            "textlist":remaining_list,
            "tablelist":table_list
        }
        # Function to split list into batches
    @staticmethod
    def split_into_batches(lst, batch_size):
        for i in range(0, len(lst), batch_size):
            yield lst[i:i+batch_size]
    def process_url(self,url_json):
        # url_json = {"slug":"sediment-extractor","link":"https://blr1.vultrobjects.com/patents/202211077651/documents/3-6b53b815709400005c34b69b4ead8a79.pdf"}
        link = url_json.get("link")
        slug = url_json.get("slug")
        pdf = requests.get(link).content
        # pdf = open("/root/Rayserver/rayapproaches/pdf24_merged.pdf","rb").read()
        pdf = convert_from_bytes(pdf, thread_count=20)
        start_time = time.time()
        logger.info(f"started at:{str(datetime.datetime.utcnow())}")
        cor_1 = list(self.layout_pool.map(lambda k,v:k.detect.remote(v),pdf))
        # cor_1 =list(ray.get([self.layout_model.detect.remote(i) for i in pdf]))
        # logger.info(cor_1)
        # self.del_model()
        t1 = time.time()
        page_pred = list(zip(pdf,cor_1))
        # results = list(ray.get([get_pdf_text.remote(i) for i in page_pred]))
        results = [self.get_pdf_text(i) for i in page_pred]

        remaining_list = list(chain.from_iterable([res.get("textlist") for res in results if res.get("textlist")  is not None]))
        table_list = list(chain.from_iterable([res.get("tablelist") for res in results if res.get("tablelist") is not None]))
        result_in_batch = self.split_into_batches(remaining_list,90)
        t2 = time.time()
        logger.info(f"time to process lists:{t2 - t1}")
        t1 = time.time()
        html_string = """"""
        for result in self.pool.map(lambda a,v:a.process_image.remote(v),result_in_batch):
            logger.info(result)
            for text in result:
                html_string+=f"<p>{text}</p>"
            # print("pass")
        logger.info(html_string)
        # for result in self.pool.map(lambda a,v:a.process_table.remote(v),table_list):
        #     logger.info(result)
        # self.del_ocr_model()
        # self.acquire_table_pool()
        table_images_in_batch = self.split_into_batches(table_list,batch_size=60)
        table_processes = [self.table_ocr.process_table.remote(table_l) for table_l in table_images_in_batch if len(table_l) != 0]

        result = ray.get(table_processes)
        # for res in result:
        #     print(res)
        # for table_l in table_images_in_batch:
        #     if len(table_list) != 0:
        #         res = ray.get([self.ocr.process_table.remote(table_list)])
        #         logger.info(res)
        # for result in self.pool.map(lambda a,v:a.process_table.remote(v),table_list):
        #     logger.info(result)
        # self.del_table_pool()
        # self.acquire_pool()
        res = self.post_to_api({
            "html":html_string,
            "slug":slug
        })
        end_time = time.time()
        logger.info(f"time take:{time.time()-t1}")
        logger.info(f"actual time took:{end_time - start_time}")
        logger.info(f"finished at:{str(datetime.datetime.utcnow())}")
        # self.acquire_model()
        





# p = ProcessActor()
# data = [
# {"slug":"text","link":"https://blr1.vultrobjects.com/patents/2023/04/30/9c0474371786b9d0362d459e5a021043.pdf"}
# ]
# for i in data:
#     p.process_url(i)
@serve.deployment(num_replicas=1)
class MainActorServe:
    def __init__(self) -> None:
        self.process_actor = ProcessActor.remote()
        # self.process_actor_2 = ProcessActor.remote()
        # self.process_actor_2 = ProcessActor.remote()
        # self.pool = ActorPool([self.process_actor])
    async def __call__(self, request:Request):
        if request.url.path == "/predict":
            url_json = await request.json()
            # self.pool.map_unordered(lambda a,v:a.process_url.remote(v),url_json)
            for data in url_json:
                self.process_actor.process_url.remote(data)
            return "200" 



app: serve.Application = MainActorServe.bind()
serve.run(name="newapp", target=app,
          route_prefix="/", host="0.0.0.0", port=8000)
