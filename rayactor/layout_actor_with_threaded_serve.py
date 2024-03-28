import io
import re
import cv2
import ray
import os
import boto3
import time
import json
import requests
import numpy as np
import logging
import layoutparser as lp
import threading
import pytesseract
import cProfile
import asyncio
from pdf2image import convert_from_path, convert_from_bytes
from typing import Dict, List, Tuple, Optional
from paddleocr import PaddleOCR, PPStructure
from enum import Enum
from PIL import Image
from uuid import uuid4
from dotenv import load_dotenv
from urllib.parse import quote_plus
from bs4 import BeautifulSoup
from ray import serve
from starlette.requests import Request
import base64
from ray.util.actor_pool import ActorPool
import easyocr
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor


logger = logging.getLogger("ray.serve")

class Label(str, Enum):
    EXTRA = "extra"
    TITLE = "title"
    TEXT = "text"
    FORMULA = "formula"
    TABLE = "table"
    LIST = "list"
    FIGURE = "figure"

class VultrImageUploader(object):
    def __init__(self) -> None:
        load_dotenv("/home/debo/Rayserver/.env")
        self.hostname = os.getenv("HOST_URL")
        secret_key = os.getenv("VULTR_OBJECT_STORAGE_SECRET_KEY")
        access_key = os.getenv("VULTR_OBJECT_STORAGE_ACCESS_KEY")
        self.figures_bucket = os.getenv("FIGURES_BUCKET")
        session = boto3.session.Session()
        self.client = session.client('s3', **{
            "region_name": self.hostname.split('.')[0],
            "endpoint_url": "https://" + self.hostname,
            "aws_access_key_id": access_key,
            "aws_secret_access_key": secret_key
        })

    def upload_image(self, image: bytes):
        image_name = f"{str(uuid4())}.jpg"
        self.client.upload_fileobj(io.BytesIO(
            image), self.figures_bucket, image_name, ExtraArgs={'ACL': 'public-read'})
        image_url = f"https://{self.hostname}/{self.figures_bucket}/{image_name}"
        return image_url



@ray.remote
class TesseractProcessor:
    def __init__(self) -> None:
        self.tesseract_path = "/usr/bin/tesseract"
        self.custom_config = r'--oem 3 --psm 6 -c tessedit_use_gpu=1'
        pytesseract.pytesseract.tesseract_cmd = self.tesseract_path
    def convert_image_to_text_tessaract(self,image):
        t1 = time.perf_counter()

        process_text = pytesseract.image_to_string(image,config=self.custom_config)
        # process_text = self.easyocr.readtext(np.array(image),detail=0,paragraph=True)[0]
        t2 = time.perf_counter() - t1
        logger.info(f"time took to process tesseract is: {t2}")
        logger.info(process_text)
        return process_text
@ray.remote
class OcrProcessor:
    def __init__(self) -> None:
        self.table_engine = PPStructure(lang='en', show_log=True, ocr=True)
        self.pool = ActorPool([TesseractProcessor.remote() for processor in range(6)])
        # self.easyocr = easyocr.Reader(["en"])
    

    def remove_html_body_tags(self, html_string):
        soup = BeautifulSoup(html_string, 'html.parser')
        # Remove <html> and <body> tags
        if soup.html:
            soup.html.unwrap()
        if soup.body:
            soup.body.unwrap()
        return str(soup)
    def process_layout(self,req: Tuple):
        page,layout = req
        html_string = """"""
        html_string += self.update_html(html_string, page, layout)
        return html_string
    
    def update_html(self, html_code, page, layout_predicted):
        results = self.pool.map(lambda a,v: a.convert_image_to_text_tessaract.remote(v),[page.crop((block.block.x_1, block.block.y_1, block.block.x_2, block.block.y_2)) for block in layout_predicted if block.type not in (Label.TABLE.value,Label.FIGURE.value,Label.FORMULA.value) ])
        logger.info(results)
        for result in results:
            text = result
            html_code += f"<p>{text}</p>"
        return html_code

@ray.remote(num_gpus=0.5,concurrency_groups={"io": 2, "compute": 10})
# @ray.remote(num_gpus=1)
class Layoutinfer:
    def __init__(self) -> None:
        model_path: str="/home/debo/Rayserver/model/model_final.pth"
        config_path: str="/home/debo/Rayserver/model/config.yaml"
        extra_config: List=[]
        label_map: Dict[int, str]={0: "extra", 1: "title", 2: "text", 3: "formula",
                    4: "table", 5: "figure", 6: "list"}
        self.model = lp.Detectron2LayoutModel(config_path, model_path,
                                            extra_config=extra_config,
                                            label_map=label_map)
    @ray.method(concurrency_group="compute")
    async def detect(self, image):
        result = self.model.detect(image)
        return result



@serve.deployment(num_replicas=1)
class LayoutRequest:
    def __init__(self) -> None:
        self.model = Layoutinfer.remote()
        self.pool = ActorPool([self.model])
        self.ocr = OcrProcessor.remote()
        self.ocr_pool = ActorPool([self.ocr])
    async def __call__(self, request:Request):
        if request.url.path == "/patent":
            url_json = await request.json()
            link = url_json.get("link")
            pdf = requests.get(link).content
            pdf = convert_from_bytes(pdf,thread_count=10)
            start_time = time.time()
            cor_1 = list(self.pool.map(lambda a,v: a.detect.remote(v),pdf))
            end_time = time.time()
            elapsed_time = end_time - start_time
            # logger.info(len(cor_1))
            start_time = time.time()
            # html_code = ray.get([self.ocr.process_layout.remote(page,layout) for page,layout in zip(pdf,cor_1)])
            html_code = list(self.ocr_pool.map(lambda a,v: a.process_layout.remote(v),list(zip(pdf,cor_1))))
            end_time = time.time()
            logger.info("Total time taken:", elapsed_time, "seconds")
            html_time = end_time - start_time
            start_time = time.time()
            results = [i for i in html_code]
            end_time = time.time()
            list_time = end_time - start_time
            return {
                "message":"submitted",
                "time":elapsed_time,
                "html_time": html_time,
                "list_time": list_time,
                "result": results
            }


app: serve.Application = LayoutRequest.bind()
serve.run(name="newapp",target=app,route_prefix="/",host="0.0.0.0",port=8000)

# @ray.remote
# def infer(req):
#     res = model.detect(req)
#     return res
# if __name__ == "__main__":
#     # app = Layoutinfer.bind()
    
    
    
#     # model2 = Layoutinfer.remote()
#     # pdf = requests.get("https://blr1.vultrobjects.com/patents/202217062765/documents/2-24cedfd84f9667807b6ee8686ca3df03.pdf").content # 122
#     # pdf = requests.get("https://blr1.vultrobjects.com/patents/202247067002/documents/4-475075f807f045cf98a0a6b974bcf5d4.pdf").content #141 pages
#     pdf = requests.get("https://blr1.vultrobjects.com/patents/202217005905/documents/2-61492acc6b023adedfe020f11e23c212.pdf").content
#     pdf = convert_from_bytes(pdf)
#     pdf_len = len(pdf)
#     model = Layoutinfer.remote()
#     # model = Layoutinfer.options(max_concurrency=pdf_len).remote()
#     pool = ActorPool([model])
#     start_time = time.time()
#     cor_1 = list(pool.map(lambda a,v: a.detect.remote(v),pdf))
#     # cor_1 = ray.get([model.detect.remote(page) for page in pdf])
#     # cor_2 = [model2.detect.remote(page) for page in pdf[pdf_len//2:]]

#     # cor = cor_1+cor_2
#     # for result in cor:
#     #     print(ray.get(cor))    
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     print(len(cor_1))
#     print("Total time taken:", elapsed_time, "seconds")
#     print(cor_1[-1])
#     # print(cor_1[0])
    
#     # print("\n==============================\n")
#     # print(cor_1[-1])