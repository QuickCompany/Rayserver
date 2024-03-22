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

logger = logging.getLogger("ray.logger")

@ray.remote
def infer(req):
    res = requests.post("http://localhost:8000/post")
    return res.content

@serve.deployment(max_replicas_per_node=3,ray_actor_options={"num_gpus":0.5})
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
    async def __call__(self, request: Request):
        if request.url.path == "/post":
            json_data = await request.body()
            # logger.info(json_data)
            json_data = json.loads(json_data.decode())
            # logger.info(json_data)
            image = Image.open(base64.b64decode((json_data.get("image"))).decode())
            logger.info(image)
            result = self.model.detect(image)
            return result

app = Layoutinfer.bind()
# serve.run(route_prefix="/",host="0.0.0.0")
# if __name__ == "__main__":
#     # app = Layoutinfer.bind()
    
#     pdf = requests.get("https://blr1.vultrobjects.com/patents/document/164459125/azure_file/287d166a5c3292e4e904b19a0120e5c5.pdf").content
#     cor = [infer.remote(json.dumps({
#         "image":base64.b64encode(page.tobytes()).decode("utf-8")
#     })) for page in convert_from_bytes(pdf)]
#     for result in cor:
#         print(ray.get(cor))