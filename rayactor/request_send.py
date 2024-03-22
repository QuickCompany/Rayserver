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


@ray.remote(num_gpus=0.1)
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
    def detect(self, image):
        result = self.model.detect(image)
        return result



# @ray.remote
# def infer(req):
#     res = model.detect(req)
#     return res
if __name__ == "__main__":
    # app = Layoutinfer.bind()
    model = Layoutinfer.remote()
    model2 = Layoutinfer.remote()
    pdf = requests.get("https://blr1.vultrobjects.com/patents/document/184426534/azure_file/b4c7d47cdeffcf2cc6f0873879527f80.pdf").content
    pdf = convert_from_bytes(pdf)
    pdf_len = len(pdf)
    start_time = time.time()
    cor_1 = [model.detect.remote(page) for page in pdf[0:pdf_len//2]]
    cor_2 = [model2.detect.remote(page) for page in pdf[pdf_len//2:]]

    cor = cor_1+cor_2
    for result in cor:
        print(ray.get(cor))
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Total time taken:", elapsed_time, "seconds")