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


@ray.remote(num_gpus=0.5)
class Layoutinfer:
    def __init__(self) -> None:
        model_path: str="/root/new_layoutmodel_training/merged_dataset/finetuned-model-18th-march/model_final.pth"
        config_path: str="/root/new_layoutmodel_training/merged_dataset/finetuned-model-18th-march/config.yaml"
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
    start_time = time.time()
    model = Layoutinfer.options(max_concurrency=2).remote()
    # model2 = Layoutinfer.remote()
    # pdf = requests.get("https://blr1.vultrobjects.com/patents/202217062765/documents/2-24cedfd84f9667807b6ee8686ca3df03.pdf").content # 122
    pdf = requests.get("https://blr1.vultrobjects.com/patents/202247067002/documents/4-475075f807f045cf98a0a6b974bcf5d4.pdf").content #141 pages
    # pdf = requests.get("https://blr1.vultrobjects.com/patents/202217005905/documents/2-61492acc6b023adedfe020f11e23c212.pdf").content
    pdf = convert_from_bytes(pdf)
    # pdf_len = len(pdf)
    
    cor_1 = ray.get([model.detect.remote(page) for page in pdf])
    # cor_2 = [model2.detect.remote(page) for page in pdf[pdf_len//2:]]

    # cor = cor_1+cor_2
    # for result in cor:
    #     print(ray.get(cor))

    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(len(cor_1))
    print("Total time taken:", elapsed_time, "seconds")

    print(cor_1[0])
    
    print("\n==============================\n")
    print(cor_1[-1])