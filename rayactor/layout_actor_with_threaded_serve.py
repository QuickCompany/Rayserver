import io
import threading
import ray
import os
import boto3
import time
import requests
import numpy as np
import logging
import layoutparser as lp
import pytesseract
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
import easyocr
from concurrent.futures import ProcessPoolExecutor, as_completed


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
        load_dotenv("/root/Rayserver/.env")
        self.hostname = os.getenv("HOST_URL")
        secret_key = os.getenv("VULTR_OBJECT_STORAGE_SECRET_KEY")
        access_key = os.getenv("VULTR_OBJECT_STORAGE_ACCESS_KEY")
        self.figures_bucket = os.getenv("FIGURES_BUCKET")
        session = boto3.session.Session()
        self.client = session.client(
            "s3",
            **{
                "region_name": self.hostname.split(".")[0],
                "endpoint_url": "https://" + self.hostname,
                "aws_access_key_id": access_key,
                "aws_secret_access_key": secret_key,
            },
        )
    def convert_image_to_byte(self, image):
        byte_io = io.BytesIO()
        image.save(byte_io, format="JPEG")
        cropped_image_bytes = byte_io.getvalue()

        return cropped_image_bytes
    def upload_image(self, image):
        # Define a function to upload image in the background
        def upload(image, image_name):
            image_bytes = self.convert_image_to_byte(image)
            self.client.upload_fileobj(
                io.BytesIO(image_bytes),
                self.figures_bucket,
                image_name,
                ExtraArgs={"ACL": "public-read"},
            )

        # Generate a unique image name
        image_name = f"{str(uuid4())}.jpg"

        # Start a new thread to upload the image
        upload_thread = threading.Thread(target=upload, args=(image, image_name))
        upload_thread.start()

        # Construct the image URL
        image_url = f"https://{self.hostname}/{self.figures_bucket}/{image_name}"

        # Return the image URL
        return image_url

@ray.remote
class TesseractProcessor:
    def __init__(self) -> None:
        self.tesseract_path = "/usr/bin/tesseract"
        self.custom_config = r"--oem 3 --psm 6 -c tessedit_use_gpu=1"
        pytesseract.pytesseract.tesseract_cmd = self.tesseract_path

    def convert_image_to_text(self, image):
        t1 = time.perf_counter()

        process_text = pytesseract.image_to_string(image, config=self.custom_config)
        # process_text = self.easyocr.readtext(np.array(image),detail=0,paragraph=True)[0]
        t2 = time.perf_counter() - t1
        logger.info(f"time took to process tesseract is: {t2}")
        logger.info(process_text)
        return process_text


class HtmlProcessQueue:
    def __init__(self) -> None:
        self.queue = list()

    def add_item(self, item):
        self.queue.append(item)

    def get_work_item(self):
        if self.queue:
            return self.queue.pop(0)
        else:
            return None

    def get_queue(self):
        return self.queue

    def clear_queue(self):
        self.queue.clear()
        return


@ray.remote(num_gpus=0.1)
class EasyOcrProcessor:
    def __init__(self) -> None:
        self.easyocr = easyocr.Reader(["en"],gpu=True)

    def convert_image_to_text(self, data:Tuple):
        image,block_type = data
        t1 = time.perf_counter()
        process_text = self.easyocr.readtext(np.array(image), detail=0, paragraph=True)[
            0
        ]
        t2 = time.perf_counter() - t1
        logger.info(f"time took to process tesseract is: {t2}")
        if block_type == Label.TITLE.value:
            return f"<h2>{process_text}</h2>"
        elif block_type == Label.TEXT.value:
            return f"<p>{process_text}</p>"
        elif block_type == Label.LIST.value:
            logger.info(process_text)
            return f"<ul>{process_text}</ul>"
        else:
            return None


# @ray.remote
# class PaddleTableProcessor:
#     def __init__(self) -> None:
#         self.table_engine = PPStructure(lang='en', layout=False)

#     def remove_html_body_tags(self, html_string):
#         soup = BeautifulSoup(html_string, 'html.parser')
#         # Remove <html> and <body> tags
#         if soup.html:
#             soup.html.unwrap()
#         if soup.body:
#             soup.body.unwrap()
#         return str(soup)
#     def process_table_or_image(self,block,image):
#         if block.type == Label.TABLE.value:
#             res = list(self.table_engine(np.array(image)))
#             html_code = """"""
#             logger.info(res)
#             for table in res:
#                 logger.info(f"this is table data: {table}")
#                 table_html = table['res']['html']
#                 preprocessed_table_html = self.remove_html_body_tags(
#                         table_html)
#                 print(preprocessed_table_html)
#                 html_code += f"{preprocessed_table_html}"
#             return html_code
#         else:
#             html_code = """"""
#             url = self.img_uploader.upload_image(self.convert_image_to_byte(image))
#             html_code += f"<img src=\"{url}\">"
#             return html_code
#     def convert_image_to_byte(self, image):
#         byte_io = io.BytesIO()
#         image.save(byte_io, format="JPEG")
#         cropped_image_bytes = byte_io.getvalue()

#         return cropped_image_bytes


@ray.remote
class OcrProcessor:
    def __init__(self) -> None:
        self.img_uploader = VultrImageUploader()
        self.pool = ActorPool([EasyOcrProcessor.remote(),EasyOcrProcessor.remote()])
    def get_tasks_list(self, req: Tuple):
        page, layout_predicted = req
        html_string = []
        html_code = """"""
        table_figure_formula_list = []
        remaining_list = []

        for idx,block in enumerate(layout_predicted):
            cropped_page = page.crop((block.block.x_1, block.block.y_1, block.block.x_2, block.block.y_2))
            if block.type in (Label.TABLE.value, Label.FIGURE.value, Label.FORMULA.value):
                table_figure_formula_list.append((cropped_page, block.type,idx))
            elif block.type != Label.EXTRA.value and block.type in (Label.TEXT.value,Label.LIST.value,Label.TITLE.value):
                remaining_list.append((cropped_page, block.type))
        

        for text in self.pool.map(
            lambda a, v: a.convert_image_to_text.remote(v),
            [
                block
                for block in remaining_list
            ],
        ):
            html_string.append(text)
        
        for block in table_figure_formula_list:
            cropped_page,block_type,idx = block
            if block_type == Label.TABLE.value:
                pass
            elif block_type == Label.FIGURE.value or block_type == Label.FORMULA.value:
                url = self.img_uploader.upload_image(cropped_page)
                html_string.insert(idx, f'<img class="img-fluid" src="{url}" />')
        for text in html_string:
            html_code+=text
        return html_code
    
    


@ray.remote(num_gpus=0.5, concurrency_groups={"io": 2, "compute": 10})
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


@ray.remote
class MainInferenceActor:
    def __init__(self) -> None:
        self.model = Layoutinfer.remote()
        # self.pool1 = ActorPool([EasyOcrProcessor.remote(),EasyOcrProcessor.remote()])
        # self.pool2 = ActorPool([EasyOcrProcessor.remote()])
        self.pool = ActorPool([self.model])
        # self.work_queue = HtmlProcessQueue()
        self.ocr = OcrProcessor.remote()
        self.queue = Queue()
        # self.ocr1 = OcrProcessor.remote(self.pool2)
        self.api = "https://www.quickcompany.in/api/v1/patents"
        self.ocr_pool = ActorPool([self.ocr,])
    def release_pool(self):
        del self.pool
        del self.model

    def acquire_pool(self):
        self.model = Layoutinfer.remote()
        self.pool = ActorPool([self.model])
    
    def process_pdf(self, url_json):
        link = url_json.get("link")
        pdf = requests.get(link).content
        pdf = convert_from_bytes(pdf, thread_count=10)
        start_time = time.time()
        cor_1 = list(self.pool.map(lambda a, v: a.detect.remote(v), pdf))
        end_time = time.time()
        elapsed_time = end_time - start_time
            # self.release_pool()
            # logger.info(len(cor_1))
        start_time = time.time()
            # html_code = ray.get([self.ocr.process_layout.remote(page,layout) for page,layout in zip(pdf,cor_1)])
        html_code = """"""
        for text in self.ocr_pool.map(
                    lambda a, v: a.get_tasks_list.remote(v), list(zip(pdf, cor_1))
                ):
            html_code+=text
        with open(f"test.txt","w+") as f:
            f.write(html_code)
        res = requests.post(url=self.api+f"?slug=modem-control-using-millimeter-wave-energy-measurement&html={html_code}")
        logger.info(res.content)
        return start_time,elapsed_time,html_code

@serve.deployment(num_replicas=2)
class LayoutRequest:
    def __init__(self) -> None:
        self.actor = MainInferenceActor.remote()
    async def __call__(self, request: Request):
        if request.url.path == "/patent":
            url_json = await request.json()
            for pdf_json in url_json:
                self.actor.process_pdf.remote(pdf_json)
            return {
                "message": "submitted"
            }

    


app: serve.Application = LayoutRequest.bind()
serve.run(name="newapp", target=app, route_prefix="/", host="0.0.0.0", port=8000)


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
