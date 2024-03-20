import io
import re
import cv2
import os
import boto3
import time
import json
import requests
import numpy as np
import layoutparser as lp
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
        load_dotenv("/root/new_layoutmodel_training/merged_dataset/.env")
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


class LayOutInference(object):
    def __init__(self, model_path: str, config_path: str, extra_config: List, label_map: Dict[int, str]) -> None:
        self.model = lp.Detectron2LayoutModel(config_path, model_path,
                                              extra_config=extra_config,
                                              label_map=label_map)
        # Specify the languages you want to support
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')

        self.table_engine = PPStructure(lang='en', show_log=True, ocr=True)
        self.vultr_img_uploader = VultrImageUploader()

    def remove_html_body_tags(self, html_string):
        soup = BeautifulSoup(html_string, 'html.parser')
        # Remove <html> and <body> tags
        if soup.html:
            soup.html.unwrap()
        if soup.body:
            soup.body.unwrap()
        return str(soup)

    def convert_image_to_byte(self, image):
        byte_io = io.BytesIO()
        image.save(byte_io, format="JPEG")
        cropped_image_bytes = byte_io.getvalue()

        return cropped_image_bytes

    def do_inference(self, pdf_link):
        t1 = time.perf_counter()
        pdf = requests.get(pdf_link).content
        pages = convert_from_bytes(pdf)
        html_code = """"""
        for count, page in enumerate(pages):
            layout_predicted = self.model.detect(page)
            for block in layout_predicted._blocks:
                if block.type == Label.EXTRA.value:
                    pass
                elif block.type == Label.TEXT.value:
                    ocr_results = self.extract_text(page, block)
                    text = self.extract_text_from_paddocr_output(ocr_results)
                    print(text)
                    html_code += f"\n<p>{text}</p>"
                elif block.type == Label.FORMULA.value:
                    print(Label.FORMULA)
                    url = self.vultr_img_uploader.upload_image(self.convert_image_to_byte(
                        page.crop((block.block.x_1, block.block.y_1, block.block.x_2, block.block.y_2))))
                    html_code += f"\n<img src=\"{url}\">"
                elif block.type == Label.TABLE.value:
                    results = self.table_engine(np.array(page.crop(
                        (block.block.x_1, block.block.y_1, block.block.x_2, block.block.y_2))))
                    for table in results:
                        table_html = table['res']['html']
                        preprocessed_table_html = self.remove_html_body_tags(
                            table_html)
                        print(preprocessed_table_html)
                        html_code += f"{preprocessed_table_html}"
                elif block.type == Label.LIST.value:
                    ocr_results = self.extract_text(page, block)
                    text = self.extract_text_from_paddocr_output(ocr_results)
                    print(text)
                    html_code += f"\n<ul>{text}</ul>"
                elif block.type == Label.TITLE.value:
                    print(Label.TITLE)
                    ocr_results = self.extract_text(page, block)
                    text = self.extract_text_from_paddocr_output(ocr_results)
                    print(text)
                    html_code += f"\n<h3>{text}</h3>"
                elif block.type == Label.FIGURE.value:
                    url = self.vultr_img_uploader.upload_image(self.convert_image_to_byte(
                        page.crop((block.block.x_1, block.block.y_1, block.block.x_2, block.block.y_2))))
                    html_code += f"<img src=\"{url}\">"

        end_time = time.perf_counter() - t1
        print(f"time taken: {end_time}")
        return html_code

    def extract_text(self, page, block):
        cropped_img = page.crop(
            (block.block.x_1, block.block.y_1, block.block.x_2, block.block.y_2))
        img_array = np.array(cropped_img)
        # Rearrange color channels for RGB if needed
        img_array = img_array[:, :, ::-1]
        ocr_results = self.ocr.ocr(img_array)
        return ocr_results

    def extract_text_from_paddocr_output(self, output):
        """
        This function extracts text from the specific format of paddleocr output.

        Args:
            output: A list of lists containing bounding boxes and recognized text with confidence scores.

        Returns:
            A string containing the combined recognized text.
        """
        text = ""
        for block in output:
            # Extract coordinates and text from the block
            for inner in block:
                _, text_arr = inner
                in_text, conf = text_arr
                text += in_text
        return text.strip()


@serve.deployment(num_replicas=1,ray_actor_options={"num_cpus": 2, 'num_gpus': 0.5})
class LayoutParserDeployment:
    def __init__(self) -> None:
        model_path = '/root/new_layoutmodel_training/merged_dataset/finetuned-model-18th-march/model_final.pth'
        config_path = '/root/new_layoutmodel_training/merged_dataset/finetuned-model-18th-march/config.yaml'
        extra_config = ["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8,]
        label_map = {0: "extra", 1: "title", 2: "text", 3: "formula",
                        4: "table", 5: "figure", 6: "list"}
        self.model = LayOutInference(model_path, config_path, extra_config, label_map)
    
    async def __call__(self, request:Request ):
        if request.url.path == "/v1/patents":
            json_data = await request.body()
            json_data = json.loads(json_data.decode())
            patents = json_data.get("patents")
            result = list()
            for patent in patents:
                slug = patent.get("slug")
                link = patent.get("link")
                html_code = self.model.do_inference(link)
                result.append(
                    {
                        "slug":slug,
                        "prediction":html_code
                    })
            return result
        else:
            return {
                "img":"save details"
            }


app: serve.Application = LayoutParserDeployment.bind()
serve.run(name="newapp",target=app,route_prefix="/v1",host="0.0.0.0",port=8000)

# if __name__ == "__main__":
    # model_path = '/root/new_layoutmodel_training/merged_dataset/finetuned-model-18th-march/model_final.pth'
    # config_path = '/root/new_layoutmodel_training/merged_dataset/finetuned-model-18th-march/config.yaml'
    # extra_config = ["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8,]
    # label_map = {0: "extra", 1: "title", 2: "text", 3: "formula",
    #                 4: "table", 5: "figure", 6: "list"}
    # model = LayOutInference(model_path, config_path, extra_config, label_map)

    # result = model.do_inference(
    #     'https://blr1.vultrobjects.com/patents/document/184426534/azure_file/b4c7d47cdeffcf2cc6f0873879527f80.pdf')
    # print(result)
    # with open("report.html", "w") as f:
    #     f.write(result)
    

    # 2: Deploy the application locally.
    # serve.run(app,host="0.0.0.0")