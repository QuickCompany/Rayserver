import io
import cv2
import os
import time
import requests
import numpy as np
import layoutparser as lp
from pdf2image import convert_from_path,convert_from_bytes
from typing import Dict,List,Tuple,Optional
from paddleocr import PaddleOCR,PPStructure
from enum import Enum
from PIL import Image
from uuid import uuid4

class Label(str,Enum):
    EXTRA = "extra"
    TITLE = "title"
    TEXT = "text"
    FORMULA = "formula"
    TABLE = "table"
    LIST = "list"
    FIGURE = "figure"


class LayOutInference(object):
    def __init__(self,model_path:str,config_path:str,extra_config:List,label_map: Dict[int,str]) -> None:
        self.model = lp.Detectron2LayoutModel(config_path, model_path,
                    extra_config=extra_config,
                    label_map=label_map)
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')  # Specify the languages you want to support
        
        self.table_engine = PPStructure(lang='en',show_log=True , ocr=True)
        self.table_engine2 = PPStructure(lang='en',layout=False)
        
    def do_inference(self,pdf_link):
        t1 = time.perf_counter()
        pdf = requests.get(pdf_link).content
        pages = convert_from_bytes(pdf)
        for count,page in enumerate(pages):
            layout_predicted = self.model.detect(page)
            for block in layout_predicted._blocks:
                if block.type == Label.EXTRA.value:
                    print(Label.EXTRA)
                    print(f"skipped")

                elif block.type == Label.TEXT.value:
                    print(Label.TEXT)
                    cropped_img = page.crop((block.block.x_1, block.block.y_1, block.block.x_2, block.block.y_2))
                    cropped_img.save(f"{str(uuid4())}.jpg")
                    if cropped_img:
                        img_array = np.array(cropped_img)
                        img_array = img_array[:, :, ::-1]  # Rearrange color channels for RGB if needed
                        ocr_results = self.ocr.ocr(img_array)
                        for result in ocr_results:
                            print(result)
                elif block.type == Label.FORMULA.value:
                    print(Label.FORMULA)
                elif block.type == Label.TABLE.value:
                    print(Label.TABLE)
                elif block.type == Label.LIST.value:
                    print(Label.LIST)
                elif block.type == Label.TITLE.value:
                    print(Label.TITLE)
                elif block.type == Label.FIGURE.value:
                    print(Label.FIGURE)
                
        end_time = time.perf_counter() - t1
        print(f"time taken: {end_time}")



if __name__ == "__main__":
    model_path = '/home/debo/Rayserver/model/model_final.pth'
    config_path = '/home/debo/Rayserver/model/config.yaml'
    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8,]
    label_map={0: "extra", 1: "title", 2: "text", 3:"formula",
                    4:"table" , 5: "figure" , 6:"list"}
    model = LayOutInference(model_path,config_path,extra_config,label_map)

    result = model.do_inference('https://blr1.vultrobjects.com/patents/document/164459125/azure_file/287d166a5c3292e4e904b19a0120e5c5.pdf')
    print(result)