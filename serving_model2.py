from starlette.requests import Request
import os
import ray
import boto3
import logging
from ray import serve
import layoutparser as lp
from typing import Dict
import requests
import numpy as np
from flask import jsonify
import cv2
from paddleocr import PaddleOCR , PPStructure
from label_studio_sdk.utils import parse_config
from dotenv import load_dotenv

# load_dotenv("/root/rayserver/.env")



ray_serve_logger = logging.getLogger("ray.serve")
@serve.deployment(route_prefix="/",num_replicas=1, ray_actor_options={"num_cpus": 2 ,'num_gpus':1})
class Translator:
    def __init__(self):
        model_path = "/root/layout_parsing/finetuned-model/model_final.pth"
        config_path = "/root/layout_parsing/config.yml"
        hostname = "blr1.vultrobjects.com"
        secret_key = "dfdfsfsdfsdfsdf"
        access_key = "xdfdfdfdfdf"
        self.labelstudio_api_url = "https://labelstudio.io/api/predictions/"
        self.headers = {
                    "Authorization":"Token fdfdf",
                }
        session = boto3.session.Session()
        self.client = session.client('s3', **{
                                "region_name": hostname.split('.')[0],
                                "endpoint_url": "https://" + hostname,
                                "aws_access_key_id": access_key,
                                "aws_secret_access_key": secret_key
                                })
        self.model = lp.models.Detectron2LayoutModel(
            config_path,
            model_path,
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
            label_map={0: "extra", 1: "title", 2: "text", 3:"formula",
                       4:"table" , 5: "figure" , 6:"list"}
        )

        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')  # Specify the languages you want to support
        
        self.table_engine = PPStructure(lang='en',show_log=True , ocr=True)
        self.table_engine2 = PPStructure(lang='en',layout=False)

    def preprocess(self, data):
        img_array = np.frombuffer(data, dtype=np.uint8)

            # Decode the binary image data using cv2.imdecode
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        return image
    def get_previous_prediction_result_from_labelstudio(self,task_id):
        res = requests.get(self.labelstudio_api_url+f"{task_id}",headers=self.headers)
        if res.status_code == 200:
            ray_serve_logger.info(res.json())
            return res.json()
        else:
            return None
    def get_image_from_s3(self,image_path):
        ray_serve_logger.info(image_path)
        image_name = os.path.basename(image_path)
        response = self.client.get_object(Bucket="labelstudio-patent-images",Key=image_name)
        image_data = response['Body'].read()
        np_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        return image
    async def __call__(self, request:Request):
        if request.url.path == "/health":
            return {
                "status":"ok"
            }
        elif request.url.path == "/setup":
            return {
                "status": "setup done"
            }
        elif request.url.path == "/predict":
            json_data = await request.json()  
            ray_serve_logger.info(json_data)
            predictions = []   
            results = []
            tasks = json_data.get("tasks")
            label_config = parse_config(json_data.get("label_config"))
            ray_serve_logger.info(label_config)
            from_name = list(label_config.items())[0][0]
            ray_serve_logger.info(from_name)
            to_name = label_config.get("label").get("to_name")[0]
            labels = label_config.get("label").get("labels")
            ray_serve_logger.info(to_name)
            ray_serve_logger.info(labels)
            for task in tasks:
                ray_serve_logger.info(task)
                task_id = task.get("id")
                data = task.get("data")
                image_path = data.get("image")
                image_data = self.get_image_from_s3(image_path)
                img_width, img_height = image_data.shape[1], image_data.shape[0]
                layout_predicted = self.model.detect(image_data)
                ray_serve_logger.info(layout_predicted)
                results = list()
                
                for block in layout_predicted:
                    ray_serve_logger.info(block)
                    l = [block.type, block.block.x_1 - 30, block.block.y_1,block.block.x_2 + 10, block.block.y_2]
                    _,x,y,xmax,ymax = l
                    output_label = block.type
                    score = block.score
                    results.append({
                    'from_name': from_name,
                    'to_name': to_name,
                    'type': 'rectanglelabels',
                    'value': {
                        'rectanglelabels': [output_label],
                        'x': float(x) / img_width * 100,
                        'y': float(y) / img_height * 100,
                        'width': (float(xmax) - float(x)) / img_width * 100,
                        'height': (float(ymax) - float(y)) / img_height * 100,
                    },
                    'score': score,
                    })
                    # ray_serve_logger.info(results)
                is_prediction_exist = True if self.get_previous_prediction_result_from_labelstudio(task_id=task_id) is not None else False
                if is_prediction_exist:
                   res = requests.put(self.labelstudio_api_url,headers=self.headers,json={
                    "result":results,
                    "score": score,
                    "model_version":"detectron2",
                    "task":task_id
                    })
                else:
                    res = requests.post(self.labelstudio_api_url,headers=self.headers,json={
                        "result":results,
                        "score": score,
                        "model_version":"detectron2",
                        "task":task_id
                        })
                ray_serve_logger.info(res.content)
                predictions.append({
                    "result":results,
                    "score": score,
                    "model_version":"something"
                    })
                # for field_name, uploaded_file in form_data.items():
                #     # Get the binary content of the uploaded file
                #     binary_data = await uploaded_file.read()

                #     image = self.preprocess(binary_data)
                #     layout_predicted = self.model.detect(image)
                #     layout_predicted.sort(key = lambda layout_predicted:layout_predicted.coordinates[1], inplace=True)
                #     co_ordinates = []
                #     for block in layout_predicted:
                #         l = [block.type, block.block.x_1 - 30, block.block.y_1,block.block.x_2 + 10, block.block.y_2]
                #         co_ordinates.append(l)

                #     results.append(co_ordinates)
                
                # Return the result as a dictionary
            ray_serve_logger.info(predictions)
            return predictions

ray.init(address="auto")
# Create and bind the deployment
translator_app = Translator.bind()

serve.run(target=translator_app, host='0.0.0.0', port=8000)