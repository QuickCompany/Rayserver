from starlette.requests import Request
import ray
import logging
from ray import serve
import layoutparser as lp
from typing import Dict
import requests
import numpy as np
import cv2
from paddleocr import PaddleOCR , PPStructure
from label_studio_sdk.utils import parse_config

ray_serve_logger = logging.getLogger("ray.serve")
@serve.deployment(route_prefix="/",num_replicas=1, ray_actor_options={"num_cpus": 2 ,'num_gpus':1})
class Translator:
    def __init__(self):
        model_path = "/root/layout_parsing/finetuned-model/model_final.pth"
        config_path = "/root/layout_parsing/config.yml"
        
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
    def get_image_from_labelstudio(self,task_id,image_path):
        label_studio_url = "https://labelstudio.debras.in"
        image_url = f'{label_studio_url}/api/tasks/{task_id}{image_path}'
        ray_serve_logger.info(image_url)
        response = requests.get(image_url)
        image_data = response.content
        return image_data
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
                image_data = cv2.imread(self.get_image_from_labelstudio(task_id,image_path))
                layout_predicted = self.model.detect(image_data)
                ray_serve_logger.info(layout_predicted)
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
            return {"layout_result": results}

ray.init(address="auto")
# Create and bind the deployment
translator_app = Translator.bind()

serve.run(target=translator_app, host='0.0.0.0', port=8000)