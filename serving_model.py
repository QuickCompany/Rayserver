from starlette.requests import Request
import ray
from ray import serve
import layoutparser as lp
from typing import Dict
import numpy as np
import cv2
from paddleocr import PaddleOCR , PPStructure

@serve.deployment(route_prefix="/",num_replicas=10, ray_actor_options={"num_cpus": 4 ,'num_gpus':1})
class Translator:
    def __init__(self):
        model_path = "/root/layout_parsing/finetuned-model/model_final.pth"
        config_path = "/root/layout_parsing/config.yml"
        
        self.model = lp.Detectron2LayoutModel(
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
    
    async def __call__(self, request):
        form_data = await request.form()        
        if request.url.path == "/layout":
            results = []
            for field_name, uploaded_file in form_data.items():
                # Get the binary content of the uploaded file
                binary_data = await uploaded_file.read()

                image = self.preprocess(binary_data)
                layout_predicted = self.model.detect(image)
                layout_predicted.sort(key = lambda layout_predicted:layout_predicted.coordinates[1], inplace=True)
                co_ordinates = []
                for block in layout_predicted:
                    l = [block.type, block.block.x_1 - 30, block.block.y_1,block.block.x_2 + 10, block.block.y_2]
                    co_ordinates.append(l)

                results.append(co_ordinates)
            
            # Return the result as a dictionary
            return {"layout_result": results}
        
        elif request.url.path == "/ocr":
            results = {}

            for field_name, uploaded_file in form_data.items():
  
                # Get the binary content of the uploaded file
                binary_data = await uploaded_file.read()
                image = self.preprocess(binary_data)
                result = self.ocr.ocr(image, cls=True)

                results[field_name] = result

            return {'result': results}
        
        elif request.url.path == "/table_ocr":
            results = {}
            for field_name, uploaded_file in form_data.items():
                # Get the binary content of the uploaded file
                binary_data = await uploaded_file.read()

                image = self.preprocess(binary_data)
                # Perform OCR on the image
                result = self.table_engine(image)
                
                
                for line in result:
                    line.pop('img')
                    if line['type'] == 'table' :
                        results[field_name] = line
                    
                    else:
                        result = self.table_engine2(image)
                        for line in result:
                            line.pop('img')
                            results[field_name] = line
                
            return {'result': results}
        
        else:
            return {"error": "Invalid endpoint"}

    # async def __call__(self, request):
    #     # Read binary image data from the request
    #     form_data = await request.form()        
        

    #     for field_name, uploaded_file in form_data.items():
    #         # Get the binary content of the uploaded file
    #         binary_data = await uploaded_file.read()

    #         image = self.preprocess(binary_data)
    #         layout_predicted = self.model.detect(image)
    #         layout_predicted.sort(key = lambda layout_predicted:layout_predicted.coordinates[1], inplace=True)
    #         co_ordinates = []
    #         for block in layout_predicted:
    #             l = [block.type, block.block.x_1 - 30, block.block.y_1,block.block.x_2 + 10, block.block.y_2]
    #             co_ordinates.append(l)


    #     # Return the result as a dictionary
    #     return {"layout_result": co_ordinates}

ray.init(address="auto")
# Create and bind the deployment
translator_app = Translator.bind()

serve.run(target=translator_app, host='0.0.0.0', port=8000)