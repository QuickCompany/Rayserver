import os
import spacy
import ray
import logging
import requests
import numpy as np
from ray import serve
from dotenv import load_dotenv
from starlette.requests import Request
from label_studio_sdk.utils import parse_config

ray_serve_logger = logging.getLogger("ray.serve")


@serve.deployment(route_prefix="/", num_replicas=1, ray_actor_options={"num_cpus": 1, 'num_gpus': 0.5})
class Translator:
    def __init__(self):
        load_dotenv("/root/rayserver/.env")
        model_path = ""
        hostname = os.getenv("HOST_URL")
        labelstudio_access_token = os.getenv("LABELSTUDIO_API_TOKEN")
        self.labelstudio_api_url = os.getenv("LABELSTUDIO_API_URL")
        self.headers = {
            "Authorization": f"Token {labelstudio_access_token}",
        }
        self.model = spacy.load("/home/debo/Rayserver/label-studio-spacy/ner_model_server/output/model-best")
        # Specify the languages you want to support

    def get_previous_prediction_result_from_labelstudio(self, task_id):
        res = requests.get(self.labelstudio_api_url +
                           f"{task_id}", headers=self.headers)
        if res.status_code == 200:
            ray_serve_logger.info(res.json())
            return res.json()
        else:
            return None

    def formatted_model_results(self, from_name, to_name, img_height, img_width, layout_predicted_results):
        results = list()
        return results

    def create_prediction(self, results, task_id, is_prediction_exist, score):
        if is_prediction_exist:
            res = requests.put(self.labelstudio_api_url, headers=self.headers, json={
                "result": results,
                "score": score,
                "model_version": "spacy",
                "task": task_id
            })
        else:
            res = requests.post(self.labelstudio_api_url, headers=self.headers, json={
                "result": results,
                "score": score,
                "model_version": "spacy",
                "task": task_id
            })

        return res

    def process_single_task(self, from_name, to_name, task):
        task_id = task.get("id")
        previous_prediction_result = self.get_previous_prediction_result_from_labelstudio(
            task_id=task_id)
        is_prediction_exist = True if previous_prediction_result is not None else False
        data = task.get("data")
        result = None
        return result

    async def __call__(self, request: Request):
        if request.url.path == "/health":
            return {
                "status": "ok"
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
            from_name = list(label_config.items())[0][0]
            ray_serve_logger.info(from_name)
            to_name = label_config.get("label").get("to_name")[0]
            for task in tasks:
                ray_serve_logger.info(task)
                results, task_id, is_prediction_exist, score = self.process_single_task(
                    from_name, to_name, task)
                res = self.create_prediction(
                    results, task_id, is_prediction_exist, score)
                ray_serve_logger.info(res.content)
                predictions.append({
                    "result": results,
                    "score": score,
                    "model_version": "something"
                })
            ray_serve_logger.info(predictions)
            return predictions

        else:
            return {"error": "Invalid endpoint"}


ray.init(address="auto")
# Create and bind the deployment
translator_app = Translator.bind()
serve.run(target=translator_app, host='0.0.0.0', port=8000)
