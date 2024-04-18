import json
from flask import Flask, request, jsonify
import spacy
import requests
import os
from label_studio_sdk.utils import parse_config
from dotenv import load_dotenv

# Load environment variables (assuming .env file exists)
# load_dotenv("/home/debo/Rayserver/.env")

# Initialize Flask app
app = Flask(__name__)

# Configure logging (optional, you can remove this line)
# logging.basicConfig(level=logging.DEBUG)

# Get Flask app logger
logger = app.logger

# Load spaCy model (replace path with your model location)
model = spacy.load("/home/debo/Rayserver/label-studio-spacy/output/model-best")

# Define Label Studio related variables
labelstudio_api_url = "http://localhost:8081/api/predictions/"
labelstudio_access_token = "7aabfe2d261e2798b1fe6c16e77d79404f5c3025"
headers = {"Authorization": f"Token {labelstudio_access_token}","Content-Type":"application/json"}


def get_previous_prediction(task_id):
    url = f"{labelstudio_api_url}{task_id}"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        return None
def get_prediction_details(task_id):
    res = requests.request("GET",url=f"{labelstudio_api_url}{task_id}",headers=headers)
    print(f"prediction:{res.text}")
    return res.json()

def create_prediction(results, task_id, score):
    print(task_id)
    url = labelstudio_api_url
    is_prediction_exist = get_previous_prediction(task_id)
    print(f"exists:{is_prediction_exist}")
    if is_prediction_exist is not None:
        method = "PUT"
    else:
        method = "POST"
    data = {
        "result": results,
        "score": score,
        "model_version": "spacy",
        "task": task_id
    }
    print(data)
    response = requests.request(method, url, headers=headers, data=json.dumps(data))
    print(response.text)
    return response


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/setup", methods=["POST"])
def setup():
    return jsonify({"status": "setup done"})


@app.route("/predict", methods=["POST"])
def predict():
    if not request.is_json:
        return jsonify({"error": "Invalid request format, requires JSON data"}), 400

    data = request.get_json()
    print(data)
    label_config = parse_config(data.get("label_config"))
    print(label_config)
    
    
    from_name = list(label_config.items())[0][0]
    print(from_name)
    to_name = label_config.get("label").get("to_name")[0]
    print(to_name)
    for task in data.get("tasks", []):
        results = list()
        text = task.get("data", {}).get("text")
        task_id = task.get("id")
        print(f"the task is:{task_id}")
        if not text:
            logger.warning(f"Task {task.get('id')} is missing text data")
            continue

        doc = model(text)
        entities = []
        for e in doc.ents:
            results.append({
                'from_name':from_name,
                'to_name': to_name,
                'type': 'labels',
                'value': {
                    'start': e.start_char,
                    'end': e.end_char,
                    'text': e.text,
                    'labels': [e.label_]
                }
            })
        print(results)
        prediction = {
            "result": results,
            "model_version":"spacy_model"
        }
        create_prediction(results,task_id,score=100)
        # Check if prediction already exists in Label Studio (optional)
        # prediction = get_previous_prediction(task.get("id"))
        # is_prediction_exist = prediction is not None

        # Create prediction in Label Studio (optional)
        # create_prediction(entities, task.get("id"), is_prediction_exist, 1.0)  # Assuming score of 1.0

    return jsonify({"message": "all tasks procesed"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
