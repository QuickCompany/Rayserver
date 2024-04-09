import os
import cv2
import ray
import numpy as np
import easyocr
import requests
import boto3
from pathlib import Path
import logging

logger = logging.getLogger("ray.serve")


# hostname = "blr1.vultrobjects.com"
# secret_key = "FdkooCERJEiRHR3WK6OuZTGAXJGZh8Kdr3FBKJDh"
# access_key = "36YYH8IHES7BLUSZ3843"

# session = boto3.session.Session()
# client = session.client('s3', **{
#     "region_name": hostname.split('.')[0],
#     "endpoint_url": "https://" + hostname,
#     "aws_access_key_id": access_key,
#     "aws_secret_access_key": secret_key
# })



# bucket_name = 'labelparser-patent-training-images'
# response = client.list_objects(Bucket=bucket_name)
# content = response.get("Contents")
image_path = "/root/Rayserver/rayapproaches/images"

# for file in content:
#     file_name = file.get("Key")
#     print(file_name)
#     response = client.download_file(bucket_name,file_name,os.path.join(image_path,file_name))



def load_easyocr():
    reader = easyocr.Reader(lang_list=["en"],gpu=True)
    return reader



model = load_easyocr()
model_ref = ray.put(model)

@ray.remote(num_gpus=1)
def process_image(model,image_data):
    try:
        # text = model.readtext(np.array(image_data),detail=0,paragraph=True)
        text = model.readtext_batched(image_data,n_height=800,n_width=600)
        return text
    except Exception as e:
        logger.info(e)

# Function to split list into batches
def split_into_batches(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i+batch_size]

# Function to read images using cv.imread
def read_images(image_paths):
    images = []
    for path in image_paths:
        image = np.array(cv2.imread(str(path)))
        images.append(image)
    return images

# List of image paths
image_paths = list(Path("/root/Rayserver/rayapproaches/images").rglob("*.jpg"))[:21]
results = list()
# Splitting into batches
batches = list(split_into_batches(image_paths, 1))

# Reading images in batches
batch_images = []
for batch in batches:
    batch_images.append(read_images(batch))


for img in batch_images:
    results.append(process_image.remote(model_ref,list(img)))


output = ray.get(results)

print(output[0])