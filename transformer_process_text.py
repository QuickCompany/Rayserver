from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import requests
import torch
import time

# load image from the IAM database (actually this model is meant to be used on printed text)
url = 'https://fki.tic.heia-fr.ch/static/img/a01-122-02-00.jpg'
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
device = 'cuda:0' if torch.cuda.is_available else 'cpu'
print(device)
t1 = time.perf_counter()
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-printed')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-small-printed').to(device)
model.to(device)
pixel_values = processor(images=image, return_tensors="pt").pixel_values
# print(dir(pixel_values))
generated_ids = model.generate(pixel_values.to(device))
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

# print(generated_text)
print(f"time taken:{time.perf_counter()-t1}")

time.sleep(10)