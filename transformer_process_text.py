import torch
import time
from PIL import Image
import requests
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
print(torch.version.cuda)
# Load image from the IAM database (actually this model is meant to be used on printed text)
url = 'https://fki.tic.heia-fr.ch/static/img/a01-122-02-00.jpg'
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

# Detect available device correctly
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

t1 = time.perf_counter()

# Load processor and model
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-printed')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-small-printed').to(device)

# Transfer image pixel values to GPU if available
pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

# Generate text on the GPU
generated_ids = model.generate(pixel_values).to(device=device)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(f"Generated text: {generated_text}")
print(f"Time taken: {time.perf_counter() - t1}")

# Sleep to observe GPU usage (optional)
time.sleep(10)