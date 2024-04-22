# import json
# from doctr.io import DocumentFile
# from doctr.models import ocr_predictor
# import torch 
# from pdf2image import convert_from_bytes
# import numpy as np

# device = torch.device("cuda:0")
# # PDF
# model = ocr_predictor(pretrained=True).to(device=device)
# pages = convert_from_bytes(open("b4c7d47cdeffcf2cc6f0873879527f80.pdf","rb").read())
# print(pages)
# # multi_img_doc = DocumentFile.from_images(pages)
# result = model([np.asarray(page) for page in pages])

# print(result.export())

# json_data = result.export()
# print(json_data.keys())

# json.dump(json_data,open("save_data.json","w+"))

# for page in json_data.get("pages"):
#     print(page.keys())
#     for block in page.get("blocks"):
#         print(block.keys())
#         for line in block.get("lines"):
#             print(line.keys())
#             print(line.get("words"))

# from typing import List,Dict
# import layoutparser as lp

# class Layoutinfer:
#     def __init__(self) -> None:
#         model_path: str = "/root/Rayserver/model/model_final.pth"
#         config_path: str = "/root/Rayserver/model/config.yaml"
#         extra_config: List = []
#         label_map: Dict[int, str] = {
#             0: "extra",
#             1: "title",
#             2: "text",
#             3: "formula",
#             4: "table",
#             5: "figure",
#             6: "list",
#         }
#         self.model = lp.models.Detectron2LayoutModel(
#             config_path, model_path, extra_config=extra_config, label_map=label_map
#         )

#     def detect(self, image):
#         result = self.model.detect(image)
#         return result


# l = Layoutinfer()


import easyocr
reader = easyocr.Reader(['ch_sim','en']) # this needs to run only once to load the model into memory
result = reader.readtext('chinese.jpg')

print(result)