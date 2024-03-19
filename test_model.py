import layoutparser as lp
import cv2
import os
import requests
from pdf2image import convert_from_path
from detectron2.config import get_cfg


# model_path = '/root/layout_parsing/finetuned-model/model_final.pth'
model_path = '/home/debo/Rayserver/model/model_final.pth'
config_path = '/home/debo/Rayserver/model/config.yaml'

model = lp.Detectron2LayoutModel(config_path, model_path,
                    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8,],
                    label_map={0: "extra", 1: "title", 2: "text", 3:"formula",
                    4:"table" , 5: "figure" , 6:"list"})
color_map = {
    'text':   (0, 0, 255),     # Red (BGR format)
    'title':  (255, 0, 0),     # Blue (BGR format)
    'list':   (0, 255, 0),     # Green (BGR format)
    'table':  (0, 255, 255),   # Yellow (BGR format)
    'figure': (255, 192, 203), # Pink (BGR format)
    'extra':  (0, 0, 0),       # Black (BGR format)
    'formula': (128, 128, 128) # Grey (BGR format)
}

def image_conversion(pdf_file, output_folder):

    pages = convert_from_path(pdf_file, 500)
    for count, page in enumerate(pages):
        page.save(f'{output_folder}/{output_folder}_{count}.jpg', 'JPEG')

def conversion(app_id , url):
    if not os.path.exists(app_id):
        os.makedirs(app_id)

    pdf = requests.get(url)

    filename = f'./{app_id}/{app_id}.pdf'
    with open(filename, 'wb') as f:
        f.write(pdf.content)

    # Example usage:
    pdf_file = f'./{app_id}/{app_id}.pdf'# Replace with your PDF file name
    output_folder = app_id # Replace with the folder where you want to save the images
    image_conversion(pdf_file, output_folder)

    os.remove(filename)

def main():
    url = 'https://blr1.vultrobjects.com/patents/document/164459125/azure_file/287d166a5c3292e4e904b19a0120e5c5.pdf'
    # url = "https://blr1.vultrobjects.com/patents/document/184426534/azure_file/b4c7d47cdeffcf2cc6f0873879527f80.pdf"
    app_id = 'new_test_20'
    app_id = app_id.replace("/","_")
    conversion(app_id , url)
    
    for i in os.listdir(app_id):
        path = f'./{app_id}/{i}'
        image = cv2.imread(path)
    
        layout_predicted = model.detect(image)
        print(layout_predicted)
        for block in layout_predicted:
            x1, y1, x2, y2 = int(block.block.x_1), int(block.block.y_1), int(block.block.x_2), int(block.block.y_2)
            category = block.type
            color = color_map[category]
            thickness = 2
            cv2.rectangle(image, (x1, y1), (x2, y2), color=color, thickness=thickness)
            text = category  # You can change this to whatever text you want
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_thickness = 4
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            text_x = x1 + 5  # Adjust this value to position the text as desired
            text_y = y1 + text_size[1] + 5  # Adjust this value to position the text as desired
            cv2.putText(image, text, (text_x, text_y), font, font_scale, color=(255, 0, 255), thickness=font_thickness)

        cv2.imwrite(path, image)

main()