import requests
from pdf2image import convert_from_bytes
import io
import datetime
import os
from PIL import Image, ImageDraw
import json
import shutil
import base64
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

color_map = {
    'text':   'blue',
    'title':  'red',
    'list':   'pink',
    'table':  'black',
    'figure': 'brown',
    'extra':  'green',
    'formula': (51, 51, 51)
}

def draw_boxes_on_image(image, boxes):
    draw = ImageDraw.Draw(image)
    for idx, box in enumerate(boxes):
        color = color_map[box[0]]
        x1, y1, x2, y2 = float(box[1]), float(box[2]), float(box[3]), float(box[4])
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)
        font_size = 100
        draw.text((x1, y1), str(idx + 1), fill='black', font=None, size=font_size)

def enhance_image(image):
    enhanced_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    enhanced_image = cv2.GaussianBlur(enhanced_image, (5, 5), 0)
    gray_image = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
    thresholded_image = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    return Image.fromarray(cv2.cvtColor(thresholded_image, cv2.COLOR_BGR2RGB))

def crop_boxes_on_image(image, boxes):
    cropped_images = []
    for idx, box in enumerate(boxes):
        label = box[0]
        x1, y1, x2, y2 = float(box[1]), float(box[2]), float(box[3]), float(box[4])
        cropped_image = image.crop((x1, y1, x2, y2))
        cropped_images.append((label, cropped_image))
    return cropped_images

def layout_analysis(image):
    url = 'http://45.76.165.126:8000/layout'
    img_buffer = io.BytesIO()
    image.convert('RGB').save(img_buffer, format='JPEG')
    img_bytes = img_buffer.getvalue()
    image_data = {'data': ('image.jpg', img_bytes)}
    response = requests.post(url, files=image_data)
    return json.loads(response.text)

def ocr_processing(image_bytes, label):
    url = 'http://45.76.165.126:8000/ocr'
    image_data = {'data': ('image.jpg', image_bytes)}
    # print(image_data)
    response = requests.post(url, files=image_data)
    ocr_results = json.loads(response.text)['result']
    print(ocr_results)

    results = [result[-1][0] for result in ocr_results]

    if label.lower() == 'text':
        return '<p>' + " ".join(results) + "</p>"
    
    if label.lower() == 'title':
        return '<h3>' + " ".join(results) + "</h3>"


def main(pdf_url, folder_name):
    # start_time = datetime.datetime.now()
    folder_name = folder_name.replace('/', '_')
    os.makedirs('test', exist_ok=True)
    os.makedirs(f'test/{folder_name}', exist_ok=True)
    try:
        shutil.rmtree('./cropped_images')
    except OSError as e:
        print(f"Error: {e}")
    
    if os.path.isfile('./test.html'):
        os.remove('./test.html')

    os.makedirs('test', exist_ok=True)
    os.makedirs(f'test/{folder_name}', exist_ok=True)

    response = requests.get(pdf_url)
    pdf_bytes = response.content

    with open(f'test/{folder_name}/{folder_name}.pdf','wb') as f:
        f.write(pdf_bytes)

    images = convert_from_bytes(pdf_bytes)
    
    file = open(f'test/{folder_name}/{folder_name}.html', 'w')
    
    table_style = """
    <style>
        table {
            border-collapse: collapse;
            width: 100%;
            border: 1px solid black; /* Add a border to the entire table */
        }

        th, td {
            border: 1px solid black; /* Add a border to table cells */
            padding: 8px;
            text-align: left;
        }

        th {
            background-color: #f2f2f2; /* Light gray background for table header cells */
        }
    </style>
    """
    file.write(table_style)
    
    # with ThreadPoolExecutor() as executor:
    #     for i, image in enumerate(images):
    #         result_boxes = layout_analysis(image)['layout_result'][0]
    #         # print(result_boxes)
    #         cropped_images = crop_boxes_on_image(image, result_boxes)
    #         for idx, (label, cropped_image) in enumerate(cropped_images):
    #             enhanced_cropped_image = enhance_image(cropped_image)
    #             img_buffer = io.BytesIO()
    #             enhanced_cropped_image.convert('RGB').save(img_buffer, format='JPEG')
    #             img_bytes = img_buffer.getvalue()

    #             if label.lower() == 'figure' or label.lower() == 'formula':
    #                 base64_data = base64.b64encode(img_bytes).decode('utf-8')
    #                 html_code = f'<img src="data:image/jpeg;base64,{base64_data}" alt="Image"/>'
    #                 file.write(html_code)
                
    #             elif label.lower() == 'table':
    #                 url = 'http://45.76.165.126:8000/table_ocr'
    #                 image_data = {'data': ('image.jpg', img_bytes),
    #                               'type': label.lower()}
    #                 response = requests.post(url, files=image_data)
    #                 ocr_results = json.loads(response.text)['result']
    #                 if ocr_results:
    #                     file.write(ocr_results[0]['res']['html'])
    
    #             elif label.lower() != 'extra' and label.lower() != 'list':
    #                 result_html = ocr_processing(img_bytes, label)
    #                 file.write(result_html)
    #                 print("pass")

    for i, image in enumerate(images):
        result_boxes = layout_analysis(image)['layout_result'][0]

        draw_boxes_on_image(image, result_boxes)
        # cropped_images = crop_boxes_on_image(image, result_boxes)
        # for idx, (label, cropped_image) in enumerate(cropped_images):
        #     enhanced_cropped_image = enhance_image(cropped_image)
        #     if label.lower() != 'extra':
        #         output_dir = f'./cropped_images/page_{i}/'
        #         os.makedirs(output_dir, exist_ok=True)
        #         output_path = os.path.join(output_dir, f"cropped_{idx}.jpg")
        #         enhanced_cropped_image.save(output_path)

        output_image_path = f"./test/{folder_name}/modified_image_{i}.jpg"
        image.save(output_image_path)
    
    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    print("Elapsed Time:", elapsed_time)

if __name__ == "__main__":

    start_time = datetime.datetime.now()

    # urls = {
    #     '6844/CHE/2014' :'https://blr1.vultrobjects.com/patents/document/151707336/azure_file/65f05027ac46fe63fc5cea0c149961dc.pdf',
    #     '5520/CHE/2012' : 'https://blr1.vultrobjects.com/patents/document/151707475/azure_file/92beaf919e198cd369a05bd97d0d8205.pdf',
    #     '10373/CHENP/2013' : 'https://blr1.vultrobjects.com/patents/document/186428633/azure_file/ab0b8a3cb7d03affd45d37dbf1e79245.pdf',
    #     '202341040989' : 'https://blr1.vultrobjects.com/patents/2023/06/16/21071b38c59ac7e6b712fe2324ad03cd.pdf',
    #     'IN/PCT/2000/258/CHE' : 'https://blr1.vultrobjects.com/patents/document/149452145/azure_file/7090324113bfae4bdca9751c01e4023c.pdf',
    #     'IN/PCT/2000/485/CHE' : 'https://blr1.vultrobjects.com/patents/document/149490947/azure_file/d577deb1d16f78aaa8afa5dcaca34b0e.pdf',
    #     '702/MUM/2000' : 'https://blr1.vultrobjects.com/patents/document/157437883/azure_file/da3276356cc4a15f73dc292f01725a83.pdf',
    #     '1735/CHENP/2003' : 'https://blr1.vultrobjects.com/patents/document/162384596/azure_file/133fd7d81bc9b909091576f31a33eac2.pdf',
    #     '10405/CHENP/2012' : 'https://blr1.vultrobjects.com/patents/document/154075525/azure_file/8fa6dcfe6122164b685d22f06b7ab56e.pdf',
    #     '1880/MUM/2010' : 'https://blr1.vultrobjects.com/patents/document/152360697/azure_file/34c4fdba55ec697aa2518ae642eece16.pdf',
    #     '2706/DEL/2010' : 'https://blr1.vultrobjects.com/patents/document/152598010/azure_file/6a5f3f6c58b20616951554160c9efbf6.pdf',
    #     '202217039268' : "https://blr1.vultrobjects.com/patents/document/184472213/azure_file/1f9d61415d84cc489e03757b7892e281.pdf",
    #     'vikhil_test1' : 'https://blr1.vultrobjects.com/patents/document/161854667/azure_file/734a86a80c081b874bcd674c7ee9ea68.pdf',
    #     'vikhil_test2' : 'https://blr1.vultrobjects.com/patents/document/151708160/azure_file/6ea2dc68dce25eac534daccd4aa85014.pdf',
    #     ##
    #     '201817007231' : 'https://blr1.vultrobjects.com/patents/document/151709659/azure_file/7ca47c88ad5381b7c3c0b865f7812952.pdf',
    #     '201827007094': 'https://blr1.vultrobjects.com/patents/document/151709088/azure_file/99e85783c7b4d7c0c92cb2d28d1b22e6.pdf',
    #     '201827003516' : 'https://blr1.vultrobjects.com/patents/document/151707838/azure_file/1fd481fbed0f8785f8cf0c01f69ddadd.pdf',
    #     '201847000039' : "https://blr1.vultrobjects.com/patents/document/151707388/azure_file/29c832a555f4210e5c34a2d81ac10e0c.pdf",
    #     '201748038672' : 'https://blr1.vultrobjects.com/patents/document/151707744/azure_file/73bf0239775dc2cb9052ffc1efb8dbc5.pdf',
    #     '201747027136' : 'https://blr1.vultrobjects.com/patents/document/151709796/azure_file/229a2e3667f3208ce20f4e0a2fe7ac22.pdf',
    #     '202347031336' : 'https://blr1.vultrobjects.com/patents/2023/05/02/1369df42ef267c6deae5d294f49efa1a.pdf',
    #     '2739/CHENP/2009' : 'https://blr1.vultrobjects.com/patents/document/154812090/azure_file/3c735deb5dccc8f901abb9d4c54fb8d3.pdf',
    #     '541/KOLNP/2009' : 'https://blr1.vultrobjects.com/patents/document/154143108/azure_file/9f2d29b3d7de0b1c940b44f75539375a.pdf'
    # }

    urls = {
    #    '202241054225' :'https://blr1.vultrobjects.com/patents/document/184426465/azure_file/7149f4565c2bc036305c186d1266f119.pdf'
        # '202347032536' : 'https://blr1.vultrobjects.com/patents/2023/05/08/5dcb9720031770be16f1ef0f416b01c6.pdf'
        # '202347032453'  : 'https://blr1.vultrobjects.com/patents/2023/05/08/242fcab2563238e825307a2649022858.pdf'
        '202347032508'   : 'https://blr1.vultrobjects.com/patents/2023/05/08/906827f902e11213967569e36f09692c.pdf',
        # '202347032485'   : 'https://blr1.vultrobjects.com/patents/2023/05/08/bae67b8efa0a64a40e83b41a86046d7e.pdf'
    }
    
    for key , value in urls.items():
        main(value, key)

    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    print("Elapsed Time:", elapsed_time)
    # url = 'https://blr1.vultrobjects.com/patents/2023/05/08/0cdbf1ca1b7d8ff2f3ba1f80a0d95eda.pdf'
    # main(url, 'vikhil')