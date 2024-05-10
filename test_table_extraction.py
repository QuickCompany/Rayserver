from img2table.document import Image
from img2table.ocr import EasyOCR
from pdf2image import convert_from_path
from io import BytesIO
import numpy as np

ocr = EasyOCR()

pdf = convert_from_path("./b4c7d47cdeffcf2cc6f0873879527f80.pdf")
first_page = np.array(pdf[0]).tobytes()

# first_page.save(open("test.jpg","wb+"))

image = Image(BytesIO(first_page).read())

extracted_tables = image.extract_tables(ocr=ocr,min_confidence=50)

print(extracted_tables)

for table in extracted_tables:
    print(table.html)
    with open("save_data.html","w+") as f:
        f.write(table.html)