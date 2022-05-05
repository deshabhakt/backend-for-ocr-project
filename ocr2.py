import os.path
import pytesseract
import re
import nltk
# from ocr1 import Opencv
from pdf2image import convert_from_path
from pytesseract import image_to_string
# from test_new_ocr import image_to_clear

try:
    from PIL import Image
except ImportError:
    import Image


class PdfToImage:

    def __init__(self, file_name):
        self.file_name = file_name

    def convert_pdf_to_img(self):
        return convert_from_path(self.file_name)

    def convert_image_to_text(self, file):
        text = image_to_string(file)
        return text

    def get_text_from_pdf(self):
        images = self.convert_pdf_to_img()
        final_text = ""
        for pg, img in enumerate(images):
            final_text += self.convert_image_to_text(img)

        return final_text


class InvoiceOCR:

    def __init__(self, file_name,file):
        self.file_name = file_name
        self.file = file
    def ocr_core(self):
        name, extension = os.path.splitext(self.file_name)
        if extension == ".pdf":
            pti = PdfToImage(self.file_name)
            text = pti.get_text_from_pdf()
            return text

        else:
        #     Opencv(f"{self.file_name}").crop_an_image()
        #     new_file = "/home/tejpal97/Downloads/new3.jpg"
            # new_file = self.file_name
            # text = pytesseract.image_to_string(Image.open(new_file))
            
            text = pytesseract.image_to_string(self.file)
            return text

    def date_of_order(self, text):
        match = re.findall(r'\d{2}[/.-]\d{2}[/.-]\d{4}', text)
        Date = " "
        Date = Date.join(match)
        return Date

    def vendor_name(self, text):
        text = text.strip()
        title = nltk.sent_tokenize(text)
        head = title[0].splitlines()[0]
        return head

    def listToString(self, string):
        str1 = ""
        return str1.join(string)

    def payable_Amount(self, text):
        pattern = []
        data = (
            'Bill Amount', 'Total Amount', 'Total Payable', 'Gross Amount', 'Total Value', 'Grand Total', 'TOTAL',
            'Total',
            'AMOUNT', 'Amount', 'SUBTOTAL', 'Sub Total')
        n = len(data) - 1
        for i in range(0, n):
            if len(pattern) == 0:
                pattern = re.findall('{}(.*)'.format(data[i]), text)

        if len(pattern) == 0:
            pattern = re.findall(r'\D*(\d*\.\d{2})', text)
        if len(pattern) == 0:
            return None
        else:
            return max(pattern)

    def time_of_order(self, text):
        time1 = re.findall('[0-9]?[0-9]:[0-9][0-9]', text)
        return time1
        # return time1[0]

    def GST_NO(self, text):
        c = (re.findall('\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}[Z]{1}[A-Z\d]{1}', text))
        if not c:
            data = ('GSTIN NO', 'GSTIN', 'GSTN', 'GST NO', 'GST')
            n = len(data) - 1
            for i in range(0, n):
                if not c:
                    c = re.findall('{}(.*)'.format(data[i]), text)

        if not c:
            return None
        c = self.listToString(c)
        return c

    def final_output(self):
        import json
        t = self.ocr_core()
        a = {"Vendor": self.vendor_name(t), "Order Date and Receive Date": self.date_of_order(t),
             "Total Price": self.payable_Amount(t), "Time": self.time_of_order(t), "TAX ID": self.GST_NO(t)}
        b = json.dumps(a)
        return b

if __name__ == "__main__":
    image_file = "/home/tejpal97/Downloads/gray7.jpg"
    ocr_recognition = InvoiceOCR(image_file)
    print(ocr_recognition.final_output())

