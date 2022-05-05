import base64
import json
import numpy as np
import io
from PIL import Image

from flask import Flask, request
from flask_cors import CORS, cross_origin

from model import predict_text
from ocr2 import InvoiceOCR

app = Flask(__name__)
CORS(app)

cors = CORS(app, resources={"/upload": {"origins": "*"}})

app.config['CORS_HEADERS'] = ['Content-Type', 'Access-Control-Allow-Origin']

@app.route('/')
def home():
    return "<h1>Hello World</h1>"


@app.route('/upload', methods=['POST','GET'])
@cross_origin(origin='*', headers=['Content-Type', 'Access-Control-Allow-Origin'])
def upload_file():
    if request.method == 'POST':

        decoded_data = request.data.decode('utf-8')

        data = json.loads(decoded_data)
        
        image_data = data['base64image'].split('base64,')[-1]
        
        image_name = data['image_name']
        
        input_type = data['input_type']
        
        base64_decoded = base64.b64decode(image_data)
        
        image = Image.open(io.BytesIO(base64_decoded))

        image_np = np.array(image)
        
        # save_path = 'uploaded_images/' + image_name
        # plt.imsave(save_path, image_np)
        
        if input_type == "Handwriting Recognition":

            predicted_text = predict_text(image_np)
        
        else:
            ocr_recognition = InvoiceOCR(image_name,image)
            predicted_text = ocr_recognition.final_output()
            
        
        response = {"predicted_text": predicted_text}
        
        # response = {"data":'predicted_test'}
        print(response)

        return response

if __name__ == '__main__':
   app.run()
