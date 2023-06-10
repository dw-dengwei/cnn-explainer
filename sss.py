from flask import Flask, request, jsonify
from PIL import Image
import io

app = Flask(__name__)

@app.route('/get_feature_map', methods=['POST'])
def get_feature_map():
    img_data = request.files['image']
    img_bytes = io.BytesIO(img_data.read())
    im = Image.open(img_bytes)
    width, height = im.size
    print(im.size)
    return jsonify({'width': width, 'height': height})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 
