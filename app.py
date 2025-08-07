from flask import Flask, render_template, request, jsonify
import numpy as np
import base64
import re
from PIL import Image, ImageOps
from io import BytesIO
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model("mnist_classifier_best.keras")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_data = data['image']

    # Tách base64 string
    image_data = re.sub('^data:image/.+;base64,', '', image_data)
    img = Image.open(BytesIO(base64.b64decode(image_data))).convert('L')
    img = ImageOps.invert(img)  # Đảo ngược màu ảnh
    img = img.resize((28, 28))

    img_array = np.array(img).reshape(1, 28, 28, 1).astype('float32') / 255
    prediction = model.predict(img_array)[0]
    predicted_digit = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    return jsonify({'digit': predicted_digit, 'confidence': round(confidence, 2)})

if __name__ == '__main__':
    app.run(debug=True)
