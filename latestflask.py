from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL

from flask import Flask, request, jsonify
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

app = Flask(__name__)

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model(
    "/home/dash/dash/hackoverflow/converted_keras (3)/keras_model.h5", compile=False)

# Load the labels
class_names = open(
    "/home/dash/dash/hackoverflow/converted_keras (3)/labels.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
# data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
# image = Image.open("/home/dash/Downloads/phone.jpeg").convert("RGB")

# resizing the image to be at least 224x224 and then cropping from the center
# size = (224, 224)
# image = ImageOps.fit(image, size, Image.LANCZOS)

# turn the image into a numpy array
# image_array = np.asarray(image)

# Normalize the image
# normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Load the image into the array
# data[0] = normalized_image_array


def preprocess_image(image_path):
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = Image.open(image_path).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array
    return data

# Predicts the model
# prediction = model.predict(data)
# index = np.argmax(prediction)
# class_name = class_names[index]
# confidence_score = prediction[0][index]

# Print prediction and confidence score
# print("Class:", class_name[2:], end="")
# print("Confidence Score:", confidence_score)


# API endpoint for prediction


@app.route('/predict', methods=['POST'])
def predict():
    image_url = request.json.get('image_url')
    if image_url:
        try:
            data = preprocess_image(image_url)
            prediction = model.predict(data)
            index = np.argmax(prediction)
            class_name = class_names[index].strip()
            confidence_score = float(prediction[0][index])
            response = {
                "class_name": class_name,
                "confidence_score": confidence_score
            }
            return jsonify(response)
        except Exception as e:
            return jsonify({"error": "Prediction failed. {}".format(str(e))})
    else:
        return jsonify({"error": "Please provide an 'image_url' in the request body."})


@app.route('/helo', methods=['GET'])
def helo():
    return "helloo"


if __name__ == '__main__':
    app.run(debug=True)
