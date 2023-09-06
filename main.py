from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import numpy as np
import tensorflow as tf

app = FastAPI()

# Load your TFLite model here
interpreter = tf.lite.Interpreter(model_path="nsfw-float32.tflite")
interpreter.allocate_tensors()

with open("labels.txt", "r") as f:
    labels = f.read().splitlines()

@app.post("/classify/")
async def classify_image(file: UploadFile):
    # Read and preprocess the uploaded image
    image = Image.open(io.BytesIO(await file.read()))
    image = image.resize((299, 299))
    image = np.array(image).astype(np.float32) / 255.0
    image = image.reshape((1, 299, 299, 3))

    # Set input tensor
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]['index'], image)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    output_details = interpreter.get_output_details()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    output_data = output_data[0]
    predicted_class = labels[np.argmax(output_data)]


    # Return classification results as JSON
    return {"class": predicted_class}
