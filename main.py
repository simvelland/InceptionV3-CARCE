from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import os
import gdown

app = FastAPI()

GOOGLE_DRIVE_FILE_ID = "1-9jS_gGDi9syUEv__sVF_s9CoowAQy9f"
MODEL_PATH = "InceptionV3_for_brain_tumor_detection.h5"

def download_model():
    if  os.path.exists(MODEL_PATH):

        os.remove(MODEL_PATH)
        
        print("Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)
        print("Download complete!")

        if os.path.exists(MODEL_PATH):
            
            file_size = os.path.getsize(MODEL_PATH)
            print(f"Download complete! File size: {file_size} bytes")
            
            if file_size < 10_000_000:  # Less than 10MB likely means it's a corrupted HTML file
                print("ERROR: The downloaded file is too small. Please check the Google Drive link.")
            else:
                print("ERROR: Model download failed!")

download_model()
model = load_model(MODEL_PATH)  

class_labels = ["Glioma", "Meningioma", "No tumor", "Pituitary"]

def preprocess_image(image):
    img = Image.open(io.BytesIO(image)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = await file.read()
    processed_image = preprocess_image(image)

    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = float(np.max(predictions))

    return {"class": class_labels[predicted_class], "confidence": confidence}
