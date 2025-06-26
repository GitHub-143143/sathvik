import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
model_path = "Blood_Cell.keras"
model = load_model(model_path)

# Image settings — MUST match the input shape used during model training
img_height, img_width = 128, 128

# Full path to the test image (make sure the extension is correct)
img_path = r"D:\apsche-project\dataset\train images\EOSINOPHIL\_0_207.jpeg"

# Class labels — must match the model's output order
class_labels = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']

try:
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    predicted_class = class_labels[predicted_index]
    confidence = predictions[0][predicted_index]

    # Output result
    print(f"🖼️ Image: {os.path.basename(img_path)}")
    print(f"🔍 Predicted Class: {predicted_class}")
    print(f"📊 Confidence: {confidence:.2f}")

except Exception as e:
    print(f"❌ Error processing image {img_path}: {e}")
