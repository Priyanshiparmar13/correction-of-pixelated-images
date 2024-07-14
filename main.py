import base64
import io
import numpy as np
from PIL import Image
import tensorflow as tf
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import cv2

# Load models
pixelation_model = tf.keras.models.load_model('models/pixelation_detection_model.h5')
correction_model = tf.keras.models.load_model('models/pixelation_correction_model.h5')

# Path to the input image
input_image_path = 'test/test1.jpg'
output_image_path = 'test/test1_corrected.jpg'

# Load and preprocess the image
image = Image.open(input_image_path).convert('RGB')
image = image.resize((224, 224))  # Resize the image to match the model input size
image_array = np.array(image) / 255.0
image_array = np.expand_dims(image_array, axis=0)

# Check if the image is pixelated
prediction = pixelation_model.predict(image_array)
pixelated = prediction[0] > 0.5  # Use a threshold for binary classification

if pixelated:
    print("The image is pixelated. Correcting the image...")
    
    # Correct the pixelated image
    corrected_image = correction_model.predict(image_array)[0]
    corrected_image = (corrected_image * 255).astype(np.uint8)
    
    # Save the corrected image
    corrected_image_pil = Image.fromarray(corrected_image)
    corrected_image_pil.save(output_image_path)
    
    # Calculate PSNR and SSIM
    original_image = (image_array[0] * 255).astype(np.uint8)
    psnr_value = psnr(original_image, corrected_image)
    ssim_value = ssim(original_image, corrected_image, multichannel=True)
    
    print(f"PSNR: {psnr_value:.2f}")
    print(f"SSIM: {ssim_value:.4f}")
else:
    print("The image is not pixelated.")
