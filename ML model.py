import cv2
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np



image_path = 'testdata'

# Load the image
image = cv2.imread(image_path)

# Resize the image
resized_image = cv2.resize(image, (1280, 720))


gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)


cv2.imwrite(image_path, gray_image)


# Load the pre-trained model
model = MobileNetV2(weights='imagenet', include_top=True)

# Load and preprocess the image
img = image.load_img(image_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Extract features
features = model.predict(x)