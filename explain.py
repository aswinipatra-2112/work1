import keras
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Model
from keras.models import load_model

# Load your customized model
# Replace 'custom_model.h5' with the path to your model file
model = keras.models.load_model('/kaggle/input/dense-50-epochs-weights/dense_checkpoint_best_0.0429.h5')

num_layers_to_remove = 1
if len(model.layers) > num_layers_to_remove:
    model_layers = model.layers[:-num_layers_to_remove]
else:
    raise ValueError("Number of layers in the model is less than 7. Cannot remove.")

# Create a new model with the desired layers
new_model = Model(inputs = model_layers[0].input, outputs = model_layers[-1].output)
new_model.save('/kaggle/working/modified_model.h5')
#new_model.summary()

model = keras.models.load_model('/kaggle/working/modified_model.h5')

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Unable to read image from path: {image_path}")
    img = cv2.resize(img, (224, 224))  # Resize to match the input size of your model
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = img.astype(np.float32) / 255.0  # Normalize pixel values
    return img

# Preprocess the image
image_path = '/kaggle/input/stressed1/rgb_35image109.jpg'  # Replace with the path to your input image
try:
    input_image = preprocess_image(image_path)
except FileNotFoundError as e:
    print(e)
    # Handle the FileNotFoundError (e.g., provide an alternative image path)

# Create a Gradient*Input analyzer
def gradient_input_analyzer(model, input_image):
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        model_output = model(input_image)
    gradients = tape.gradient(model_output, input_image)
    return gradients

# Convert input_image to TensorFlow tensor
input_image_tensor = tf.convert_to_tensor(input_image[np.newaxis, ...])

# Compute the gradients with respect to the input image
gradients = gradient_input_analyzer(model, input_image_tensor)

# Take the absolute value of the gradients and normalize
heatmap = np.abs(gradients.numpy()).max(axis=-1)

# Reshape the heatmap to 2D array
heatmap_2d = heatmap[0]
# Calculate mean and standard deviation
heatmap_mean = heatmap_2d.mean()
heatmap_std = heatmap_2d.std()

# Check if standard deviation is not zero or NaN
if heatmap_std != 0 and not np.isnan(heatmap_std):
    # Standardize the heatmap
    heatmap = (heatmap_2d - heatmap_mean) / heatmap_std
else:
    # Handle the case where the standard deviation is zero or NaN
    # For example, set the standardized heatmap to zero or another default value
    heatmap = np.zeros_like(heatmap_2d)  # Set to zero
    # Or assign a default value
    # heatmap = np.full_like(heatmap_2d, default_value)

# Plot the input image and heatmap side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot input image
axes[0].imshow(input_image)
axes[0].set_title('Input Image')
axes[0].axis('off')

# Plot heatmap
axes[1].imshow(heatmap, cmap='viridis')
axes[1].set_title('Heatmap')
axes[1].axis('off')
plt.savefig('/kaggle/working/output_image.jpg')

plt.tight_layout()
plt.show() 