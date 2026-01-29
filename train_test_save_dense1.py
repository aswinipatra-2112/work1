from pandas import DataFrame
import os
import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.regularizers import l2
from keras.applications.densenet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from keras.models import save_model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

batch_size = 128

def build_mobilenet_model(input_shape):
    weights_path = 'densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5'
    base_model = DenseNet121(weights = weights_path, include_top=False, input_shape=input_shape)
    x = base_model.output
       
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)  # Additional dense layer
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)  # Changed to 1 neuron and sigmoid activation
    
    model = Model(inputs=base_model.input, outputs=predictions)
    initial_learning_rate = 0.001
    steps_per_epoch = 16092 // batch_size  # Assuming train_data is your training dataset
    decay_steps = steps_per_epoch*2  # Adjust this based on your preference
    decay_rate = 0.9
    staircase = True

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=staircase
    )
    optimizer = Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

    

def train_model(model, train_generator, valid_generator, epochs, callbacks=None):
    
    if callbacks is None:
        callbacks = []  # If callbacks are not provided, initialize an empty list

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=valid_generator,
        callbacks=callbacks  # Pass the provided callbacks to model.fit
    )
    return history   

def evaluate_model(model, test_generator, class_mapping=None):
    
      
    # Predict class labels directly from probabilities using a threshold of 0.5
    pred_labels = (model.predict(test_generator) > 0.5).astype(int)
    
    pred_labels = pred_labels.flatten().tolist()
    print(pred_labels)
    
    true_labels = test_generator.classes
    
    print(true_labels)
    
    
    # Create a mapping from numerical indices to class labels
    if class_mapping is not None:
        pred_labels = [class_mapping[label] for label in pred_labels]
        true_labels = [class_mapping[label] for label in true_labels]
    
    print(pred_labels)
    print(true_labels)
    
    # Evaluate the model
    cm = confusion_matrix(true_labels, pred_labels)
    report = classification_report(true_labels, pred_labels)
    acc = accuracy_score(true_labels, pred_labels)
    
      
    # Display confusion matrix and classification report
    print("Confusion Matrix (Xception):")
    print(cm)
    print("\nClassification Acc (Xception):")
    print(acc)
    print("\nClassification Report (Xception):")
    print(report)
 

# Load data
data_dir = "./rgb_train"  # Directory for training Data

# Get the list of sub-folders (classes)
classes = os.listdir(data_dir)

# Dictionary to hold file paths for each class
class_paths = {class_name: [] for class_name in classes}

# Iterate through each class and collect file paths
for class_name in classes:
    class_path = os.path.join(data_dir, class_name)
    class_paths[class_name] = [os.path.join(class_path, file) for file in os.listdir(class_path)]

# Flatten the list of file paths and create labels
all_files = []
labels = []

for class_name, files in class_paths.items():
    all_files.extend(files)
    labels.extend([class_name] * len(files))

# Convert file paths and labels to DataFrames
all_df = DataFrame({'filename': all_files, 'class': labels})

# Convert class labels to numerical labels
class_mapping = {class_name: str(index) for index, class_name in enumerate(classes)}
all_df['class'] = all_df['class'].map(class_mapping)

# Convert 'class' column to string type
all_df['class'] = all_df['class'].astype(str)

# Split data into training (80%) and validation (20%)
train_df, valid_df = train_test_split(all_df, test_size=0.2, random_state=42)

# Directory for Test Data
test_data_dir = "./rgb_test"

# Create data generator for training
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rescale=1./255,
    shear_range=0.2,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col='filename',
    y_col='class',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary'  # binary classification
)

# Create data generator for validation
valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, rescale=1./255)

valid_generator = valid_datagen.flow_from_dataframe(
    valid_df,
    x_col='filename',
    y_col='class',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary'  # binary classification
)

# Create data generator for testing
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary',  # binary classification
    shuffle=False
)


# Build and train the model
input_shape_mobilenet = (224, 224, 3)  # MobileNet input shape
model_mobilenet = build_mobilenet_model(input_shape_mobilenet)
print(model_mobilenet.summary())
checkpoint = ModelCheckpoint(f"dense_checkpoint_best_{{val_loss:.4f}}.h5", 
                             monitor='val_loss', 
                             save_best_only=True, 
                             mode='min', 
                             verbose=1)

# Train the model with ModelCheckpoint callback
history_mobilenet = train_model(model_mobilenet, train_generator, valid_generator, epochs=60, callbacks=[checkpoint])

#Loading the best so far 
min_val_loss_index = np.argmin(history_mobilenet.history['val_loss'])
best_val_loss = history_mobilenet.history['val_loss'][min_val_loss_index]      

result = np.round(best_val_loss, decimals=4)
formatted_result = format(result, '.4f')
print("\The best validation loss so far")
print(formatted_result)
    
best_weights_path = "dense_checkpoint_best_"+str(formatted_result)+".h5"           
model_mobilenet.load_weights(best_weights_path)

# Define the class mapping

class_mapping = {0: 'healthy', 1: 'stressed'}

# Evaluate the model on the test set
evaluate_model(model_mobilenet, test_generator, class_mapping=class_mapping)

image_name="dense_loss_acc"
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history_mobilenet.history['loss'])
plt.plot(history_mobilenet.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train Loss', 'Validation Loss'], loc='upper right')

# Plot training & validation accuracy values
plt.subplot(1, 2, 2)
plt.plot(history_mobilenet.history['accuracy'])
plt.plot(history_mobilenet.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='lower right')

plt.savefig(image_name)
    
print("Val Loss:")
print(history_mobilenet.history['val_loss'])
print("\nTraining Loss:")
print(history_mobilenet.history['loss'])
print("\Val Acc:")
print(history_mobilenet.history['val_accuracy'])
print("\Train Acc:")
print(history_mobilenet.history['accuracy'])
                                    
    
