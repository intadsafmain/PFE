from IPython import get_ipython
from IPython.display import display
# %%
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
import numpy as np
import os
import matplotlib.pyplot as plt
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Update the dataset paths to point to your data in Google Drive
train_dir = '/content/drive/MyDrive/dataset/dataset'
val_dir = '/content/drive/MyDrive/dataset/validation'

# Define the specific path to save the final fine-tuned model in your Google Drive
final_model_save_dir = '/content/drive/MyDrive/my_model_saves/' # Directory to save models
final_model_h5_path = os.path.join(final_model_save_dir, 'my_food_classifier_final.h5')
tflite_model_save_path = os.path.join(final_model_save_dir, 'food_classifier_final.tflite')

num_classes = 21
input_shape = (224, 224, 3)

# Step 2: Create tf.data.Dataset for better performance
def create_dataset(directory, shuffle=True):
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        image_size=input_shape[:2],
        batch_size=32,
        label_mode='categorical',
        shuffle=shuffle
    )
    dataset = dataset.map(lambda x, y: (x / 255.0, y), num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(1000)
    return dataset.prefetch(tf.data.AUTOTUNE)

print("Loading training dataset...")
train_dataset = create_dataset(train_dir, shuffle=True)

print("Loading validation dataset...")
val_dataset = create_dataset(val_dir, shuffle=False)

# Step 3: Compute Class Weights
class_counts = [len(os.listdir(os.path.join(train_dir, cls))) for cls in sorted(os.listdir(train_dir))]
total_samples = sum(class_counts)
class_weights = {i: total_samples / (len(class_counts) * count) for i, count in enumerate(class_counts)}
print("Computed class weights:", class_weights)

# Step 4: Load or Initialize the Model
# Initialize a new model from scratch
print("Initializing new model...")
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Step 5: Callbacks (simplified for saving only at the end)
# EarlyStopping and ReduceLROnPlateau are still useful for training regulation.
# LRTensorBoard is for logging.

# Ensure the directory for saving models in Google Drive exists
os.makedirs(final_model_save_dir, exist_ok=True)

# Custom callback to log learning rate
class LRTensorBoard(tf.keras.callbacks.TensorBoard):
    def __init__(self, log_dir, **kwargs):
        super().__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate.numpy()
        logs.update({'lr': lr})
        super().on_epoch_end(epoch, logs)

# Callbacks list
callbacks = [EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
             ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1),
             LRTensorBoard(log_dir='./logs')]


# Step 6: Train the Model
print("Training the model...")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=20, # Set the desired number of epochs for initial training
    callbacks=callbacks,
    class_weight=class_weights
)

# Step 7: Fine-Tune the Model
print("Fine-tuning the model...")
base_model = model.layers[0]
base_model.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history_fine_tune = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10, # Set the desired number of epochs for fine-tuning
    callbacks=callbacks,
    class_weight=class_weights
)

# Step 8: Save the Final Fine-Tuned Model to Google Drive
print(f"Saving the final fine-tuned model to '{final_model_h5_path}'...")
try:
    model.save(final_model_h5_path)
    print("Final fine-tuned model saved successfully.")
except Exception as e:
    print(f"Error saving the final model: {e}")


# Step 9: Plot Training History
def plot_training_history(history):
    # Combine history from both training phases for plotting
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.show()

# Note: plot_training_history currently only plots the last history object (fine_tune)
# You might want to combine history objects if you want to see the full training history.
plot_training_history(history_fine_tune)

# Step 10: Generate Evaluation Report
y_true = np.concatenate([y for x, y in val_dataset], axis=0)
y_pred = model.predict(val_dataset)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_true, axis=1)

class_labels = sorted(os.listdir(train_dir))

print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=class_labels))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true_classes, y_pred_classes))

# Step 11: Optional - Export to TensorFlow Lite
print(f"Exporting model to TensorFlow Lite format to '{tflite_model_save_path}'...")
try:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(tflite_model_save_path, 'wb') as f:
        f.write(tflite_model)
    print("Model exported to TFLite successfully.")
except Exception as e:
    print(f"Error exporting model to TFLite: {e}")

# %%
# You can use these commands to verify the files in your Drive
# !find /content/drive/MyDrive/my_model_saves/ -name 'my_food_classifier_final.h5'
# !find /content/drive/MyDrive/my_model_saves/ -name 'food_classifier_final.tflite'
