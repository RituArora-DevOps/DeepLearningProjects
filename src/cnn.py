import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

# Disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ensure Tensorflow uses CPU only
os.environ['CUDA_VISBLE_DEVICES'] = ''
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

# STEP 1 - Preprocessing and preparing the data
# Load the dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalization
X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255

# Convert class labels to one-hot encoded vectors
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Labels
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Prnt the shapes of the datasets to verify transformations
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape after one-hot encoding: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"X_test shape after one-hot encoding: {y_test.shape}")

# output library
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../output'))

# Define the plot directory within the output directory
plot_path = os.path.join(output_dir, 'plots')

# Create the directory if it doesn't exist
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

# Function to display a sample of images from the dataset and save the plot
def display_images(images, labels, y_data, rows=4, cols=4, save_path=None):
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    axes = axes.ravel()
    for i in np.arange(0, rows*cols):
        index = np.random.randint(0,len(images))
        axes[i].imshow(images[index])
        label_index = np.argmax(y_data[index])
        axes[i].set_title(labels[label_index])
        axes[i].axis('off')
    plt.subplots_adjust(hspace=0.5)
    if save_path:
        plt.savefig(save_path)
        print(f'Plot saved to {save_path}')
    plt.show()
    plt.close()

# Define the file path to save the plot
plot_file = os.path.join(plot_path, 'display_images.png')

# Display a sample of training images with their labels and save the plot
display_images(X_train, labels, y_train, save_path=plot_file)

# STEP 2 - Defining model
# Define a simple CNN
def create_simple_cnn_model():
    model = Sequential([
        Conv2D(32, (3,3), activation = 'relu', input_shape=(32,32,3)),
        MaxPooling2D((2,2)),
        Conv2D(64,(3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    return model

# Ensure the output directory exists
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../output'))

# Create the directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define the model path
model_path = os.path.join(output_dir, 'cifar10_simple_model.h5')

# Check if the model 
if os.path.isfile(model_path):
    # Load the pre-trained model
    model = tf.keras.models.load_model(model_path)
    print(f"Loaded existing model from {model_path}")
else:
    # Create the simple CNN model
    model = create_simple_cnn_model()

    # Compile the model with Adam optimizer and categorical crossentropy loss function
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Print the model summary to understand its architecture
    model.summary()

    # Train the model on the training data
    history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))

    # Save the trained model to the output directory
    model.save(model_path)

    # Plot the training and validation accuracy over epochs
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0,1])
    plt.legend(loc='lower right')

    # Save the plot to a file
    plot_file = os.path.join(plot_path, '02_02_end_simple_model.png')
    plt.savefig(plot_file)
    print(f'Plot saved to {plot_file}')

    plt.show()

plt.show()
plt.close()
# Evaluate the model on the test date to get the loss and accuracy
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_accuracy}')
