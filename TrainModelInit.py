import os
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))  # L1
model.add(MaxPooling2D())
model.add(Conv2D(32, (3, 3), activation='relu'))  # L2
model.add(MaxPooling2D())
model.add(Conv2D(32, (3, 3), activation='relu'))  # L3
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define paths for train and test sets
train_data_dir = os.path.join("d:/projects/face mask detection system/train")
test_data_dir = os.path.join("d:/projects/face mask detection system/test")

# Data augmentation and generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Create train and test data generators
training_set = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(150, 150),
    batch_size=16,
    class_mode='binary'
)

test_set = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(150, 150),
    batch_size=16,
    class_mode='binary'
)

# Train the model
history = model.fit(
    training_set,
    epochs=25,
    validation_data=test_set
)

# Save the model
model.save('maskmodel_25.h5')
print("==============================saved the model===============================================")

# Plot the training and validation accuracy
fig = plt.figure(1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot the training and validation loss
fig = plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
