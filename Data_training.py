import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os


dataset_dir = 'captured_images'                          
model_output_path = 'cnn_model1.h5'
label_output_path = 'labels1.txt'


img_width, img_height = 128,128                          
batch_size = 32
epochs = 25

datagen = ImageDataGenerator(rescale=0.2, validation_split=0.2)
Train_data=r"D:\BTECH\Deep learning projects\Object Detection\captured_images\Train_set"
Test_data=r"D:\BTECH\Deep learning projects\Object Detection\captured_images\Test_set"

train_generator = datagen.flow_from_directory(
    Train_data,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    Test_data,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)


num_classes = len(train_generator.class_indices)


with open(label_output_path, 'w') as f:
    for class_label, class_index in train_generator.class_indices.items():
        f.write(f"{class_index}: {class_label}\n")

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_steps=validation_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator
)


model.save(model_output_path)

print(f"Model saved to {model_output_path}")
print(f"Labels saved to {label_output_path}")
