import cv2
import numpy as np
from tensorflow.keras.models import load_model


model_path = 'cnn_model1.h5'  
try:
    model = load_model(model_path)
    print(f"Loaded model from {model_path}")
except FileNotFoundError:
    print(f"Error: Model file '{model_path}' not found. Make sure the path is correct.")
    


labels_path = 'labels1.txt'  
try:
    with open(labels_path, 'r') as f:
        class_labels = f.read().splitlines()
    print(f"Loaded class labels from {labels_path}")
except FileNotFoundError:
    print(f"Error: Labels file '{labels_path}' not found. Make sure the path is correct.")


def preprocess_frame(frame):
    
    resized_frame = cv2.resize(frame, (128,128))
    
    normalized_frame = resized_frame / 255.0
   
    input_data = np.expand_dims(normalized_frame, axis=0)
    return input_data


def detect_objects(frame, model, class_labels):
    
    input_data = preprocess_frame(frame)
   
    predictions = model.predict(input_data)

    predicted_class_index = np.argmax(predictions)
    predicted_class_label = class_labels[predicted_class_index]

    return predicted_class_label


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()


while True:
    
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    try:
        predicted_class = detect_objects(frame, model, class_labels)
    except NameError:
        print("Error: Model or labels not loaded correctly. Exiting.")
        break

   
    cv2.putText(frame, f'Predicted: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
