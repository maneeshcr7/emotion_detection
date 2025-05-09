import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os
from PIL import Image

class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, 7)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 6 * 6)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def load_model(model_path):
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        model = EmotionCNN()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

# Emotion labels
emotion_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# Initialize camera
try:
    cap = cv2.VideoCapture(0)  # 0 for default camera
    if not cap.isOpened():
        raise Exception("Failed to open camera")
except Exception as e:
    print(f"Error initializing camera: {str(e)}")
    exit(1)

# Load face detector
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        raise Exception("Failed to load face cascade classifier")
except Exception as e:
    print(f"Error loading face detector: {str(e)}")
    cap.release()
    exit(1)

# Load model
model_path = "models/emotion_model.pth"
model = load_model(model_path)
if model is None:
    print("Failed to load model. Please ensure the model file exists and is valid.")
    print("You need to train the model first using train_model.py")
    exit(1)

# Define image transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            try:
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Extract face region
                face_roi = gray[y:y + h, x:x + w]
                
                # Transform image for model input
                face_tensor = transform(face_roi).unsqueeze(0)

                # Predict emotion
                with torch.no_grad():
                    outputs = model(face_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    emotion_index = torch.argmax(probabilities).item()
                    confidence = probabilities[0][emotion_index].item()
                
                emotion = emotion_labels[emotion_index]

                # Display emotion and confidence
                text = f"{emotion}: {confidence:.2f}"
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            except Exception as e:
                print(f"Error processing face: {str(e)}")
                continue

        cv2.imshow("Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"An error occurred: {str(e)}")
finally:
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
