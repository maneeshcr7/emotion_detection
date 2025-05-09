import cv2
import os
import numpy as np
from datetime import datetime

def create_directories():
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Create subdirectories for each emotion
    emotions = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    for emotion in emotions:
        emotion_dir = os.path.join('data', emotion)
        if not os.path.exists(emotion_dir):
            os.makedirs(emotion_dir)

def collect_data():
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    # Load face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Create directories
    create_directories()
    
    # Current emotion being collected
    current_emotion = 0
    emotions = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    images_per_emotion = 100
    count = 0
    
    print("\nData Collection Instructions:")
    print("1. Press 'n' to move to the next emotion")
    print("2. Press 'q' to quit")
    print(f"\nCurrent emotion: {emotions[current_emotion]}")
    print("Make the corresponding facial expression and hold it...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]
            
            # Display current emotion and count
            cv2.putText(frame, f"Emotion: {emotions[current_emotion]}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Count: {count}/{images_per_emotion}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Save image if space is pressed
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                if count < images_per_emotion:
                    # Resize face to 48x48
                    face_roi = cv2.resize(face_roi, (48, 48))
                    
                    # Save image
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"data/{emotions[current_emotion]}/{timestamp}.jpg"
                    cv2.imwrite(filename, face_roi)
                    count += 1
                    print(f"Saved image {count}/{images_per_emotion}")
            
            # Move to next emotion if 'n' is pressed
            elif key == ord('n'):
                if count >= images_per_emotion:
                    current_emotion += 1
                    count = 0
                    if current_emotion >= len(emotions):
                        print("\nData collection complete!")
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                    print(f"\nCurrent emotion: {emotions[current_emotion]}")
                    print("Make the corresponding facial expression and hold it...")
            
            # Quit if 'q' is pressed
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return
        
        cv2.imshow('Data Collection', frame)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    collect_data() 