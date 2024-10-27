from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import cvlib as cv

# Load model
model = load_model('gender_detection.h5')

# Open webcam
webcam = cv2.VideoCapture(0)
    
classes = ['man', 'woman']

# Function to check if the current frame is in low light condition
def is_low_light(frame, threshold=50):
    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate mean brightness
    mean_brightness = np.mean(gray_frame)
    
    # Check against the threshold
    return mean_brightness < threshold

# Initialize counters for accuracy calculation
total_faces = 0
correct_predictions = 0

# Loop through frames
while webcam.isOpened():

    # Read frame from webcam 
    status, frame = webcam.read()

    if not status:
        print("Error: Could not read frame.")
        break

    # Check for low light condition
    if is_low_light(frame):
        cv2.putText(frame, "Low Light Condition", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Adequate Light", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Apply face detection
    faces, confidence = cv.detect_face(frame)

    # Initialize counters for this frame
    man_count = 0
    woman_count = 0

    # Loop through detected faces
    for idx, f in enumerate(faces):

        # Get corner points of face rectangle        
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # Draw rectangle over face
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # Crop the detected face region
        face_crop = np.copy(frame[startY:endY, startX:endX])

        if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
            continue

        # Preprocessing for gender detection model
        face_crop = cv2.resize(face_crop, (96, 96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # Apply gender detection on face
        conf = model.predict(face_crop)[0]  # model.predict returns a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]

        # Get label with max accuracy
        idx = np.argmax(conf)
        label = classes[idx]

        # Increment the appropriate counter
        if label == 'man':
            man_count += 1
        else:
            woman_count += 1

        label = "{}: {:.2f}%".format(label, conf[idx] * 100)

        Y = startY - 10 if startY - 10 > 10 else startY + 10

        # Write label and confidence above face rectangle
        cv2.putText(frame, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

        # Update accuracy calculation
        total_faces += 1
        correct_predictions += conf[idx] > 0.5  # Assuming 0.5 as the threshold for correctness

    # Calculate accuracy
    accuracy = (correct_predictions / total_faces) * 100 if total_faces > 0 else 0

    # Display the number of men and women detected
    cv2.putText(frame, f"Men: {man_count}, Women: {woman_count}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the accuracy
    cv2.putText(frame, f"Accuracy: {accuracy:.2f}%", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Check for alert condition (only one woman detected, regardless of men count)
    if woman_count == 1:
        cv2.putText(frame, "ALERT: Woman Alone", (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Display output
    cv2.imshow("Gender Detection", frame)

    # Press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
webcam.release()
cv2.destroyAllWindows()
