from KalmanFilter import KalmanFilter
from Detector import detect
import numpy as np
import cv2

kalman = KalmanFilter(0.1, 1, 1, 1, 0.1, 0.1)

# Create video capture object
cap = cv2.VideoCapture('randomball.avi')  # Replace with the correct path

# Check if the video was opened correctly
if not cap.isOpened():
    print("Error opening video stream or file")

# Prepare a window for visualization
cv2.namedWindow('Tracking', cv2.WINDOW_NORMAL)

# List to store the points for the trajectory
trajectory_points = []

# Longueur maximale du buffer pour le chemin de tracking
max_buffer_length = 50

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Object detection
    centers = detect(frame)

    if centers:
        # If centroids are detected, use the first one for tracking
        measurement, radius = centers[0]

        # Prediction step of the Kalman Filter
        predicted_state, _ = kalman.predict()

        # Dessinez un cercle vert autour du centre détecté
        cv2.circle(frame, (int(measurement[0]), int(measurement[1])), radius, (0, 255, 0), 2)

        # Draw a blue rectangle as the predicted object position
        cv2.rectangle(frame,
                      (int(predicted_state[0, 0] - 15), int(predicted_state[1, 0] - 15)),
                      (int(predicted_state[0, 0] + 15), int(predicted_state[1, 0] + 15)),
                      (255, 0, 0), 2)

        # Update step of the Kalman Filter
        updated_state, _ = kalman.update(np.reshape(measurement, (2, 1)))

        # Draw a red rectangle as the estimated object position
        cv2.rectangle(frame,
                      (int(updated_state[0, 0] - 15), int(updated_state[1, 0] - 15)),
                      (int(updated_state[0, 0] + 15), int(updated_state[1, 0] + 15)),
                      (0, 0, 255), 2)

        # Stockez les points pour le chemin de tracking
        trajectory_points.append((int(updated_state[0, 0]), int(updated_state[1, 0])))
        # Limitez la longueur du buffer
        if len(trajectory_points) > max_buffer_length:
            trajectory_points.pop(0)

        # Dessinez le chemin de tracking (jaune)
        for i in range(1, len(trajectory_points)):
            cv2.line(frame, trajectory_points[i - 1], trajectory_points[i], (0, 255, 255), 2)

    # Show the frame with the tracking visualization
    cv2.imshow('Tracking', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all frames
cap.release()
cv2.destroyAllWindows()