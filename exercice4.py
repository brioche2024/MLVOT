import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import os
from KalmanFilter import KalmanFilter


# Load detections from det.txt
def load_detections(detection_file):
    detections = []
    with open(detection_file, 'r') as f:
        for line in f:
            tokens = line.strip().split(',')
            frame_number = int(tokens[0])
            id = int(tokens[1])  # -1 for no ID assigned
            bbox = list(map(float, tokens[2:6]))  # Convert string to float for bbox
            confidence = float(tokens[6])
            # Ignore the world coordinates for 2D challenge
            detections.append({"frame":frame_number, "id":id, "bbox":bbox, "confidence":confidence})
    return detections


# Function to calculate IoU
def bb_intersection_over_union(boxA, boxB):
    x1, y1, w1, h1 = boxA
    x2, y2, w2, h2 = boxB

    # Determine the coordinates of the intersection rectangle
    xA = max(x1, x2)
    yA = max(y1, y2)
    xB = min(x1 + w1, x2 + w2)
    yB = min(y1 + h1, y2 + h2)

    # Compute the area of intersection
    interArea = max(0, xB - xA) * max(0, yB - yA)

    box1_area = w1 * h1
    box2_area = w2 * h2

    # Calculate union area
    union_area = box1_area + box2_area - interArea

    # Compute the intersection over union by taking the intersection area and dividing it by the sum of prediction + ground-truth areas - the interesection area
    iou = interArea / union_area
    # Return the intersection over union value
    return 1 - iou


class Track:
    def __init__(self, detection, track_id, kalman_filter):
        self.id = track_id
        self.kalman_filter = kalman_filter
        self.history = []
        self.bbox = detection['bbox']

    def predict(self):
        # Use the Kalman filter to predict the next state
        return self.kalman_filter.predict()

    def update(self, detection):
        self.bbox = detection['bbox']
        # Centroid of the bounding box
        centroid_x = detection['bbox'][0] + detection['bbox'][2] / 2.0
        centroid_y = detection['bbox'][1] + detection['bbox'][3] / 2.0
        # Update the Kalman filter with the new measurement
        z = np.array([centroid_x, centroid_y])
        self.kalman_filter.update(z)
        # Update the track history
        self.history.append(detection['bbox'])

    def get_predicted_centroid(self):
        # Extract the predicted centroid from the state vector
        predicted_state = self.kalman_filter.xK
        return predicted_state[0, 0], predicted_state[0, 1]


class Tracker:
    def __init__(self, sigma_iou, dt, u_x, u_y, std_acc, x_sdt_meas, y_sdt_meas):
        self.sigma_iou = sigma_iou
        self.tracks = []
        self.next_id = 1
        self.dt = dt
        self.u_x = u_x
        self.u_y = u_y
        self.std_acc = std_acc
        self.x_sdt_meas = x_sdt_meas
        self.y_sdt_meas = y_sdt_meas

    def get_similarity_matrix(self, detections):
        similarity_matrix = np.zeros((len(detections), len(self.tracks)))
        for d, detection in enumerate(detections):
            for t, track in enumerate(self.tracks):
                similarity_matrix[d, t] = bb_intersection_over_union(detection['bbox'], track.bbox)
        return similarity_matrix

    def manage_tracks(self, detections):
        for track in self.tracks:
            track.predict()

        # Compute the cost matrix using 1 - IoU
        cost_matrix = self.get_similarity_matrix(detections)

        # Apply the Hungarian algorithm
        d_idx, t_idx = linear_sum_assignment(cost_matrix)

        # Initialize sets for unmatched detections and tracks
        unmatched_detections = set(range(len(detections)))
        unmatched_tracks = set(range(len(self.tracks)))

        # Process the results from the Hungarian algorithm
        for det_idx, trk_idx  in zip(d_idx, t_idx):
            # Vérifiez si le coût est en dessous du seuil pour déterminer une correspondance
            if cost_matrix[det_idx, trk_idx] < 1 - self.sigma_iou:  # En supposant que sigma_iou est le seuil d'IoU
                # Mettez à jour la piste avec la nouvelle détection
                self.tracks[trk_idx].update(detections[det_idx])
                # Retirez les éléments correspondants des ensembles de non-appariés
                unmatched_detections.remove(det_idx)
                unmatched_tracks.remove(trk_idx)

        # Remove tracks that have no matching detections
        self.tracks = [track for i, track in enumerate(self.tracks) if i not in unmatched_tracks]

        # Create new tracks for detections that were not matched to any existing track
        for det_idx in unmatched_detections:
            kf = KalmanFilter(self.dt, self.u_x, self.u_y, self.std_acc, self.x_sdt_meas, self.y_sdt_meas)
            new_track = Track(detections[det_idx], self.next_id, kf)
            self.tracks.append(new_track)
            self.next_id += 1


def draw_tracks(frame, tracks):
    for track in tracks:
        # Draw bounding box
        bbox = track.bbox
        top_left = (int(bbox[0]), int(bbox[1]))
        bottom_right = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

        # Draw ID
        id_position = (int(bbox[0]), int(bbox[1]) - 10)
        cv2.putText(frame, f'ID: {track.id}', id_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Draw trajectory
        for i in range(1, len(track.history)):
            # Calculate the central points of the current and previous bounding boxes
            prev_bbox = track.history[i - 1]
            curr_bbox = track.history[i]

            prev_center = (int(prev_bbox[0] + prev_bbox[2] / 2), int(prev_bbox[1] + prev_bbox[3] / 2))
            curr_center = (int(curr_bbox[0] + curr_bbox[2] / 2), int(curr_bbox[1] + curr_bbox[3] / 2))

            # Draw a line between the central points
            cv2.line(frame, prev_center, curr_center, (0, 0, 255), 2)

def reset_file(file_path):
    with open(file_path, 'w') as file:
        pass

def save_tracking_results(tracks, sequence_name, frame_id):
    output_filename = f"{sequence_name}.txt"
    output_filepath = os.path.join('.', output_filename)  # Specify your directory

    with open(output_filepath, 'a') as file:
        for track in tracks:
            # Format: [frame_number, id, x, y, width, height, confidence, -1, -1, -1]
            # Note: The '-1' placeholders assume that the world coordinates are not available.
            # You need to replace frame_number and the world coordinates with actual values if available.
            world_x, world_y, world_z = -1, -1, -1  # Replace with actual world coordinates if available
            line = f"{frame_id},{track.id},{track.bbox[0]},{track.bbox[1]},{track.bbox[2]},{track.bbox[3]},1,{world_x},{world_y},{world_z}\n"
            file.write(line)

    return output_filepath

# Initialize the tracker
tracker = Tracker(0.6, 0.1, 1, 1, 1, 0.1, 0.1)

# Sequence name
sequence_name = 'img1'
reset_file(f'{sequence_name}.txt')

# Load detections
detections = load_detections('det/det.txt')

max_frame_number = 525

# Process each frame
for frame_number in range(1, max_frame_number + 1):
    frame_detections = [d for d in detections if d["frame"] == frame_number]
    tracker.manage_tracks(frame_detections)

    output_path = save_tracking_results(tracker.tracks, sequence_name, frame_number)

    # Load the frame image
    frame_path = f'{sequence_name}/{frame_number:06d}.jpg'
    frame = cv2.imread(frame_path)

    # Draw tracks on the frame
    draw_tracks(frame, tracker.tracks)
    resized_image = cv2.resize(frame, (0, 0), fx=0.75, fy=0.75)
    # Display the frame
    cv2.imshow('Frame', resized_image)
    if cv2.waitKey(10) & 0xFF == ord('q'):  # Press 'q' to exit early
        break

cv2.destroyAllWindows()