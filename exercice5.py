import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import os
from KalmanFilter import KalmanFilter
from torchvision.models import resnet50
from torchvision.transforms import transforms
from yolov5 import YOLOv5

import torch
import torch.nn as nn


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


def detect_with_yolo(model, frame):
    results = model.predict(frame)
    detections = []
    for det in results.xyxy[0]:  # detections per image
        if det[-1] == 0:  # Assuming '0' is the class ID for pedestrians
            print(len(det))
            print(det)
            bbox = [det[0].item(), det[1].item(), det[2].item() - det[0].item(), det[3].item() - det[1].item()]  # x, y, x, h
            confidence = det[4].item()
            detections.append({"id": -1, "bbox": bbox, "confidence": confidence})
    return detections


def extract_features(image, bbox):
    """
    Extract CNN features for the image patch defined by bbox.
    """
    img_height, img_width, _ = frame.shape
    x, y, w, h = bbox
    x, y, w, h = map(int, [x, y, w, h])  # Convert to integers
    x = max(0, x)
    y = max(0, y)

    patch = image[y:y+h, x:x+w]
    patch = preprocess(patch).unsqueeze(0)  # Ajoute une dimension de batch

    patch = patch.to(device)
    with torch.no_grad():
        features = model(patch)  # Add batch dimension and get features

    features = features.cpu().flatten().numpy()
    return features  # Convert to numpy array and flatten


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
    return iou


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

    def get_similarity_matrix(self, detections, frame):
        similarity_matrix = np.zeros((len(detections), len(self.tracks)))

        # Extract features for the detections
        detection_features = [extract_features(frame, det['bbox']) for det in detections]

        for d_idx, det_features in enumerate(detection_features):
            for t_idx, track in enumerate(self.tracks):
                # Extract features for the track using the current bbox
                track_features = extract_features(frame, track.bbox)

                # Calculate the visual similarity
                # For example, using the cosine similarity:
                visual_similarity = np.dot(det_features, track_features) / (
                            np.linalg.norm(det_features) * np.linalg.norm(track_features))

                # Calculate the IoU
                iou_score = bb_intersection_over_union(detections[d_idx]['bbox'], track.bbox)

                # Combine IoU and visual similarity into a single score
                # For example, here we're simply averaging them
                combined_score = (iou_score + visual_similarity) / 2

                # Update the similarity matrix with the combined score
                similarity_matrix[d_idx, t_idx] = combined_score

        # Convert the combined similarity score to a cost for the Hungarian algorithm
        cost_matrix = 1 - similarity_matrix
        return cost_matrix

    def manage_tracks(self, detections, frame):
        for track in self.tracks:
            track.predict()

        # Compute the cost matrix using 1 - IoU
        cost_matrix = self.get_similarity_matrix(detections, frame)

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
            # Extract bbox coordinates and convert them to integers
            x, y, w, h = map(int, track.bbox)  # Convert bbox coordinates to integers

            world_x, world_y, world_z = -1, -1, -1

            # Construct the output line with integer bbox coordinates
            line = f"{frame_id},{track.id},{x},{y},{w},{h},1,{world_x},{world_y},{world_z}\n"
            file.write(line)

    return output_filepath


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load YOLO model
yolo_model = YOLOv5("yolov5s.pt")

# Initialize a pre-trained ResNet-50
model = resnet50(pretrained=True)
model = nn.Sequential(*(list(model.children())[:-1]))  # Remove the last classification layer
model.eval()  # Set to evaluation mode
model = model.to(device)

# Define a transform to preprocess the image patches
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

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
    # Load the frame image
    frame_path = f'{sequence_name}/{frame_number:06d}.jpg'
    frame = cv2.imread(frame_path)

    frame_detections = detect_with_yolo(yolo_model, frame)
    tracker.manage_tracks(frame_detections, frame)

    output_path = save_tracking_results(tracker.tracks, sequence_name, frame_number)


    # Draw tracks on the frame
    draw_tracks(frame, tracker.tracks)
    resized_image = cv2.resize(frame, (0, 0), fx=0.75, fy=0.75)
    # Display the frame
    cv2.imshow('Frame', resized_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit early
        break

cv2.destroyAllWindows()