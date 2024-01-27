import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import os


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


class Tracker:
    def __init__(self, sigma_iou):
        self.sigma_iou = sigma_iou
        self.tracks = []
        self.next_id = 1

    def get_similarity_matrix(self, detections):
        similarity_matrix = np.zeros((len(detections), len(self.tracks)))
        for d, detection in enumerate(detections):
            for t, track in enumerate(self.tracks):
                similarity_matrix[d, t] = bb_intersection_over_union(detection['bbox'], track['bbox'])
        return similarity_matrix

    def manage_tracks(self, detections):
        if not self.tracks:  # If there are no existing tracks
            # Create new tracks for all detections
            for detection in detections:
                self.tracks.append({
                    'id': self.next_id,
                    'bbox': detection['bbox'],
                    'history': [],
                })
                self.next_id += 1
            return

        # Compute the cost matrix using 1 - IoU
        cost_matrix = self.get_similarity_matrix(detections)

        # Apply the Hungarian algorithm
        d_idx, t_idx = linear_sum_assignment(cost_matrix)

        # Initialize sets for unmatched detections and tracks
        unmatched_detections = set(range(len(detections)))
        unmatched_tracks = set(range(len(self.tracks)))

        # Create a set for the indices of the tracks to avoid modifying the list during iteration
        all_track_indices = set(range(len(self.tracks)))

        # Process the results from the Hungarian algorithm
        for det_idx, trk_idx  in zip(d_idx, t_idx):
            # Vérifiez si le coût est en dessous du seuil pour déterminer une correspondance
            if cost_matrix[det_idx, trk_idx] < 1 - self.sigma_iou:  # En supposant que sigma_iou est le seuil d'IoU
                # Mettez à jour la piste avec la nouvelle détection
                self.tracks[trk_idx]['bbox'] = detections[det_idx]['bbox']
                self.tracks[trk_idx]['history'].append(detections[det_idx]['bbox'])
                # Retirez les éléments correspondants des ensembles de non-appariés
                unmatched_detections.remove(det_idx)
                unmatched_tracks.remove(trk_idx)

        # Remove tracks that have no matching detections
        self.tracks = [track for i, track in enumerate(self.tracks) if i not in unmatched_tracks]

        # Create new tracks for detections that were not matched to any existing track
        for d_idx in unmatched_detections:
            self.tracks.append({
                'id': self.next_id,
                'bbox': detections[d_idx]['bbox'],
                'history': [],
            })
            self.next_id += 1


def draw_tracks(frame, tracks):
    for track in tracks:
        # Draw bounding box
        bbox = track['bbox']
        top_left = (int(bbox[0]), int(bbox[1]))
        bottom_right = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

        # Draw ID
        id_position = (int(bbox[0]), int(bbox[1]) - 10)
        cv2.putText(frame, f'ID: {track["id"]}', id_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Draw trajectory
        if 'history' in track:  # Check if history is being recorded in the track
            for i in range(1, len(track['history'])):
                # Calculate the central points of the current and previous bounding boxes
                prev_bbox = track['history'][i - 1]
                curr_bbox = track['history'][i]

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
            line = f"{frame_id},{track['id']},{track['bbox'][0]},{track['bbox'][1]},{track['bbox'][2]},{track['bbox'][3]},1,{world_x},{world_y},{world_z}\n"
            file.write(line)

    return output_filepath


# Initialize the tracker
tracker = Tracker(sigma_iou=0.6)

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

    # Write the tracks on the file
    output_path = save_tracking_results(tracker.tracks, sequence_name, frame_number)

    # Load the frame image
    frame_path = f'img1/{frame_number:06d}.jpg'
    frame = cv2.imread(frame_path)

    # Draw tracks on the frame
    draw_tracks(frame, tracker.tracks)
    resized_image = cv2.resize(frame, (0, 0), fx=0.80, fy=0.80)
    # Display the frame
    cv2.imshow('Frame', resized_image)
    if cv2.waitKey(10) & 0xFF == ord('q'):  # Press 'q' to exit early
        break

cv2.destroyAllWindows()