import cv2
import numpy as np


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
    return iou


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
        if not self.tracks:  # If this is the first frame or no tracks exist
            # Assign new track IDs to all detections
            for detection in detections:
                self.tracks.append({
                    'id': self.next_id,
                    'bbox': detection['bbox'],
                    'history': [],
                })
                self.next_id += 1
            return

        # Assuming detections is a list of dictionaries with 'bbox' and 'id' keys
        # Assuming tracks is a list of dictionaries with 'bbox', 'id', 'unmatched_frames' keys

        iou_matrix = self.get_similarity_matrix(detections)

        # Initialize lists for matches, unmatched detections, and unmatched tracks
        matches = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(self.tracks)))

        # Find matches and unmatched detections
        if len(self.tracks) > 0:
            for d_idx, detection in enumerate(detections):
                best_iou = self.sigma_iou
                best_track_idx = None
                for t_idx, track in enumerate(self.tracks):
                    iou = iou_matrix[d_idx, t_idx]
                    if iou >= best_iou:
                        best_iou = iou
                        best_track_idx = t_idx

                if best_track_idx is not None:
                    matches.append((d_idx, best_track_idx))
                    unmatched_detections.remove(d_idx)
                    if best_track_idx in unmatched_tracks:
                        unmatched_tracks.remove(best_track_idx)

        # Update matched tracks
        for d_idx, t_idx in matches:
            self.tracks[t_idx]['bbox'] = detections[d_idx]['bbox']
            self.tracks[t_idx]['history'].append(detections[d_idx]['bbox'])

        # Delete unmatched tracks
        for t_idx in unmatched_tracks:
            self.tracks[t_idx] = None

        self.tracks = [track for track in self.tracks if track]

        # Create new tracks for unmatched detections
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


# Initialize the tracker
tracker = Tracker(sigma_iou=0.4)


# Load detections
detections = load_detections('det/det.txt')

max_frame_number = 525

# Process each frame
for frame_number in range(1, max_frame_number + 1):
    frame_detections = [d for d in detections if d["frame"] == frame_number]
    tracker.manage_tracks(frame_detections)

    # Load the frame image
    frame_path = f'img1/{frame_number:06d}.jpg'
    frame = cv2.imread(frame_path)

    # Draw tracks on the frame
    draw_tracks(frame, tracker.tracks)
    resized_image = cv2.resize(frame, (0, 0), fx=0.80, fy=0.80)
    # Display the frame
    cv2.imshow('Frame', resized_image)
    if cv2.waitKey(50) & 0xFF == ord('q'):  # Press 'q' to exit early
        break

cv2.destroyAllWindows()