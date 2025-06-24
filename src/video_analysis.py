from .yolo_model import model, labels, box_colors
from .kit_clustering import get_kits_colors, get_kits_classifier, classify_kits, get_left_team_label
from .utils import get_grass_color, get_players_boxes
import cv2
import numpy as np

def process_frame(frame, kits_clf, left_team_label, grass_hsv, width):
    # Run inference
    result = model.track(frame, conf=0.5, persist=True, verbose=False, tracker="src/botsort_custom.yaml")[0]
    annotated_frame = frame.copy()
    players_imgs, players_boxes = get_players_boxes(result)
    kits_colors = get_kits_colors(players_imgs, grass_hsv, annotated_frame)

    # Update classifier and team info if first frame
    if kits_clf is None:
        kits_clf = get_kits_classifier(kits_colors)
        left_team_label = get_left_team_label(players_boxes, kits_colors, kits_clf)
        grass_color = get_grass_color(result.orig_img)
        grass_hsv = cv2.cvtColor(np.uint8([[list(grass_color)]]), cv2.COLOR_BGR2HSV)

    # Annotate each detected object
    for box in result.boxes:
        track_id = int(box.id.item()) if box.id is not None else -1
        label = int(box.cls.cpu().numpy()[0])
        conf = float(box.conf.cpu().numpy()[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        if label == 0:  # Player
            kit_color = get_kits_colors([result.orig_img[y1:y2, x1:x2]], grass_hsv)
            team = classify_kits(kits_clf, kit_color)
            if team == left_team_label:
                label = 0  # Player-L
            else:
                label = 1  # Player-R
        elif label == 1:  # Goalkeeper
            if x1 < 0.5 * width:
                label = 2  # GK-L
            else:
                label = 3  # GK-R
        else:
            label = label + 2  # Adjust other labels (Ball, Refs, Staff)
        
        # Draw bounding box and label
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_colors[str(label)], 2)
        cv2.putText(
            annotated_frame,
            f"{conf:.2f}",
            (x1 - 30, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            box_colors[str(label)],
            2,
        )
    
    return annotated_frame, kits_clf, left_team_label, grass_hsv

def video_generator(video_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    kits_clf = None
    left_team_label = 0
    grass_hsv = None

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        # Process and annotate the frame
        annotated_frame, kits_clf, left_team_label, grass_hsv = process_frame(
            frame, kits_clf, left_team_label, grass_hsv, width
        )
        # Convert BGR to RGB for Gradio display
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        yield annotated_frame_rgb

    cap.release() 