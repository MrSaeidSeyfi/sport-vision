import cv2
import numpy as np

def get_grass_color(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    grass_color = cv2.mean(img, mask=mask)
    return grass_color[:3]

def get_players_boxes(result):
    players_imgs = []
    players_boxes = []
    for box in result.boxes:
        label = int(box.cls.cpu().numpy()[0])
        if label == 0:  # Assuming 0 is for players
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            player_img = result.orig_img[y1:y2, x1:x2]
            players_imgs.append(player_img)
            players_boxes.append(box)
    return players_imgs, players_boxes 