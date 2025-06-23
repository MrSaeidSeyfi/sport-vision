import numpy as np
import cv2
from sklearn.cluster import KMeans
from .utils import get_grass_color

def get_kits_colors(players, grass_hsv=None, frame=None):
    kits_colors = []
    if grass_hsv is None:
        grass_color = get_grass_color(frame)
        grass_hsv = cv2.cvtColor(np.uint8([[list(grass_color)]]), cv2.COLOR_BGR2HSV)
    for player_img in players:
        hsv = cv2.cvtColor(player_img, cv2.COLOR_BGR2HSV)
        lower_green = np.array([grass_hsv[0, 0, 0] - 10, 40, 40])
        upper_green = np.array([grass_hsv[0, 0, 0] + 10, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        mask = cv2.bitwise_not(mask)
        upper_mask = np.zeros(player_img.shape[:2], np.uint8)
        upper_mask[0:player_img.shape[0] // 2, player_img.shape[1] // 3: 2 * player_img.shape[1] // 3] = 255
        mask = cv2.bitwise_and(mask, upper_mask)
        kit_color = np.array(cv2.mean(player_img, mask=mask)[:3])
        kits_colors.append(kit_color)
    return kits_colors

def get_kits_classifier(kits_colors):
    kits_kmeans = KMeans(n_clusters=2)
    kits_kmeans.fit(kits_colors)
    return kits_kmeans

def classify_kits(kits_classifier, kits_colors):
    team = kits_classifier.predict(kits_colors)
    return team

def get_left_team_label(players_boxes, kits_colors, kits_clf):
    left_team_label = 0
    team_0 = []
    team_1 = []
    for i in range(len(players_boxes)):
        x1, y1, x2, y2 = map(int, players_boxes[i].xyxy[0].cpu().numpy())
        team = classify_kits(kits_clf, [kits_colors[i]]).item()
        if team == 0:
            team_0.append(np.array([x1]))
        else:
            team_1.append(np.array([x1]))
    team_0 = np.array(team_0)
    team_1 = np.array(team_1)
    if len(team_0) > 0 and len(team_1) > 0:
        if np.average(team_0) > np.average(team_1):
            left_team_label = 1
    return left_team_label 