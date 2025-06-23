from ultralytics import YOLO

labels = [
    "Player-L", "Player-R", "GK-L", "GK-R", "Ball",
    "Main Ref", "Side Ref", "Staff"
]
box_colors = {
    "0": (150, 50, 50),   # Player-L
    "1": (37, 47, 150),   # Player-R
    "2": (41, 248, 165),  # GK-L
    "3": (166, 196, 10),  # GK-R
    "4": (155, 62, 157),  # Ball
    "5": (123, 174, 213), # Main Ref
    "6": (217, 89, 204),  # Side Ref
    "7": (22, 11, 15),    # Staff
}

model = YOLO("./weights/best.pt").to("cuda") 