import os
import cv2
import numpy as np
import csv
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")
camera_height = 1.65

# Directories
input_dir = r"C:\Users\ATHARVA\Downloads\OBJECT DETECTION AND DEPTH ESTIMATION\images"
label_dir = r"C:\Users\ATHARVA\Downloads\OBJECT DETECTION AND DEPTH ESTIMATION\labels"
output_dir = r"C:\Users\ATHARVA\Downloads\OBJECT DETECTION AND DEPTH ESTIMATION\output"
calib_dir = r"C:\Users\ATHARVA\Downloads\OBJECT DETECTION AND DEPTH ESTIMATION\calib"
os.makedirs(output_dir, exist_ok=True)

# Output CSV
csv_path = os.path.join(output_dir, "yolo_vs_gt_distances.csv")
csv_file = open(csv_path, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Image", "Car", "YOLO Distance (m)", "GT Distance (m)", "IoU"])

# Global lists for graph
yolo_distances = []
gt_distances = []
yolo_only = []
gt_only = []

def load_intrinsic_matrix(file_path):
    matrix = np.loadtxt(file_path)
    return matrix.reshape(3, 3)

def calculate_world_coordinates(K, pixel_coords, height):
    ray_direction = np.linalg.inv(K) @ pixel_coords
    if ray_direction[1] == 0:
        return 0
    scale = height / ray_direction[1]
    world_coords = np.array([ray_direction[0] * scale, 0, ray_direction[2] * scale])
    return np.linalg.norm(world_coords)

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0

# Main loop
for image_file in os.listdir(input_dir):
    if image_file.endswith(".png") and image_file.startswith("006"):
        prefix = os.path.splitext(image_file)[0]
        image_path = os.path.join(input_dir, image_file)
        label_path = os.path.join(label_dir, f"{prefix}.txt")
        calib_path = os.path.join(calib_dir, f"{prefix}.txt")

        if not os.path.exists(image_path) or not os.path.exists(label_path) or not os.path.exists(calib_path):
            print(f"Missing file for {prefix}: skipping")
            continue

        K = load_intrinsic_matrix(calib_path)
        results = model.predict(source=image_path, save=False)

        # Parse ground truth
        gt_boxes = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6 and parts[0].lower() == "car":
                    try:
                        xmin = float(parts[1])
                        ymin = float(parts[2])
                        xmax = float(parts[3])
                        ymax = float(parts[4])
                        depth = float(parts[5])
                        gt_boxes.append(([xmin, ymin, xmax, ymax], depth))
                    except ValueError:
                        continue

        # Parse YOLO detections
        yolo_boxes = []
        for result in results:
            for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
                if int(cls) == 2:
                    xmin, ymin, xmax, ymax = map(float, box[:4])
                    pixel = np.array([(xmin + xmax) / 2, ymax, 1])
                    dist = calculate_world_coordinates(K, pixel, camera_height)
                    yolo_boxes.append(([xmin, ymin, xmax, ymax], dist))

        # Match YOLO to GT
        car_count = 0
        used_gt = set()
        for yolo_box, yolo_dist in yolo_boxes:
            car_count += 1
            matched = False
            for idx, (gt_box, gt_dist) in enumerate(gt_boxes):
                iou = calculate_iou(yolo_box, gt_box)
                if iou >= 0.75 and idx not in used_gt:
                    used_gt.add(idx)
                    csv_writer.writerow([image_file, car_count, f"{yolo_dist:.2f}", f"{gt_dist:.2f}", f"{iou:.2f}"])
                    yolo_distances.append(yolo_dist)
                    gt_distances.append(gt_dist)
                    matched = True
                    break
            if not matched:
                yolo_only.append(yolo_dist)

        # Add unmatched GT boxes
        for idx, (gt_box, gt_dist) in enumerate(gt_boxes):
            if idx not in used_gt:
                gt_only.append(gt_dist)

csv_file.close()
print(f"Detection complete. CSV saved to: {csv_path}")
