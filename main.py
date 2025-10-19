import os
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
camera_height = 1.65

input_dir = r"C:\Users\ATHARVA\Downloads\OBJECT DETECTION AND DEPTH ESTIMATION\images"
label_dir = r"C:\Users\ATHARVA\Downloads\OBJECT DETECTION AND DEPTH ESTIMATION\labels"
output_dir = r"C:\Users\ATHARVA\Downloads\OBJECT DETECTION AND DEPTH ESTIMATION\output"
calib_dir = r"C:\Users\ATHARVA\Downloads\OBJECT DETECTION AND DEPTH ESTIMATION\calib"
os.makedirs(output_dir, exist_ok=True)

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

def wrap_text(image, text, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.6, thickness=2):
    max_width = image.shape[1] - 20
    words = text.split(" ")
    lines = []
    current_line = words[0]
    for word in words[1:]:
        test_line = f"{current_line} {word}"
        text_size = cv2.getTextSize(test_line, font, font_scale, thickness)[0]
        if text_size[0] <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word
    lines.append(current_line)
    return lines

def annotate_image(image_path, matches, output_path):
    image = cv2.imread(image_path)
    summary = []

    for i, (yolo_box, gt_box, iou, gt_dist, yolo_dist) in enumerate(matches, start=1):
        label = f"Car {i}"
        if yolo_box:
            color = (0, 0, 255) if gt_box else (255, 0, 0)
            cv2.rectangle(image, tuple(map(int, yolo_box[:2])), tuple(map(int, yolo_box[2:])), color, 2)
        if gt_box:
            color = (0, 255, 0) if yolo_box else (255, 0, 255)
            cv2.rectangle(image, tuple(map(int, gt_box[:2])), tuple(map(int, gt_box[2:])), color, 2)

        box = yolo_box if yolo_box else gt_box
        if box:
            x1, y1 = int(box[0]), int(box[1])
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(image, (x1, y1 - 20), (x1 + text_size[0] + 6, y1), (255, 255, 255), -1)
            cv2.putText(image, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        if iou == 0:
            summary.append(f"{label}: IoU={iou:.2f}")
        else:
            summary.append(f"{label}: IoU={iou:.2f}, GT={gt_dist:.2f}m, D={yolo_dist:.2f}m")

    summary_text = " | ".join(summary)
    wrapped_text = wrap_text(image, summary_text)
    line_height = 20
    summary_box_height = len(wrapped_text) * line_height + 20

# Draw white background for summary
    cv2.rectangle(image, (0, 0), (image.shape[1], summary_box_height), (255, 255, 255), -1)

# Draw each line of summary
    for i, line in enumerate(wrapped_text):
        y = 25 + i * line_height
        cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


    cv2.imwrite(output_path, image)
    print(f"Saved: {output_path}")

# Main loop
for image_file in os.listdir(input_dir):
    if image_file.endswith(".png") and image_file.startswith("006"):
        prefix = os.path.splitext(image_file)[0]
        image_path = os.path.join(input_dir, image_file)
        label_path = os.path.join(label_dir, f"{prefix}.txt")
        calib_path = os.path.join(calib_dir, f"{prefix}.txt")
        output_path = os.path.join(output_dir, f"annotated_{image_file}")

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
                        print(f" Could not parse line: {line.strip()}")

        print(f" Parsed {len(gt_boxes)} GT boxes from {label_path}")

        # Parse YOLO detections
        yolo_boxes = []
        for result in results:
            for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
                if int(cls) == 2:
                    xmin, ymin, xmax, ymax = map(float, box[:4])
                    pixel = np.array([(xmin + xmax) / 2, ymax, 1])
                    dist = calculate_world_coordinates(K, pixel, camera_height)
                    yolo_boxes.append(([xmin, ymin, xmax, ymax], dist))

        print(f" Detected {len(yolo_boxes)} YOLO boxes in {image_file}")

        # Match YOLO to GT
        matches = []
        used_gt = set()
        for yolo_box, yolo_dist in yolo_boxes:
            best_iou, best_gt_box, best_gt_dist, best_idx = 0, None, 0, -1
            for idx, (gt_box, gt_dist) in enumerate(gt_boxes):
                if idx in used_gt:
                    continue
                iou = calculate_iou(yolo_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_box = gt_box
                    best_gt_dist = gt_dist
                    best_idx = idx
            if best_gt_box:
                used_gt.add(best_idx)
            matches.append((yolo_box, best_gt_box, best_iou, best_gt_dist, yolo_dist))

        for idx, (gt_box, gt_dist) in enumerate(gt_boxes):
            if idx not in used_gt:
                matches.append((None, gt_box, 0, gt_dist, 0))

        annotate_image(image_path, matches, output_path)
