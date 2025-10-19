import csv
import numpy as np
import matplotlib.pyplot as plt

# Paths
csv_path = r"C:\Users\ATHARVA\Downloads\OBJECT DETECTION AND DEPTH ESTIMATION\output\yolo_vs_gt_distances.csv"
graph_path = r"C:\Users\ATHARVA\Downloads\OBJECT DETECTION AND DEPTH ESTIMATION\output\common_points.png"

# Lists to store matched distances
yolo_distances = []
gt_distances = []

# Read CSV and collect only matched points (IoU ≥ 0.75)
with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            iou = float(row["IoU"])
            if iou >= 0.75:
                yolo = float(row["YOLO Distance (m)"])
                gt = float(row["GT Distance (m)"])
                yolo_distances.append(yolo)
                gt_distances.append(gt)
        except:
            continue

# Plot setup
common_points = list(zip(yolo_distances, gt_distances))
plt.figure(figsize=(12, 8))

# Regression line
if common_points:
    x_vals, y_vals = zip(*common_points)
    m, b = np.polyfit(x_vals, y_vals, 1)
    regression_line = np.poly1d((m, b))
    x_max = max(list(x_vals)) + 5  # Add a small buffer
    x_range = np.linspace(0, x_max, 100)
    y_range = regression_line(x_range)
    plt.plot(x_range, y_range, 'k--', label='Regression Line (y = x)')
    plt.scatter(x_vals, y_vals, color='blue', label='Common Points (GT and YOLO)', s=60)

# Labels and layout
plt.xlabel("YOLO Distances (m)")
plt.ylabel("Ground Truth Distances (m)")
plt.title("YOLO vs GT: Matched Detections Only")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.xlim(0, x_max)
plt.ylim(0, max(list(y_vals)) + 20)


# Save and show
plt.savefig(graph_path)
plt.show()
print(f"Graph saved to: {graph_path}")
