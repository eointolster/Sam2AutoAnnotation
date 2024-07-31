import torch
from ultralytics import SAM, YOLO
import matplotlib.pyplot as plt
import cv2
import numpy as np
from matplotlib.patches import Polygon

# Check if CUDA is available and set device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the SAM model and move it to the device
sam_model = SAM("sam2_l.pt").to(device)

# Load the YOLO model for object detection
yolo_model = YOLO("yolov8x.pt")

# Load the original image
image_path = "fruitTable.png"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Run YOLO inference
yolo_results = yolo_model(image_path)

# Run SAM inference
sam_results = sam_model(image_path)

# Create a figure and axis
fig, ax = plt.subplots(figsize=(12, 8))
fig.suptitle("")  # Initialize an empty suptitle

# Display the original image
ax.imshow(image_rgb)

# Store polygons and their corresponding class names
polygons = []
class_names = []

# Process YOLO and SAM results
for yolo_result, sam_result in zip(yolo_results, sam_results):
    if hasattr(sam_result, 'masks') and sam_result.masks is not None:
        for mask in sam_result.masks.data:
            # Convert mask to numpy array and find contours
            mask_np = mask.cpu().numpy().astype(np.uint8)
            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Simplify the contour
                epsilon = 0.005 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Create a Polygon patch
                poly = Polygon(approx.reshape(-1, 2), alpha=0.4, fill=True)
                ax.add_patch(poly)
                polygons.append(poly)

                # Get the corresponding class name from YOLO results
                for box in yolo_result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    if cv2.pointPolygonTest(contour, ((x1+x2)/2, (y1+y2)/2), False) >= 0:
                        class_name = yolo_result.names[int(box.cls)]
                        class_names.append(class_name)
                        break
                else:
                    class_names.append("Unknown")

# Function to display annotation on hover
def on_hover(event):
    if event.inaxes is None:
        # Mouse is outside the plot area
        fig.suptitle("")
        fig.canvas.draw_idle()
        return

    for poly, class_name in zip(polygons, class_names):
        if poly.contains(event)[0]:
            fig.suptitle(f"Class: {class_name}")
            fig.canvas.draw_idle()
            return
    fig.suptitle("")
    fig.canvas.draw_idle()

# Connect the hover event
fig.canvas.mpl_connect("motion_notify_event", on_hover)

plt.axis('off')
plt.show()