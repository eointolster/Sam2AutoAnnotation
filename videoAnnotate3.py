import torch
from ultralytics import SAM, YOLO
import cv2
import numpy as np
from collections import Counter

# Check if CUDA is available and set device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the SAM model and move it to the device
sam_model = SAM("sam2_t.pt").to(device)

# Load the YOLO model for object detection
yolo_model = YOLO("yolov8n.pt")  # Using a smaller YOLO model

# Initialize the webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Generate a color palette
np.random.seed(42)
color_palette = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8)

def process_frame(frame):
    # Run YOLO inference
    yolo_results = yolo_model(frame, conf=0.3)  # Adjust confidence threshold as needed

    # Run SAM inference
    sam_results = sam_model(frame)

    # Create an overlay for annotations
    overlay = frame.copy()

    # Counter for detected objects
    object_counter = Counter()

    # Process YOLO and SAM results
    for yolo_result, sam_result in zip(yolo_results, sam_results):
        if hasattr(sam_result, 'masks') and sam_result.masks is not None:
            for i, (mask, box) in enumerate(zip(sam_result.masks.data, yolo_result.boxes)):
                # Convert mask to numpy array
                mask_np = mask.cpu().numpy().astype(np.uint8)

                # Get class name and color
                class_id = int(box.cls)
                class_name = yolo_result.names[class_id]
                color = color_palette[class_id % len(color_palette)]

                # Count objects
                object_counter[class_name] += 1

                # Apply color to the mask
                colored_mask = np.zeros_like(frame)
                colored_mask[mask_np.astype(bool)] = color

                # Blend the colored mask with the overlay
                cv2.addWeighted(overlay, 1, colored_mask, 0.5, 0, overlay)

                # Draw bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color.tolist(), 2)

                # Add label
                label = f"{class_name} {box.conf[0]:.2f}"
                cv2.putText(overlay, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 2)

    # Blend the overlay with the original frame
    result = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

    # Create text overlay
    object_text = ", ".join([f"{count} {name}" for name, count in object_counter.items()])
    cv2.putText(result, object_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return result, object_text

# For measuring FPS
from time import time

prev_time = 0
fps_array = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process the frame
    result, object_text = process_frame(frame)

    # Calculate and display FPS
    current_time = time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    fps_array.append(fps)
    if len(fps_array) > 30:  # Calculate average FPS over last 30 frames
        fps_array.pop(0)
    avg_fps = sum(fps_array) / len(fps_array)
    cv2.putText(result, f"FPS: {avg_fps:.2f}", (10, result.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the result
    cv2.imshow('Webcam Feed with Segmentation', result)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()