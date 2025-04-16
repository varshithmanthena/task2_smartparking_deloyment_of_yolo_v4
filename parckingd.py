import cv2
import numpy as np

# Load class names
with open("coco.names", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Load YOLOv4 model
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

# Setup camera stream
cap = cv2.VideoCapture(0)  # Use 0 or 1 depending on webcam/USB

# Set camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Define parking slots [(x1, y1, x2, y2)]
PARKING_SLOTS = [
    (50, 100, 150, 200),
    (160, 100, 260, 200),
    (270, 100, 370, 200),
]

def is_inside_slot(box, slot):
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    return slot[0] < cx < slot[2] and slot[1] < cy < slot[3]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    layer_names = net.getUnconnectedOutLayersNames()
    outputs = net.forward(layer_names)

    boxes = []
    class_ids = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_names[class_id] in ['car', 'truck', 'bus']:
                cx, cy, w, h = det[0:4] * np.array([width, height, width, height])
                x = int(cx - w / 2)
                y = int(cy - h / 2)
                boxes.append([x, y, int(w), int(h)])
                class_ids.append(class_id)

    # Draw detections
    vehicle_boxes = []
    for box in boxes:
        x, y, w, h = box
        vehicle_boxes.append([x, y, x+w, y+h])
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Check occupied slots
    occupied = 0
    for slot in PARKING_SLOTS:
        status = any(is_inside_slot(box, slot) for box in vehicle_boxes)
        color = (0, 0, 255) if status else (0, 255, 0)
        cv2.rectangle(frame, (slot[0], slot[1]), (slot[2], slot[3]), color, 2)
        if status:
            occupied += 1

    total = len(PARKING_SLOTS)
    empty = total - occupied
    cv2.putText(frame, f"Occupied: {occupied} | Empty: {empty}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("USB CCTV Parking Detection", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
