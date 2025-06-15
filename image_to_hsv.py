import cv2
import numpy as np

# Open the default camera (0)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open webcam.")
    exit()

print("Press SPACE to capture an image, or 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    cv2.imshow('Camera', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        # Capture frame
        break
    elif key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()

# Convert the captured frame to HSV
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# Show both images
cv2.imshow('Original', frame)
cv2.imshow('HSV', hsv)
print("Press any key to exit.")
cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()
