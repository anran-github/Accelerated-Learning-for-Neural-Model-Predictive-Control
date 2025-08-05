import cv2
import numpy as np

# Load the image
image = cv2.imread('/home/anranli/Downloads/1000049699.jpeg')

# Resize for easier processing (optional)
# image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)

# Convert to HSV for color segmentation
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define red color range (adjust as needed)
# lower_red1 = np.array([0, 50, 50])
# upper_red1 = np.array([10, 255, 255])
# lower_red2 = np.array([160, 50, 50])
# upper_red2 = np.array([180, 255, 255])

lower_red1 = np.array([0, 50, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 50, 50])
upper_red2 = np.array([180, 255, 255])

# Mask red color
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask = cv2.bitwise_or(mask1, mask2)

# Clean mask with morphological operations
kernel = np.ones((3, 3), np.uint8)
mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

# Find contours
contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# --- Detect ruler scale ---
# Crop a portion of the ruler and manually measure pixels between mm marks for calibration
# Example: Manually found 50 pixels between two 1-cm marks → scale = 10mm / 50px
scale_mm_per_px = 10 / 50  # ← replace with actual measurement

# Process each object
for i, cnt in enumerate(contours):
    area_px = cv2.contourArea(cnt)
    area_mm2 = area_px * (scale_mm_per_px ** 2)

    # Draw and label
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.putText(image, f"{area_mm2:.2f}", (cx-20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    cv2.drawContours(image, [cnt], -1, (255, 0, 0), 1)

# Show results
cv2.imshow("Detected Organs", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('detected_organs3.png', image)  # Save the result if needed
