# Loading two filmed videos and comparing their Z position and control inputs over time.
# In two columns for better visualization.
# Transition to GIF for readme file.

import cv2
import numpy as np


# -------------------------------
# User parameters
# -------------------------------
video_left_path = '/home/anranli/Downloads/drone_sin.mp4'
video_right_path = '/home/anranli/Downloads/drone_step.mp4'

output_path = "combined_output.mp4"

# Frame ranges (start and end frames)
left_start = 530
left_end = 1830

right_start = 520
right_end = 1820


# -------------------------------
# Text settings
# -------------------------------
text_speed = "2x speed"
text_left = "Sinusoidal Reference"
text_right = "Step-Wise Reference"

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.8
font_color = (255, 255, 255)
thickness = 2

# Gap size in pixels between videos
gap = 40   # <--- CHANGE THIS to increase/decrease gap


# -------------------------------
# Load videos
# -------------------------------
cap_left = cv2.VideoCapture(video_left_path)
cap_right = cv2.VideoCapture(video_right_path)

if not cap_left.isOpened() or not cap_right.isOpened():
    print("Error: One or both videos cannot be opened.")
    exit()

# FPS and size
fps = cap_left.get(cv2.CAP_PROP_FPS)
width_left = int(cap_left.get(cv2.CAP_PROP_FRAME_WIDTH))
height_left = int(cap_left.get(cv2.CAP_PROP_FRAME_HEIGHT))

width_right = int(cap_right.get(cv2.CAP_PROP_FRAME_WIDTH))
height_right = int(cap_right.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Resize right if needed
if height_left != height_right:
    scaling = height_left / height_right
    width_right = int(width_right * scaling)
    height_right = height_left

# Output size including gap
output_width = width_left + gap + width_right
output_height = height_left

# Output writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))

# -------------------------------
# Jump to starting frames
# -------------------------------
cap_left.set(cv2.CAP_PROP_POS_FRAMES, left_start)
cap_right.set(cv2.CAP_PROP_POS_FRAMES, right_start)


# -------------------------------
# Frame-by-frame loop
# -------------------------------
while True:

    ret_left, frame_left = cap_left.read()
    ret_right, frame_right = cap_right.read()

    current_left = int(cap_left.get(cv2.CAP_PROP_POS_FRAMES))
    current_right = int(cap_right.get(cv2.CAP_PROP_POS_FRAMES))

    if not ret_left or not ret_right:
        break
    if current_left > left_end or current_right > right_end:
        break

    # Resize right frame if needed
    frame_right = cv2.resize(frame_right, (width_right, height_right))

    # Create empty canvas with the gap
    combined = np.zeros((output_height, output_width, 3), dtype=np.uint8)

    # Place left video (x = 0)
    combined[:, :width_left] = frame_left

    # Place right video (x = width_left + gap)
    combined[:, width_left + gap: width_left + gap + width_right] = frame_right

    # -------------------------------
    # Add text overlays
    # -------------------------------
    cv2.putText(combined, text_left, (20, 60), font, font_scale, font_color, thickness)
    cv2.putText(combined, text_right, (width_left + gap + 20, 60), font, font_scale, font_color, thickness)

    # Speed text centered above the gap
    # text_w, text_h = cv2.getTextSize(text_speed, font, font_scale, thickness)[0]
    # speed_x = (output_width - text_w) // 2
    # cv2.putText(combined, text_speed, (speed_x, 35), font, font_scale, font_color, thickness)

    # Save frame
    out.write(combined)


# -------------------------------
# Cleanup
# -------------------------------
cap_left.release()
cap_right.release()
out.release()

print("Done! Output saved to:", output_path)


# To convert to GIF using ffmpeg (run in terminal):
'''
ffmpeg -i combined_output.mp4 -vf "fps=10,scale=960:-1:flags=lanczos,palettegen" palette.png

ffmpeg -i combined_output.mp4 -i palette.png -filter_complex \
"[0:v]fps=10,scale=960:-1:flags=lanczos[v0];[v0][1:v]paletteuse=dither=bayer:bayer_scale=5" \
-loop 0 combined_high_quality.gif
'''
