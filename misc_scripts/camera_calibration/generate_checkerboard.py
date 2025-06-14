import cv2
import numpy as np

# Checkerboard configuration
CHECKERBOARD = (7, 10)
square_size = 25  # Size of a square in mm (adjust as needed)
paper_size = (215.9, 279.4)  # Letter size in mm (8.5 x 11 inches)
dpi = 300  # Print resolution in dots per inch

# Convert paper size to pixels
paper_size_px = tuple(int(dpi * dim / 25.4) for dim in paper_size)

# Calculate the size of the checkerboard pattern
pattern_width = CHECKERBOARD[0] * square_size
pattern_height = CHECKERBOARD[1] * square_size

# Convert pattern size to pixels
square_size_px = int(dpi * square_size / 25.4)
pattern_width_px = CHECKERBOARD[0] * square_size_px
pattern_height_px = CHECKERBOARD[1] * square_size_px

# Create a blank canvas
canvas = np.ones((paper_size_px[1], paper_size_px[0]), dtype=np.uint8) * 255

# Draw the checkerboard pattern
for row in range(CHECKERBOARD[1]):
    for col in range(CHECKERBOARD[0]):
        if (row + col) % 2 == 0:
            top_left = (
                int((paper_size_px[0] - pattern_width_px) / 2 + col * square_size_px),
                int((paper_size_px[1] - pattern_height_px) / 2 + row * square_size_px),
            )
            bottom_right = (
                top_left[0] + square_size_px,
                top_left[1] + square_size_px,
            )
            cv2.rectangle(canvas, top_left, bottom_right, 0, -1)

# Save the checkerboard to a file
output_filename = "checkerboard_letter_size.png"
cv2.imwrite(output_filename, canvas)

print(f"Checkerboard pattern saved to {output_filename}")
