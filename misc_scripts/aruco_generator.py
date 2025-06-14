import cv2
import cv2.aruco as aruco
import numpy as np
from PIL import Image, ImageDraw

# Parameters
marker_size_cm = 8  # Marker size in cm
paper_size_inches = (8.5, 11)  # US Letter size in inches
markers_per_row = 2  # Number of markers per row

# Conversion constants
dpi = 72  # Dots per inch for printing
cm_to_inch = 2.54
paper_size_pixels = (int(paper_size_inches[0] * dpi), int(paper_size_inches[1] * dpi))
marker_size_pixels = int(marker_size_cm / cm_to_inch * dpi)
x_spacing_pixels = int(1.0 / cm_to_inch * dpi)  # 1 cm spacing between markers
y_spacing_pixels = 0

quiet_zone_ratio = 0.1  # Quiet zone is 20% of the total marker size
quiet_zone_pixels = int(marker_size_pixels * quiet_zone_ratio)
marker_grid_size_pixels = marker_size_pixels - 2 * quiet_zone_pixels

print(f"Generating ArUco markers with size {marker_size_cm} cm")
print(f"Total Pixel Size of the markers is {marker_size_pixels} pixels")
print(f"Actual marker size in cm is {marker_grid_size_pixels / dpi * cm_to_inch:.2f} cm")
print(f"Actual marker size in pixels is {marker_grid_size_pixels} pixels")


# Create a blank canvas for US Letter paper
canvas = Image.new("RGB", paper_size_pixels, "white")
draw = ImageDraw.Draw(canvas)

# Load ArUco dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)

# Generate markers and place them on the canvas
# marker_ids = [42, 43, 44, 45, 46, 47]  # IDs of the markers to generate
marker_ids = [0, 1, 2, 3, 4, 5]  # IDs of the markers to generate

x_offset = x_spacing_pixels
y_offset = x_spacing_pixels

for i, marker_id in enumerate(marker_ids):
    # Generate the marker
    marker_image =cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_grid_size_pixels)
    
    marker_with_border = np.full((marker_size_pixels, marker_size_pixels), 255, dtype=np.uint8)
    marker_with_border[quiet_zone_pixels:-quiet_zone_pixels, quiet_zone_pixels:-quiet_zone_pixels] = marker_image

    # Add a black border edge to the marker
    marker_with_border[:, 0] = 0
    marker_with_border[:, -1] = 0
    marker_with_border[0, :] = 0
    marker_with_border[-1, :] = 0

    
    # Convert to PIL image
    marker_pil = Image.fromarray((marker_with_border).astype(np.uint8))
    
    # save each marker as a PNG file
    marker_pil.save(f"markers/marker_{marker_id}.png")
    
    # Paste the marker onto the canvas
    canvas.paste(marker_pil, (x_offset, y_offset))

    # Update offsets for next marker
    x_offset += marker_size_pixels + x_spacing_pixels
    if (i + 1) % markers_per_row == 0:  # Move to next row after placing 3 markers
        x_offset = x_spacing_pixels
        y_offset += marker_size_pixels + y_spacing_pixels

# Save the canvas as a printable PDF
canvas.save("markers/aruco_markers_letter.pdf", "PDF", resolution=dpi)

print(f"Markers saved to 'aruco_markers_letter.pdf' with marker size {marker_size_cm} cm.")
