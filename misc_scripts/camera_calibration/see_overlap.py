import cv2

left_image = cv2.imread('left_pinky_1507029504.png')
right_image = cv2.imread('left_thumb_1507029504.png')

# cv2.imshow('Left Image', left_image)
# cv2.imshow('Right Image', right_image)

gray_left = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

# Initialize ORB detector
orb = cv2.ORB_create()

# Detect keypoints and compute descriptors
kp_left, des_left = orb.detectAndCompute(gray_left, None)
kp_right, des_right = orb.detectAndCompute(gray_right, None)

# Use BFMatcher to match keypoints between images
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des_left, des_right)

# Sort matches by distance (optional, for better matching quality)
matches = sorted(matches, key=lambda x: x.distance)

# Get x-coordinates of matched keypoints for each image
left_x_coords = [int(kp_left[m.queryIdx].pt[0]) for m in matches]
right_x_coords = [int(kp_right[m.trainIdx].pt[0]) for m in matches]

# Define the overlapping region for each image
left_overlappingBegin = min(left_x_coords)
left_overlappingEnd = max(left_x_coords)

right_overlappingBegin = min(right_x_coords)
right_overlappingEnd = max(right_x_coords)

print(f"Left Image Overlapping Range: Begin = {left_overlappingBegin}, End = {left_overlappingEnd}")
print(f"Right Image Overlapping Range: Begin = {right_overlappingBegin}, End = {right_overlappingEnd}")
