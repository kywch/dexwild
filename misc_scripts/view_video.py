"""
Video Playback Utility with Interactive Controls

This script plays a video file in a loop using OpenCV, with support for interactive
keyboard controls to pause, resume, and step through frames. It's useful for inspecting
video files frame-by-frame or reviewing specific segments during analysis.

Usage:
    python play_video.py <path_to_video>

Controls:
    - Spacebar: Pause/Resume playback
    - Right Arrow (→): Step forward by 5 frames (only when paused)
    - Left Arrow (←): Step backward by 5 frames (only when paused)
    - 'q': Quit the video player

Requirements:
    - OpenCV (`cv2`)
    - Python 3.x

Arguments:
    video_path (str): Path to the input video file
"""

import cv2
import argparse
import time

def play_video(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    is_paused = False
    stepping = False  # Track if we're stepping
    print("Playing video on repeat...")
    print(" - Press SPACE to Pause/Resume")
    print(" - Press ← to Step Left (Prev Frame)")
    print(" - Press → to Step Right (Next Frame)")
    print(" - Press 'q' to Quit\n")


    while True:
        if not is_paused or stepping:
            ret, frame = cap.read()

            # If no frame is returned, restart video (looping)
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not reset and read video.")
                    break
                
            stepping = False  # Reset stepping after advancing frames

        # Show the current frame
        cv2.imshow('Video Playback', frame)

        # Get and print the current frame position
        frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total frames for boundaries
        print(f"\rFrame position: {frame_pos}/{total_frames}", end='')

        # Wait for key press
        key = cv2.waitKey(25) & 0xFF

        # Quit if 'q' is pressed
        if key == ord('q'):
            print("\nQuitting...")
            break

        # Toggle pause/resume if 'SPACE' is pressed
        elif key == 32:  # Spacebar
            is_paused = not is_paused
            if is_paused:
                print(f"\nPaused at frame: {frame_pos}")
                print("Press 'k' again to resume.")
            else:
                print("Resumed.")

        # Step Forward (Next Frame)
        elif key == 83: # Right Arrow
            if is_paused:
                new_pos = min(frame_pos + 5, total_frames - 1)  # Avoid exceeding total frames
                cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                ret, frame = cap.read()  # Read new frame after seek
                print(f"\nStepping to right, new frame: {new_pos}")
        
        # Step Backward (Previous Frame)
        elif key == 81: # Left Arrow
            if is_paused:
                new_pos = max(0, frame_pos - 5)  # Prevent negative frames
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read() 
                cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                ret, frame = cap.read()  # Read new frame after seek
                print(f"\nStepping to left, new frame: {new_pos}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", help="Path to the video file")
    args = parser.parse_args()
    play_video(args.video_path)
    