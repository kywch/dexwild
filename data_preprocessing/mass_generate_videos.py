from data_preprocessing.mass_process_data import main
from data_preprocessing.mass_process_data import generate_hand_videos
import argparse
import os


# Only for video generation, not neeeded for training!
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mass Process Data")
    
    parser.add_argument('--all_folders', '-p',type=str, required=True, help="Path to the Data Directory.")
    parser.add_argument('--left_hand', '-l', action='store_true', help="Process Left Hand")
    parser.add_argument('--right_hand','-r',action='store_true',help="Process Right Hand")
    parser.add_argument("--generate_videos", "-vid", action="store_true", help="Generate Videos")
    parser.add_argument("--robot", "-robot", action="store_true", help="Is this Robot data?")
    
    args = parser.parse_args()
    parent_dir = args.all_folders
    left_hand = args.left_hand
    right_hand = args.right_hand
    process_slam = False
    skip_slam = False
    parallelize = True
    process_hand = False
    process_tracker = False
    clip_actions = False
    intergripper = False
    generate_videos = args.generate_videos
    hand_type = "v2"
    is_robot = False
    v2tov1 = False
    lang = False
    glob_rot = False
    check = False
    
    all_dirs = os.listdir(parent_dir)
    
    rollouts = False
    
    if rollouts:
        videos_dir = os.path.join(parent_dir, "videos")
        os.makedirs(videos_dir, exist_ok=True)
        generate_hand_videos(parent_dir, all_dirs, left_hand, right_hand, True, videos_dir)
    else:
        for folder in all_dirs:
            data_dir = os.path.join(parent_dir, folder)   
            print(data_dir)
            main(data_dir, left_hand, right_hand, process_slam, skip_slam, parallelize, process_hand, process_tracker, clip_actions, intergripper, generate_videos, hand_type, is_robot, v2tov1, lang, glob_rot, check)