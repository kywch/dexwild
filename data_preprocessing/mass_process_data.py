# Copyright (c) 2025 Tony Tao
# Licensed under the MIT License. See LICENSE file in project root for details.
"""
Mass Data Processing Pipeline for Robot and Human Demonstrations
=================================================================

This script performs modular batch processing of demonstration data.

Features:
- Supports both robot and human data modalities
- Processes Leap v1/v2 hand retargeting, 3D tracker poses, and intergripper poses
- Performs SLAM using SVO recordings and generates 3D reconstructions
- Validates data integrity, clips anomalous actions, and indexes episodes
- Optionally generates diagnostic summaries and video previews
- Organizes processed metadata into a stats directory for downstream use

Typical usage:
    python process_all.py --data_dir /path/to/dataset \
                          --left_hand --right_hand \
                          --process_svo --process_hand --process_tracker \
                          --generate_videos --lang --intergripper

Configuration:
- Reads hand marker parameters from `processing_config.yaml`
- Requires structured per-episode data directory layout

Outputs:
- Saves trajectory statistics, indexed episode mappings, and optional video/language output
- Logs and colored prints give live progress and error tracking
"""
import os
import yaml
import argparse
import time
from termcolor import colored

from dexwild_utils.data_processing import (
    index_episodes,
    left_face_corners_3d,
    right_face_corners_3d
)

from processing_helpers.debugging import (
    check_slam_failed,
    check_tracker_failed,
    check_retargeting_failed,
    generate_diagnostics,
    check_episode_lengths
)
from processing_helpers.process_svo import process_svo_wrapper
from processing_helpers.process_hands import process_hand_wrapper
from processing_helpers.process_tracker import process_tracker_wrapper
from processing_helpers.process_intergripper import process_intergripper_wrapper
from processing_helpers.video_generation import generate_hand_videos
from processing_helpers.text_generation import generate_text_labels
from processing_helpers.clip_actions import clip_eef_actions


with open("processing_config.yaml", "r") as f:
    config = yaml.safe_load(f)
                    
def main(data_dir=None, left_hand=False, right_hand=False, process_svo=False, skip_slam = False, parallelize=False, process_hand=False, process_tracker=False, clip_actions=False, intergripper=False, generate_videos=False, hand_type = "v2", is_robot = False, lang = False):
    process_start_time = time.time()
    
    ROBOT = is_robot
    
    assert os.path.exists(data_dir), "Data Directory does not exist"
    
    def print_colored(label, value):
        color = "green" if value else "red"
        print(colored(f"{label}: {value}", color))
        
    print_colored("Robot Data", ROBOT)
    print_colored("Data Directory", data_dir)
    print_colored("Parallelize", parallelize)
    print_colored("Left Hand", left_hand)
    print_colored("Right Hand", right_hand)
    print_colored("Process SVO", process_svo)
    print_colored("Skip SLAM", skip_slam)
    print_colored("Process Hand", process_hand)
    if process_hand:
        if hand_type == "v2":
            print(colored("Hand Type: Leap V2", "blue"))
        elif hand_type == "v1":
            print(colored("Hand Type: Leap V1", "yellow"))
        else:
            raise ValueError("Hand Type must be either 'v1' or 'v2'")
    print_colored("Process Tracker", process_tracker)
    print_colored("Clip Actions", clip_actions)
    print_colored("Intergripper", intergripper)
    print_colored("Generate Videos", generate_videos)
    print_colored("Generate Language", lang)

    if not left_hand and not right_hand:
        print(colored("[WARNING] You have not selected any hands to process. Press Enter to continue...", "red"))
        input("")
        
    if ROBOT:
        print(colored("Processing Robot data: Press Enter to confirm...", "green"))
        input()

    marker_ids = {}
    corner_faces = {}
    
    if left_hand:
        params = config["hands"]["left_hand"]
        cube_size = params["cube_size"]
        marker_size = params["marker_size"]
        transformation = params["transformation"]
        marker_ids["left_hand"] = params["marker_ids"]
        corner_faces ["left_hand"] = left_face_corners_3d
        
    if right_hand:   
        params = config["hands"]["right_hand"]
        cube_size = params["cube_size"]
        marker_size = params["marker_size"]
        transformation = params["transformation"]
        marker_ids["right_hand"] = params["marker_ids"]
        corner_faces ["right_hand"] = right_face_corners_3d
    
    all_episodes = index_episodes(data_dir)
    
    # [RESET all the processing done so far]
    # reset_dataset(data_dir, all_episodes)
    # quit()
    
    print(f"{len(all_episodes)} / {len(os.listdir(data_dir))} of the folder contains episodes ")
    
    print(colored("Please confirm these settings...", "blue"))
    input()
    
    if os.path.exists(os.path.join(data_dir, "stats")):
        print(colored("Folder has already been processed. Are you sure you want to continue?", "red"))
        input()
    
    os.makedirs(os.path.join(data_dir, "stats"), exist_ok=True)
    
    check_episode_lengths(data_dir, all_episodes)
    
    if lang:
        text_label = input("Enter the text label for the episodes: ")
        generate_text_labels(data_dir, all_episodes, dummy_label=text_label)
    
    if generate_videos:
        if is_robot:
            generate_hand_videos(data_dir, all_episodes, left_hand, right_hand, is_robot = True)
        else:
            generate_hand_videos(data_dir, all_episodes, left_hand, right_hand, is_robot = False)
            
            # [optionally] generate all videos for viewing in "videos" directory
            
            # slam_dir = os.path.join(data_dir, 'zed_recordings')  
            # all_svos = os.listdir(slam_dir)
            # all_svos = sorted([svo for svo in all_svos if "output" in svo])
            # generate_all_videos(data_dir, all_svos, left_hand, right_hand, parallelize)
            
    if process_svo:
        split_time = time.time()
        slam_dir = os.path.join(data_dir, 'zed_recordings')  
        all_svos = os.listdir(slam_dir)
        all_svos = sorted([svo for svo in all_svos if "output" in svo])
        
        process_svo_wrapper(all_svos, slam_dir, corner_faces, cube_size, marker_size, transformation, marker_ids, skip_slam, parallelize)
        print(f"SLAM Processed in {time.time() - split_time} seconds")
        all_episodes = check_slam_failed(data_dir, all_episodes)
        
    if process_hand:
        split_time = time.time()
        process_hand_wrapper(all_episodes, data_dir, left_hand, right_hand, parallelize, hand_type)
        print(f"Hand Processed in {time.time() - split_time} seconds")
        all_episodes = check_retargeting_failed(data_dir, all_episodes, hand_type)
    
    if process_tracker:
        split_time = time.time()
        process_tracker_wrapper(all_episodes, data_dir, left_hand, right_hand, parallelize)
        print(f"Tracker Processed in {time.time() - split_time} seconds")
        all_episodes = check_tracker_failed(data_dir, all_episodes) 
    
    if clip_actions:
        split_time = time.time()
        print(f"Clipping Actions for {len(all_episodes)} episodes")
        clip_eef_actions(data_dir, all_episodes, left_hand, right_hand)
        print(f"Actions Clipped in {time.time() - split_time} seconds")
    
    if intergripper:
        split_time = time.time()
        process_intergripper_wrapper(data_dir, all_episodes, ROBOT)
        print(f"Intergripper Processed in {time.time() - split_time} seconds")

    episode_ids = {}
    for i, episode in enumerate(all_episodes):
        episode_ids[episode] = i
    
    with open(os.path.join(data_dir, "stats", "episode_ids.txt"), 'w') as f:
        for key in episode_ids.keys():
            f.write(f"{key}: {episode_ids[key]}\n")
    
    process_end_time = time.time()  
    
    if not ROBOT:
        generate_diagnostics(data_dir)
        
    print(f"Total Processing Time: {process_end_time - process_start_time} seconds")
    print(f"Average processing time per episode: {(process_end_time - process_start_time) / len(all_episodes)} seconds")

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mass Process Data")
    
    parser.add_argument('--data_dir', '-p',type=str, required=True, help="Path to the Data Directory.")
    parser.add_argument('--left_hand', '-l', action='store_true', help="Process Left Hand")
    parser.add_argument('--right_hand','-r',action='store_true',help="Process Right Hand")
    parser.add_argument('--process_svo', '-slam', action='store_true', help="Process SLAM")
    parser.add_argument('--skip_slam', '-skip', action='store_true', help="Skip SLAM")
    parser.add_argument('--parallelize', '-para', action='store_true', help="Parallelize Processes")
    parser.add_argument('--process_hand', '-hand', action='store_true', help="Process Hand")
    parser.add_argument('--process_tracker', '-tracker', action='store_true', help="Process Tracker")
    parser.add_argument("--clip_actions", "-clip", action="store_true", help="Clip Actions")
    parser.add_argument("--intergripper", "-inter", action="store_true", help="Intergripper")
    parser.add_argument("--generate_videos", "-vid", action="store_true", help="Generate Videos")
    parser.add_argument("--hand_type", "-leap", type=str, default="v2", help="Type of hand data to process (default: leapv2)")
    parser.add_argument("--robot", "-robot", action="store_true", help="Is this Robot data?")
    parser.add_argument("--lang" , "-lang", action="store_true", help="Generate language data (default: False)")
    
    args = parser.parse_args()
    data_dir = args.data_dir
    left_hand = args.left_hand
    right_hand = args.right_hand
    process_svo = args.process_svo
    skip_slam = args.skip_slam
    parallelize = args.parallelize
    process_hand = args.process_hand
    process_tracker = args.process_tracker
    clip_actions = args.clip_actions
    intergripper = args.intergripper
    generate_videos = args.generate_videos
    hand_type = args.hand_type
    is_robot = args.robot
    lang = args.lang

    main(data_dir, left_hand, right_hand, process_svo, skip_slam, parallelize, process_hand, process_tracker, clip_actions, intergripper, generate_videos, hand_type, is_robot, lang)