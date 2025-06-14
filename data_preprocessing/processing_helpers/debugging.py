import os 
from tqdm import tqdm
from dexwild_utils.data_processing import index_episodes
import matplotlib.pyplot as plt
import numpy as np
from termcolor import colored

def check_slam_failed(data_dir, all_episodes):
    successful_episodes = []
    failed_episodes = []
    for episode in all_episodes:
        episode_path = os.path.join(data_dir, episode)
        slam_path = os.path.join(episode_path, "zed", "zed_pose.pkl")
        if not os.path.exists(slam_path):
            os.rename(episode_path, os.path.join(data_dir, "failed_" + episode))
            failed_episodes.append(episode)
            continue
        successful_episodes.append(episode)
    print(f"SLAM Failed Episodes: {failed_episodes}")
    # input("Press Enter to continue...")
    return successful_episodes

def check_retargeting_failed(data_dir, all_episodes, hand_type):
    successful_episodes = []
    failed_episodes = []
    for episode in all_episodes:
        episode_path = os.path.join(data_dir, episode)
        
        if hand_type == "v2":
            left_retargeted_path = os.path.join(episode_path, "left_leapv2", "left_leapv2.pkl")
            right_retargeted_path = os.path.join(episode_path, "right_leapv2", "right_leapv2.pkl")
        elif hand_type == "v1":
            left_retargeted_path = os.path.join(episode_path, "left_leapv1", "left_leapv1.pkl")
            right_retargeted_path = os.path.join(episode_path, "right_leapv1", "right_leapv1.pkl")
            
        if not os.path.exists(left_retargeted_path) and not os.path.exists(right_retargeted_path):
            os.rename(episode_path, os.path.join(data_dir, "failed_" + episode))
            print("glove failed for episode", episode)
            failed_episodes.append(episode)
            continue
        successful_episodes.append(episode)
        
    return successful_episodes

def check_tracker_failed(data_dir, all_episodes):
    successful_episodes = []
    failed_episodes = []
    for episode in all_episodes:
        episode_path = os.path.join(data_dir, episode)
        left_tracking_failed = os.path.join(episode_path, "left_tracking_lost.txt")
        right_tracking_failed = os.path.join(episode_path, "right_tracking_lost.txt")
        if os.path.exists(left_tracking_failed) or os.path.exists(right_tracking_failed):
            os.rename(episode_path, os.path.join(data_dir, "failed_" + episode))
            failed_episodes.append(episode)
            continue
        successful_episodes.append(episode)
    print(f"Tracking Failed Episodes: {failed_episodes}")
    return successful_episodes

def generate_diagnostics(data_dir):
    print(colored("GENERATING DIAGNOSTICS...", "blue"))
    episodes = [ep for ep in os.listdir(data_dir) if "ep" in ep]
    num_episodes = len(episodes)
    slam_failed = []
    tracker_failed = []
    glove_failed = []
    clip_failed = []
    successful_episodes = []
    failed_episodes = []
    for ep in episodes:
        if ep.startswith("failed"):
            failed_episodes.append(ep)
            # figure out why it failed
            zed_pose_path = os.path.join(data_dir, ep, "zed", "zed_pose.pkl")
            left_tracking_lost_path = os.path.join(data_dir, ep, "left_tracking_lost.txt")
            right_tracking_lost_path = os.path.join(data_dir, ep, "right_tracking_lost.txt")
            # left_glove_path = os.path.join(data_dir, ep, "left_leapv2", "left_leapv2.pkl")
            right_glove_path = os.path.join(data_dir, ep, "right_leapv2", "right_leapv2.pkl")
            if not os.path.exists(zed_pose_path):
                slam_failed.append(ep)
                continue
            if os.path.exists(left_tracking_lost_path) or os.path.exists(right_tracking_lost_path):
                tracker_failed.append(ep)
                continue
            if not os.path.exists(right_glove_path):
                glove_failed.append(ep)
                continue
        else:
            successful_episodes.append(ep)
    
    print(f"{len(successful_episodes)} out of {num_episodes} episodes were successful")
    print(f"{len(failed_episodes)} out of {num_episodes} episodes failed")
    print(f"{len(slam_failed)} episodes failed SLAM")
    print(f"{len(tracker_failed)} episodes failed tracking")
    print(f"{len(glove_failed)} episodes failed glove")
    
    print("SLAM Failed Episodes: ", slam_failed)
    print("Tracker Failed Episodes: ", tracker_failed)
    print("Glove Failed Episodes: ", glove_failed)
    
    # save the diagnostics
    with open(os.path.join(data_dir, "stats", "diagnostics.txt"), 'w') as f:
        f.write(f"{len(successful_episodes)} out of {num_episodes} episodes were successful\n")
        f.write(f"{len(failed_episodes)} out of {num_episodes} episodes failed\n")
        f.write(f"{len(slam_failed)} episodes failed SLAM\n")
        f.write(f"{len(tracker_failed)} episodes failed tracking\n")
        f.write(f"{len(glove_failed)} episodes failed glove\n")
        f.write(f"{len(clip_failed)} episodes failed clipping\n")
        
        # write the lists
        f.write("SLAM Failed Episodes: " + str(slam_failed) + "\n")
        f.write("Tracker Failed Episodes: " + str(tracker_failed) + "\n")
        f.write("Glove Failed Episodes: " + str(glove_failed) + "\n")

def undo_failed(data_dir):
    all_episodes = os.listdir(data_dir)
    all_episodes = [ep for ep in all_episodes if "failed" in ep]
    for episode in all_episodes:
        assert "failed" in episode, f"Episode {episode} has not failed"
        episode_path = os.path.join(data_dir, episode)
        episode_number  = episode.split('_')[-1]
        os.rename(episode_path, os.path.join(data_dir, "ep_" + episode_number))
            
def reset_dataset(data_dir, all_episodes):
    
    # first undo slam failed
    undo_failed(data_dir, all_episodes)
    
    all_episodes = index_episodes(data_dir)
    
    # then delete the following folders
    for episode in tqdm(all_episodes):
        episode_path = os.path.join(data_dir, episode)
        
        left_leapv2_path = os.path.join(episode_path, "left_leapv2")
        left_tracker_path = os.path.join(episode_path, "left_tracker")
        
        right_leapv2_path = os.path.join(episode_path, "right_leapv2")
        right_tracker_path = os.path.join(episode_path, "right_tracker")
        
        intergripper_path = os.path.join(episode_path, "intergripper")
        zed_obs_path = os.path.join(episode_path, "zed_obs")
        
        paths = [zed_obs_path, left_leapv2_path, left_tracker_path,
                   right_leapv2_path, right_tracker_path, intergripper_path, zed_obs_path]
        
        for path in paths:
            if os.path.exists(path):
                os.system(f"rm -rf {path}")
                
        # remove all the mp4 files in the episode directory
        mp4_files = [f for f in os.listdir(episode_path) if f.endswith('.mp4')]
        for mp4_file in mp4_files:
            os.remove(os.path.join(episode_path, mp4_file))

def check_episode_lengths(data_dir, all_episodes):
    episode_lengths = {}
    for episode in all_episodes:
        episode_path = os.path.join(data_dir, episode)
        timesteps_path = os.path.join(episode_path, "timesteps", "timesteps.txt")
        if not os.path.exists(timesteps_path):
            print(f"Episode {episode} does not have timesteps")
            continue
        try:
            timesteps = np.loadtxt(timesteps_path)
            episode_lengths[episode] = np.shape(timesteps)[0]
        except:
            print(f"Episode {episode_path} timesteps could not be loaded")
    
    # sort by episode length
    # episode_lengths = dict(sorted(episode_lengths.items(), key=lambda item: item[1]))
    
    # plot a bar graph of episode lengths
    plt.bar(episode_lengths.keys(), episode_lengths.values())
    plt.xlabel("Episode")
    # rotate x labels
    plt.xticks(rotation=90)
    # make labels smaller
    plt.tick_params(axis='x', which='major', labelsize=5)
    plt.ylabel("Length")
    plt.title("Episode Lengths")
    save_path = os.path.join(data_dir, "stats", "episode_lengths.png")
    plt.savefig(save_path)
    
    plt.close('all')
    
    # make a histogram of episode lengths
    plt.hist(episode_lengths.values(), bins=20)
    plt.xlabel("Episode Length")
    plt.ylabel("Frequency")
    plt.title("Episode Length Histogram")
    save_path = os.path.join(data_dir, "stats", "episode_lengths_hist.png")
    plt.close('all')
    
    # save a text file of episode lengths
    with open(os.path.join(data_dir, "stats", "episode_lengths.txt"), 'w') as f:
        for key in episode_lengths.keys():
            f.write(f"{key}: {episode_lengths[key]}\n")

    return episode_lengths