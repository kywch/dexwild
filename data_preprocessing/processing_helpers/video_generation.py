import os
import time
import numpy as np
import ray
import cv2
from tqdm import tqdm

from dexwild_utils.data_processing import auto_match
from dexwild_utils.ZedProcessor import get_videos_only

def match_images(imgs1_path, imgs2_path, imgs1, imgs2, time1, time2):
    
    og_imgs1 = imgs1.copy()
    og_imgs2 = imgs2.copy()
    
    imgs1, imgs2 = auto_match(imgs1, imgs2, time1, time2)
            
    if og_imgs1.shape[0] < og_imgs2.shape[0]:
        for i in range(og_imgs2.shape[0]):
            if og_imgs2[i] not in imgs2:
                os.remove(os.path.join(imgs2_path, og_imgs2[i]))
                print(f"Deleted {og_imgs2[i]}")
    else:
        for i in range(og_imgs1.shape[0]):
            if og_imgs1[i] not in imgs1:
                os.remove(os.path.join(imgs1_path, og_imgs1[i]))
                print(f"Deleted {og_imgs1[i]}")
        
    # check that the lengths are the same
    imgs1_check = np.array(sorted(os.listdir(imgs1_path), key=lambda x: int(x.split('.')[0])))
    imgs2_check = np.array(sorted(os.listdir(imgs2_path), key=lambda x: int(x.split('.')[0])))
    
    assert len(imgs1_check) == len(imgs2_check)
    # print(f"Auto-matched timestamps. New shapes: Imgs1: {imgs1_check.shape[0]}, Imgs2: {imgs2_check.shape[0]}")
    
    return imgs1_check, imgs2_check

def all_videos_from_one_svo(data_dir, all_episodes, svo_path, left_hand, right_hand):
    try:
        start_episode = os.path.basename(svo_path).split("_")[-2]
        end_episode = os.path.basename(svo_path).split("_")[-1].split(".")[0]
        episodes = range(int(start_episode), int(end_episode) + 1)
    except:
        print("Failed to handle SVO name")
        return
    
    # get the paths to the epsidoes
    timestamp_dict = {}
    
    episode_paths = {}
    for folder in all_episodes:
        for episode in episodes:
            # find the episode in the data_dir, might be failed_ep_xx
            episode_str = f"ep_{episode}"
            if folder.endswith(episode_str):
                episode_paths[episode_str] = os.path.join(data_dir, folder)
                # get the timestamps
                timestamps_path = os.path.join(episode_paths[episode_str], "timesteps", "timesteps.txt")
                if os.path.exists(timestamps_path):
                    timestamps = np.loadtxt(timestamps_path)
                    timestamp_dict[episode_str] = timestamps
                else:
                    print(f"Episode {episode} does not have timestamps")
                break
            
    # sort the keys in timstamp dict
    video_path = os.path.join(data_dir, "videos")
    get_videos_only(svo_path, video_path, timestamp_dict)

def generate_all_videos(data_dir, all_svos, left_hand, right_hand, parallelize=False):
    all_episodes = os.listdir(data_dir)
    all_episodes = [episode for episode in all_episodes if 'ep' in episode and os.path.isdir(os.path.join(data_dir, episode))]
    all_episodes = sorted(all_episodes, key=lambda x: int(x.split('_')[-1]))
    
    video_path = os.path.join(data_dir, "videos")
    if os.path.exists(video_path):
        print("Video path already exists, skipping video generation")
        return
    os.makedirs(video_path, exist_ok=True)
    
    slam_dir = os.path.join(data_dir, "zed_recordings")
    
    if parallelize:
        batch_size = min(len(all_svos), 20)
        
        batches = [all_svos[i:i + batch_size] for i in range(0, len(all_svos), batch_size)]
        
        print(f"Number of batches: {len(batches)}")
        
        # Initialize Ray
        ray.init(num_gpus = 1)  # Or ray.init(address="auto") if you're using a Ray cluster
        print(ray.cluster_resources()) 
        
        @ray.remote(num_gpus=1/30, max_retries=0)
        def parallel_all_videos_from_one_svo(data_dir, all_episodes, svo_path, left_hand, right_hand):
            all_videos_from_one_svo(data_dir, all_episodes, svo_path, left_hand, right_hand)
        
        for batch in batches:
            # Create Ray tasks
            futures = []
            
            for slam_svo in batch:
                if "output" in slam_svo:
                    start = time.time()
                    svo_path = os.path.join(slam_dir, slam_svo)
                    futures.append(parallel_all_videos_from_one_svo.remote(data_dir, all_episodes, svo_path, left_hand, right_hand))
            
            results = []
            for fut in tqdm(futures, desc="Processing SVOs", total=len(futures)):
                # Ray get one by one for nice progress
                try:
                    res = ray.get(fut)  
                    results.append(res)
                except ray.exceptions.WorkerCrashedError:
                    print("Worker failed, skipping this task.")
                except Exception as e:
                    print(f"Task failed: {e}")
            
        # Shutdown Ray
        ray.shutdown()
    else:
        for svo in all_svos:
            svo_path = os.path.join(slam_dir, svo)
            print(svo_path)
            all_videos_from_one_svo(data_dir, all_episodes, svo_path, left_hand, right_hand)
    
    # get the hand cameras
    generate_hand_videos(data_dir, all_episodes, left_hand, right_hand, is_robot=False, save_dir=video_path)
    
def generate_hand_videos(data_dir, all_episodes, left_hand, right_hand, is_robot=False, save_dir=None):
    episodes = [ep for ep in all_episodes if "ep" in ep]
    
    print("GENERATING HAND VIDEOS...")
    
    if is_robot:
        fps = 30
    else:
        fps = 60
    
    if save_dir is None:
        use_episode_dir = True
    else:
        use_episode_dir = False
        
    print("episodes: ", episodes)
    
    for episode in tqdm(episodes):
        try:
            episode_path = os.path.join(data_dir, episode)
            
            episode_name = "ep_" + episode.split("_")[-1]
            
            if use_episode_dir:
                save_dir = episode_path

            right_pinky_cam_path = os.path.join(episode_path, "right_pinky_cam")
            right_thumb_cam_path = os.path.join(episode_path, "right_thumb_cam")
            left_pinky_cam_path = os.path.join(episode_path, "left_pinky_cam")
            left_thumb_cam_path = os.path.join(episode_path, "left_thumb_cam")
            
            if right_hand and os.path.exists(right_pinky_cam_path) and os.path.exists(right_thumb_cam_path):
                right_pinky_cam_images = np.array(sorted(os.listdir(right_pinky_cam_path), key=lambda x: int(x.split('.')[0])))
                right_pinky_cam_timestamps = np.array([int(x.split('.')[0]) for x in right_pinky_cam_images])

                right_thumb_cam_images = np.array(sorted(os.listdir(right_thumb_cam_path), key=lambda x: int(x.split('.')[0])))
                right_thumb_cam_timestamps = np.array([int(x.split('.')[0]) for x in right_thumb_cam_images])
                
                # automatch the images
                if right_pinky_cam_images.shape[0] != right_thumb_cam_images.shape[0]:
                    right_pinky_cam_images, right_thumb_cam_images =  match_images(right_pinky_cam_path, right_thumb_cam_path, right_pinky_cam_images, right_thumb_cam_images, right_pinky_cam_timestamps, right_thumb_cam_timestamps)
            
            if left_hand and os.path.exists(left_pinky_cam_path) and os.path.exists(left_thumb_cam_path):
                # sort the images
                left_pinky_cam_images = np.array(sorted( os.listdir(left_pinky_cam_path), key=lambda x: int(x.split('.')[0])))
                left_pinky_cam_timestamps = np.array([int(x.split('.')[0]) for x in left_pinky_cam_images])
                
                left_thumb_cam_images = np.array(sorted(os.listdir(left_thumb_cam_path), key=lambda x: int(x.split('.')[0])))
                left_thumb_cam_timestamps = np.array([int(x.split('.')[0]) for x in left_thumb_cam_images])
                
                # automatch the images
                if left_pinky_cam_images.shape[0] != left_thumb_cam_images.shape[0]:
                    left_pinky_cam_images, left_thumb_cam_images = match_images(left_pinky_cam_path, left_thumb_cam_path, left_pinky_cam_images, left_thumb_cam_images, left_pinky_cam_timestamps, left_thumb_cam_timestamps)
            
            # okay now write the videos
            if right_hand and left_hand:
                # check if the video already exists
                if os.path.exists(os.path.join(save_dir, f"bimanual_hand_cams_{episode_name}.mp4")):
                    print(f"Video already exists for episode {episode_name}")
                    continue
                
                if right_pinky_cam_images.shape[0] != left_pinky_cam_images.shape[0]:
                    left_pinky_cam_timestamps = np.array([int(x.split('.')[0]) for x in left_pinky_cam_images])
                    left_thumb_cam_timestamps = np.array([int(x.split('.')[0]) for x in left_thumb_cam_images])
                    right_pinky_cam_timestamps = np.array([int(x.split('.')[0]) for x in right_pinky_cam_images])
                    right_thumb_cam_timestamps = np.array([int(x.split('.')[0]) for x in right_thumb_cam_images])
                    try:
                        # left right dont match
                        right_pinky_cam_images, left_pinky_cam_images = match_images(right_pinky_cam_path, left_pinky_cam_path, right_pinky_cam_images, left_pinky_cam_images, right_pinky_cam_timestamps, left_pinky_cam_timestamps)
                        right_thumb_cam_images, left_thumb_cam_images = match_images(right_thumb_cam_path, left_thumb_cam_path, right_thumb_cam_images, left_thumb_cam_images, right_thumb_cam_timestamps, left_thumb_cam_timestamps)
                    except Exception as e:
                        print(f"Episode {episode} failed to match images: {e}")
                        breakpoint()
                assert (right_pinky_cam_images.shape[0] == right_thumb_cam_images.shape[0] == left_pinky_cam_images.shape[0] == left_thumb_cam_images.shape[0]), f"Right Pinky: {right_pinky_cam_images.shape[0]}, Right Thumb: {right_thumb_cam_images.shape[0]}, Left Pinky: {left_pinky_cam_images.shape[0]}, Left Thumb: {left_thumb_cam_images.shape[0]}"
                
                vid_writer = cv2.VideoWriter(os.path.join(save_dir, f"bimanual_hand_cams_{episode_name}.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), fps, (1280, 240))
                
                for i in range(right_pinky_cam_images.shape[0]):
                    right_pinky_img = cv2.imread(os.path.join(right_pinky_cam_path, right_pinky_cam_images[i]))
                    right_thumb_img = cv2.imread(os.path.join(right_thumb_cam_path, right_thumb_cam_images[i]))
                    left_pinky_img = cv2.imread(os.path.join(left_pinky_cam_path, left_pinky_cam_images[i]))
                    left_thumb_img = cv2.imread(os.path.join(left_thumb_cam_path, left_thumb_cam_images[i]))
                    
                    combined_img = np.hstack((left_pinky_img, left_thumb_img, right_thumb_img, right_pinky_img))
                    
                    vid_writer.write(combined_img)
                    
                vid_writer.release()
            
            # sort the images
            elif right_hand and not left_hand:    
                if os.path.exists(os.path.join(save_dir, f"right_hand_cams_{episode_name}.mp4")):
                    print(f"Video already exists for episode {episode_name}")
                    continue
                
                # make the video #put the two images side by side (each image is 320x240)
                vid_writer = cv2.VideoWriter(os.path.join(save_dir, f"right_hand_cams_{episode_name}.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), fps, (640, 240))
                for i in range(right_pinky_cam_images.shape[0]):
                    right_pinky_img = cv2.imread(os.path.join(right_pinky_cam_path, right_pinky_cam_images[i]))
                    right_thumb_img = cv2.imread(os.path.join(right_thumb_cam_path, right_thumb_cam_images[i]))
                    # put the two images side by side
                    combined_img = np.hstack((right_thumb_img, right_pinky_img))
                    vid_writer.write(combined_img)
                    
                vid_writer.release()
                
            elif left_hand and not right_hand:
                if os.path.exists(os.path.join(save_dir, f"left_hand_cams_{episode_name}.mp4")):
                    print(f"Video already exists for episode {episode_name}")
                    continue
                # make the video #put the two images side by side (each image is 320x240)   
                vid_writer = cv2.VideoWriter(os.path.join(save_dir, f"left_hand_cams_{episode_name}.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), fps, (640, 240))
                for i in range(left_pinky_cam_images.shape[0]):
                    left_pinky_img = cv2.imread(os.path.join(left_pinky_cam_path, left_pinky_cam_images[i]))
                    left_thumb_img = cv2.imread(os.path.join(left_thumb_cam_path, left_thumb_cam_images[i]))
                    # put the two images side by side
                    combined_img = np.hstack((left_thumb_img, left_pinky_img))
                    vid_writer.write(combined_img)
                vid_writer.release()
        except Exception as e:
            print(f"Episode {episode} failed to generate hand videos: {e}")
            continue