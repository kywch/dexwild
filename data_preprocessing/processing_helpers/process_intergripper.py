import os
import time
import numpy as np
import ray
from tqdm import tqdm

from dexwild_utils.data_processing import load_pickle, save_pickle, auto_match
from dexwild_utils.pose_utils import pose7d_to_mat, mat_to_pose7d, save_paths3d

FIXED_LEFT_TO_RIGHT_EEF_SHIFT = 0.46  # Fixed shift to transform the frames

def process_intergripper_one_episode(episode_path, is_robot=False):
    start_time = time.time()    
    try: 
        os.makedirs(os.path.join(episode_path, "intergripper"), exist_ok=True)
        intergripper_poses = None
        
        episode = os.path.basename(episode_path)
        
        if is_robot:
            left_eef_actions_path = os.path.join(episode_path, "left_arm_eef", "left_arm_eef.pkl")
            right_eef_actions_path = os.path.join(episode_path, "right_arm_eef", "right_arm_eef.pkl")
            
            if os.path.exists(left_eef_actions_path) and os.path.exists(right_eef_actions_path):
                left_eef_actions = load_pickle(left_eef_actions_path)
                right_eef_actions = load_pickle(right_eef_actions_path)
                
                if left_eef_actions.shape[0] != right_eef_actions.shape[0]:
                    print(f"Automatching using timestamps for episode {episode}")
                    left_eef_actions, right_eef_actions = auto_match(left_eef_actions, right_eef_actions, left_eef_actions[:, 0], right_eef_actions[:, 0])
                
                timestamps = left_eef_actions[:, 0]
                left_eef_actions = left_eef_actions[:, 1:]
                right_eef_actions = right_eef_actions[:, 1:]
                
                left_eef_actions[:, 1] += FIXED_LEFT_TO_RIGHT_EEF_SHIFT # shift to transform the frames
                
                save_paths3d([left_eef_actions, right_eef_actions], os.path.join(episode_path, "intergripper", "intergripper.png"))
                
                intergripper_poses = np.zeros_like(left_eef_actions)
                
                for i in range(left_eef_actions.shape[0]):
                    left_eef_pose = left_eef_actions[i]
                    right_eef_pose = right_eef_actions[i]
                    
                    left_eef_pose_mat = pose7d_to_mat(left_eef_pose)
                    right_eef_pose_mat = pose7d_to_mat(right_eef_pose)
                    
                    intergripper_pose_mat = np.linalg.inv(right_eef_pose_mat) @ left_eef_pose_mat 
                    
                    intergripper_pose = mat_to_pose7d(intergripper_pose_mat)
                    intergripper_poses[i] = intergripper_pose
            else:
                print(f"Episode {episode} does not have left and right eef actions")
        else:
            left_tracker_path = os.path.join(episode_path, "left_tracker", "left_tracker_cam_frame_clipped_abs.pkl")
            right_tracker_path = os.path.join(episode_path, "right_tracker", "right_tracker_cam_frame_clipped_abs.pkl")
            
            if os.path.exists(left_tracker_path) and os.path.exists(right_tracker_path):
                left_tracker = load_pickle(left_tracker_path)
                right_tracker = load_pickle(right_tracker_path)
                
                if left_tracker.shape[0] != right_tracker.shape[0]:
                    print(f"Automatching using timestamps for episode {episode}")
                    left_tracker, right_tracker = auto_match(left_tracker, right_tracker, left_tracker[:, 0], right_tracker[:, 0])
                
                timestamps = left_tracker[:, 0]
                left_tracker = left_tracker[:, 1:]
                right_tracker = right_tracker[:, 1:]
                
                save_paths3d([left_tracker, right_tracker], os.path.join(episode_path, "intergripper", "intergripper.png"))
                
                intergripper_poses = np.zeros_like(left_tracker)
                
                for i in range(left_tracker.shape[0]):
                    left_pose = left_tracker[i]
                    right_pose = right_tracker[i]
                    
                    left_pose_mat = pose7d_to_mat(left_pose)
                    right_pose_mat = pose7d_to_mat(right_pose)
                    
                    intergripper_pose_mat = np.linalg.inv(right_pose_mat) @ left_pose_mat 
                    # gives the pose of the left gripper in the right gripper frame
                    
                    intergripper_pose = mat_to_pose7d(intergripper_pose_mat)
                    intergripper_poses[i] = intergripper_pose
            else:
                print(f"Episode {episode} does not have left and right tracker poses")
                
        intergripper_poses = np.hstack((timestamps.reshape(-1, 1), intergripper_poses))
        save_path = os.path.join(episode_path, "intergripper", "intergripper.pkl")
        save_pickle(intergripper_poses, save_path)
        
    except Exception as e:
        print(f"Intergripper: Episode {episode} Failed, Error {e}")
    
    return time.time() - start_time
        
@ray.remote
def ray_process_intergripper_one_episode(episode_path, is_robot=False):
    return process_intergripper_one_episode(episode_path, is_robot)
    
def process_intergripper_wrapper(data_dir, all_episodes, is_robot=False, parallelize=True):
    
    if parallelize:
        print(f"Processing {len(all_episodes)} episodes for intergripper (Parallelized with Ray)")
        ray.init() 
        print(ray.cluster_resources())
        
        batch_size = min(len(all_episodes), 200)
    
        batches = [all_episodes[i:i + batch_size] for i in range(0, len(all_episodes), batch_size)]
        
        for batch in batches:
            # Create Ray tasks
            futures = []
            
            for episode in batch:
                
                episode_path = os.path.join(data_dir, episode)
                futures.append(ray_process_intergripper_one_episode.remote(episode_path, is_robot))
                    
            results = []
            for fut in tqdm(futures, desc="Processing Trackers", total=len(futures)):
                # Ray get one by one for nice progress
                res = ray.get(fut)  
                results.append(res)

            # Print final summary
            print("All Intergrippers Processed!")
            
        # Shutdown Ray
        ray.shutdown()
    else:
        print(f"Processing {len(all_episodes)} episodes for intergripper (Sequentially)")
        raise NotImplementedError
