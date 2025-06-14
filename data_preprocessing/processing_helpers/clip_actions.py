import os
from tqdm import tqdm
import numpy as np
from dexwild_utils.pose_utils import poses7d_to_mats, mats_to_poses7d, abs_to_rel_poses, rel_to_abs_poses, pose7d_to_mat, save_one_path3d
from dexwild_utils.data_processing import load_pickle, save_pickle, clip_one_episode, get_clip_thresh

def clip_eef_actions(data_dir, all_episodes, left_arm, right_arm):
    # first pass to convert all actions to relative
    
    np.set_printoptions(precision=6 ,suppress=True)
    
    arm_dict = {"left": left_arm, "right": right_arm}
    
    for key in arm_dict.keys():
        if arm_dict[key] == True:
            all_actions = []
            first_eef_poses = {}
            rel_actions = {}
            
            # first pass to convert all actions to relative for stats calculations
            for episode in tqdm(all_episodes):
                episode_path = os.path.join(data_dir, episode)
                eef_actions_path = os.path.join(episode_path, f"{key}_tracker", f"{key}_tracker_world_abs.pkl")
            
                if os.path.exists(eef_actions_path):
                    eef_actions = load_pickle(eef_actions_path)
                    timestamps = eef_actions[:, 0]
                    poses = eef_actions[:, 1:]
                    
                    pose_mats = poses7d_to_mats(poses)
                    
                    first_eef_poses[episode] = pose_mats[0]
                    
                    rel_poses_mats = abs_to_rel_poses(pose_mats)
                    
                    rel_poses =  mats_to_poses7d(rel_poses_mats)
                    
                    all_actions.extend(rel_poses)
                    rel_actions[episode] = np.hstack((timestamps.reshape(-1, 1), rel_poses))

                    save_path = os.path.join(data_dir, episode, f"{key}_tracker", f"{key}_tracker_world_rel.pkl")
              
                    save_pickle(rel_actions[episode], save_path)
            
            all_actions = np.array(all_actions)
            print(f"Total number of {key} actions: {all_actions.shape[0]}")
            print(all_actions.shape)
            
            trans_thresh, rot_thresh = get_clip_thresh(data_dir, all_actions, percentile=99.0)
            
            trans_thresh = min(0.015, trans_thresh) # hard limit
            rot_thresh = min(0.1, rot_thresh) # hard limit
            
            clipped_rel_actions = {}
            
            print(f"{len(rel_actions.keys())} episodes to clip")
            
            failed_episodes = []
            
            # second pass to clip actions
            for episode in rel_actions.keys():
                timestamps = rel_actions[episode][:, 0]
                poses = rel_actions[episode][:, 1:]
                clipped_poses, num_clipped = clip_one_episode(poses, trans_thresh, rot_thresh)
                print(f"Episode {episode} clipped {num_clipped} / {poses.shape[0]} actions, {num_clipped / poses.shape[0] * 100:.2f}%")
                if num_clipped / poses.shape[0] > 0.10:
                    failed_episodes.append(episode)
                clipped_rel_actions = np.hstack((timestamps.reshape(-1, 1), clipped_poses))
                # save the clipped actions
                clipped_abs_actions = np.hstack((timestamps.reshape(-1, 1), mats_to_poses7d(rel_to_abs_poses(poses7d_to_mats(clipped_poses), first_eef_poses[episode]))))
                
                clipped_rel_save_path = os.path.join(data_dir, episode, f"{key}_tracker", f"{key}_tracker_world_clipped_rel.pkl")
                clipped_abs_save_path = os.path.join(data_dir, episode, f"{key}_tracker", f"{key}_tracker_world_clipped_abs.pkl")
                
                save_pickle(clipped_rel_actions, clipped_rel_save_path)
                save_pickle(clipped_abs_actions, clipped_abs_save_path)
                save_one_path3d(clipped_abs_actions[:, 1:4], os.path.join(data_dir, episode, f"{key}_tracker", f"{key}_tracker_world_clipped_abs.png"))
                
        
                eef_raw_actions_path = os.path.join(data_dir, episode, f"{key}_tracker", f"{key}_tracker_cam_frame_abs.pkl")
                if os.path.exists(eef_raw_actions_path):
                    raw_eef_actions = load_pickle(eef_raw_actions_path)
                    first_raw_pose = raw_eef_actions[0, 1:]
                    clipped_raw_abs_actions = np.hstack((timestamps.reshape(-1, 1), mats_to_poses7d(rel_to_abs_poses(poses7d_to_mats(clipped_poses), pose7d_to_mat(first_raw_pose)))))
                    save_pickle(clipped_raw_abs_actions, os.path.join(data_dir, episode, f"{key}_tracker", f"{key}_tracker_cam_frame_clipped_abs.pkl"))
                    save_one_path3d(clipped_raw_abs_actions[:, 1:4], os.path.join(data_dir, episode, f"{key}_tracker", f"{key}_tracker_cam_frame_clipped_abs.png"))
                else:
                    print(f"Episode {episode} does not have raw eef actions")
            
            print(f"{key} arm | {len(failed_episodes)} Over-Clipped Episodes: {failed_episodes}")