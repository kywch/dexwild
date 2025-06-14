# Copyright (c) 2025 Tony Tao
# Licensed under the MIT License. See LICENSE file in the project root for details.

"""
This script processes demonstration data from human and/or robot and
converts them into a robobuf-format replay buffer. It includes support for:

- Subsampling and temporal skipping of trajectory steps
- Processing rotation representations (quaternion, Euler, 6D)
- Unifying robot and human key formats
- Safe value range checks (NaN/Inf/clipping)
- Image and language observation loading
- Relative and absolute action processing modes
- Optional upsampling of specific trajectory segments

Usage:
    python dataset_to_robobuf.py \
        --retain_pct 1.0 \
        --upsample_multiplier 0 \
        --mode abs \
        --data /path/to/data \
        --buf_name my_buffer \
        --rot_repr rot6d \
        --subsample_rate 1.0 \
        --language \
        --human --left_hand --right_hand --upsample_pcts 0.6 1.0

Outputs:
    - Processed robobuf pickle saved to: data_buffers/<buf_name>/buffer/buffer.pkl
    - Summary statistics for states and actions
    - Optionally saves buffer size and action histograms
"""

import os
import copy
import pickle as pkl
import argparse
import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from tqdm import tqdm
from PIL import Image
from scipy.spatial.transform import Rotation as R
from termcolor import colored

from robobuf.buffers import ObsWrapper, Transition, ReplayBuffer
from dexwild_utils.pose_utils import (
    pose6d_to_mat, mat_to_pose6d, mat_to_rot6d, mat_to_pose9d, 
    mat_to_pose7d, pose9d_to_mat, pose7d_to_mat
)


def mat_to_pose_func(pose_repr="euler"):
    if pose_repr == "euler":
        return mat_to_pose6d
    elif pose_repr == "rot6d":
        return mat_to_pose9d
    elif pose_repr == "quat":
        return mat_to_pose7d
    else:
        raise ValueError(f"Unknown pose representation: {pose_repr}")

def pose_to_mat_func(pose_repr="euler"):
    if pose_repr == "euler":
        return pose6d_to_mat
    elif pose_repr == "rot6d":
        return pose9d_to_mat
    elif pose_repr == "quat":
        return pose7d_to_mat
    else:
        raise ValueError(f"Unknown pose representation: {pose_repr}")

def safe_add(buffer, obs, action, reward, first, traj_folder, step_idx):
    # check images
    for cam_i in range(len([k for k in obs.obs if k.startswith("cam")])):
        obs.image(cam_i)           # will raise if corrupt
    # commit
    buffer.add(Transition(obs, action, reward), is_first=first)

def check_nan_inf(name, arr, traj_folder, step_idx, threshold=100.0):
    if np.isnan(arr).any():
        print(colored(f"[NaN DETECTED] in {name} at step {step_idx} in {traj_folder}", "red"))
        return True
    if np.isinf(arr).any():
        print(colored(f"[Inf DETECTED] in {name} at step {step_idx} in {traj_folder}", "red"))
        return True
    if (arr > threshold).any() or (arr < -threshold).any():
        max_val = np.max(arr)
        min_val = np.min(arr)
        print(colored(f"[VALUE OUT OF RANGE] in {name} at step {step_idx} in {traj_folder}", "yellow"))
        print(colored(f"    Max value: {max_val}, Min value: {min_val} (Threshold: ±{threshold})", "yellow"))
        return True
    return False
    
def plot_action_histograms(buffers, output_dir):
    """Plot histograms for each dimension of the actions."""
    for name, (buffer, _) in buffers.items():
        actions = [transition.action for transition in buffer]
        actions = np.array(actions)

        num_dims = actions.shape[1]
        fig, axs = plt.subplots(1, num_dims, figsize=(num_dims * 5, 5))
        if num_dims == 1:
            axs = [axs]
        
        for i in range(num_dims):
            axs[i].hist(actions[:, i], bins=50, alpha=0.7, color='blue', edgecolor='black')
            axs[i].set_title(f'{name} - Dimension {i + 1}')
            axs[i].set_xlabel('Value')
            axs[i].set_ylabel('Frequency')

        plt.tight_layout()
        fig_path = output_dir / f"{name}_action_histograms.png"
        plt.savefig(fig_path)
        plt.close(fig)
        print(f"Histogram saved to {fig_path}")

def process_action(actions_dict, step_idx, rot_repr="euler"):
    actions_dict_result = {}
    for action_key in actions_dict.keys():
        prev_action = actions_dict[action_key][step_idx - 1]
        action = actions_dict[action_key][step_idx]
        if "eef" in action_key:
            if MODE == "rel":
                prev_action_mat = pose_to_mat_func(rot_repr)(prev_action)
                curr_action_mat = pose_to_mat_func(rot_repr)(action)
                # calculate relative pose
                rel_mat = np.linalg.inv(prev_action_mat) @ curr_action_mat
                rel_pose = mat_to_pose_func(rot_repr)(rel_mat)
                actions_dict_result[action_key] = rel_pose
            
            elif MODE == "abs" or MODE == "hybrid":
                actions_dict_result[action_key] = action
            else:
                raise ValueError(f"Unknown mode: {MODE}")
        else:
            # just take absolute if not eef
            actions_dict_result[action_key] = action
    
    return actions_dict_result

def process_all_actions(actions_dict, rot_repr):
    actions_dict_result = copy.deepcopy(actions_dict) # Watch out for aliasing
    
    total_steps = len(actions_dict[next(iter(actions_dict))])
    for step_idx in range(1, total_steps):
        processed_action = process_action(actions_dict, step_idx, rot_repr)
        
        for action_key in actions_dict.keys():
            actions_dict_result[action_key][step_idx] = processed_action[action_key]
            
    # remove the first actions_dict
    for action_key in actions_dict.keys():
        actions_dict_result[action_key] = actions_dict_result[action_key][1:]
        
    return actions_dict_result

def proc_rotation(quat_cols, rot_repr):
    if rot_repr == "euler":
        angles = R.from_quat(quat_cols).as_euler('xyz')
    elif rot_repr == "rot6d":
        angles = mat_to_rot6d(R.from_quat(quat_cols).as_matrix())
    elif rot_repr == "quat":
        angles = quat_cols
    else:
        raise ValueError(f"Unknown rotation representation: {rot_repr}")
    return angles

def human_key_to_fname(keys):
    fnames = []
    for key in keys:
        if "right_tracker" in key:
            fnames.append("right_tracker_world_clipped_abs")
        elif "left_tracker" in key:
            fnames.append("left_tracker_world_clipped_abs")
        else:
            fnames.append(key)
    return fnames

def key_to_policy_key(key, human):
    if not human: # just keep the robot keys
        return key
    else: # convert human key to key used in policy
        if key == "right_tracker":
            return "right_arm_eef"
        elif key == "left_tracker":
            return "left_arm_eef"
        
        return key

def process_trajectory(traj_folder, state_keys, action_keys, lang_keys, img_keys, rot_repr="euler", subsample_rate=1.0, human=False):
    """Process a single trajectory folder and return actions and camera images."""
    
    if human:
        state_fnames = human_key_to_fname(state_keys)
        action_fnames = human_key_to_fname(action_keys)
        lang_fnames = human_key_to_fname(lang_keys)
    else:
        state_fnames = state_keys
        lang_fnames = lang_keys
        action_fnames = action_keys
    
    skip_rate  = int(1 / subsample_rate)
    
    # Load actions
    actions = {}
    for i, action_key in enumerate(action_keys):
        with open(f"{traj_folder}/{action_key}/{action_fnames[i]}.pkl", "rb") as f:
            action_data = pkl.load(f)
        action_data = action_data[:, 1:] # Remove timestamp column

        policy_key = key_to_policy_key(action_key, human) # convert from action key to policy key so human and robot are unified
        
        if "eef" in policy_key or "intergripper" in policy_key:
            # Convert quaternions to Euler angles for end effector actions
            quat_cols = action_data[:, -4:]
            angles = proc_rotation(quat_cols, rot_repr)
            action_data = np.column_stack([action_data[:, :-4], angles])

        actions[policy_key] = action_data[::skip_rate, :]
    
    states = {}
    for i, state_key in enumerate(state_keys):
        with open(f"{traj_folder}/{state_key}/{state_fnames[i]}.pkl", "rb") as f:
            state_data = pkl.load(f)
        state_data = state_data[:, 1:]
        
        policy_key = key_to_policy_key(state_key, human) # convert from state key to policy key so human and robot are unified
        
        if "eef" in policy_key or "intergripper" in policy_key:
            # Convert quaternions to Euler angles for end effector actions
            quat_cols = state_data[:, -4:]
            angles = proc_rotation(quat_cols, rot_repr)
            state_data = np.column_stack([state_data[:, :-4], angles])
        if "global_rot" in policy_key:
            quat_cols = state_data
            angles = proc_rotation(quat_cols, rot_repr)
            state_data = angles
            
        states[policy_key] = state_data[::skip_rate, :]
        
    lang = []
    if len(lang_keys) > 0:
        with open(f"{traj_folder}/{lang_keys[0]}/{lang_fnames[0]}.txt", 'r') as f:
            lang.append(f.read())

    # Ensure same length and concatenate
    act_min_len = min([len(action) for key, action in actions.items()])
    for key, action in actions.items():
        actions[key] = action[:act_min_len]
        
    if len(state_keys) > 0:
        state_min_len = min([len(state) for key, state in states.items()])
        for key, state in states.items():
            states[key] = state[:state_min_len]
        
    if len(lang_keys) > 0:
        # copy lang to same length of trajectory
        lang = lang * (len(actions))
        
        assert len(actions[next(iter(actions))]) == len(lang), f"Actions {len(actions[next(iter(actions))])} and language {len(lang)} must have the same length"
    
    #actions are right hand, right arm, left hand, left arm

    # Load camera images
    cam_imgs = {}
    for img_key in img_keys:
        img_folder = f"{traj_folder}/{img_key}"
        img_files = sorted(glob(f"{img_folder}/*.jpg"))
        cam_imgs[img_key] = img_files[::skip_rate]
        
    min_img_len = min([len(img_files) for img_key, img_files in cam_imgs.items()])
    
    for img_key, img_files in cam_imgs.items():
        cam_imgs[img_key] = img_files[:min_img_len]

    # make sure all the lengths are the same
    
    min_len = min(act_min_len, state_min_len, min_img_len)
    
    for key, action in actions.items():
        if len(action) > min_len:
            actions[key] = action[:min_len]
            
    if len(state_keys) > 0:
        for key, state in states.items():
            if len(state) > min_len:
                states[key] = state[:min_len]
                
        assert len(actions[next(iter(actions))]) == len(states[next(iter(states))]), f"Actions {len(actions[next(iter(actions))])} and states {len(states[next(iter(states))])} must have the same length"
    
    for img_key, img_files in cam_imgs.items():
        if len(img_files) > min_len:
            cam_imgs[img_key] = img_files[:min_len]
            
    assert len(actions[next(iter(actions))]) == len(cam_imgs[img_keys[0]]), f"Actions {len(actions[next(iter(actions))])} and images {len(cam_imgs[img_keys[0]])} must have the same length"
    
    # retain only retain_pct of the trajectory
    
    traj_len = len(actions[next(iter(actions))])
    # print(f"traj_len: {traj_len}")
    for key, action in actions.items():
        actions[key] = action[:int(traj_len*retain_pct)]
    for key, state in states.items():
        states[key] = state[:int(traj_len*retain_pct)]
        
    lang = lang[:int(traj_len*retain_pct)]
    
    for img_key in img_keys:
        cam_imgs[img_key] = cam_imgs[img_key][:int(traj_len*retain_pct)]
    
    # breakpoint()
    return states, actions, cam_imgs, img_keys, lang

def main(data, retain_pct, upsample_pcts, upsample_multiplier, MODE, buffer_name, action_keys, state_keys, lang_keys, img_keys, rot_repr, subsample_rate, human): 
    # please confirm settings
    print(colored(f"HUMAN data {human}", "red"))
    print(colored(f"Using mode: {MODE}", "green"))
    print(colored(f"Using retain_pct: {retain_pct}", "green"))
    print(colored(f"using Upsample pcts: {upsample_pcts}", "green"))
    print(colored(f"Using Upsample multiplier: {upsample_multiplier}", "green"))
    print(colored(f"Using data: {data}", "green"))
    print(colored(f"Action Keys: {action_keys}", "green"))
    print(colored(f"State Keys: {state_keys}", "green"))
    print(colored(f"Language Keys: {lang_keys}", "green"))
    print(colored(f"Image Keys: {img_keys}", "green"))
    print(colored(f"Using rotation representation: {rot_repr}", "green"))
    print(colored(f"Buffer name: {buffer_name}", "green"))
    print(colored(f"Subsample rate: {subsample_rate}", "green"))
    print(colored(f"Press Enter to continue...", "yellow"))
    input()

    traj_folders = glob(f"{data}/*/ep_*")
    
    buffer_folder = os.path.dirname(data)
    
    assert os.path.exists(buffer_folder), f"Buffer folder {buffer_folder} does not exist"
    
    output_dir = os.path.join(buffer_folder, "data_buffers", buffer_name)
    
    os.makedirs(output_dir, exist_ok=True)

    print(f"Using {len(traj_folders)} trajectories")

    # Initialize buffer
    rb_mixed_hand_images = ReplayBuffer()

    print("Processing and storing actions...")
    action_logger = {}
    for action_key in action_keys:
        action_logger[key_to_policy_key(action_key, human)] = []
    state_logger = {}
    for state_key in state_keys:
        state_logger[key_to_policy_key(state_key, human)] = []
        
    for traj_folder in tqdm(traj_folders):
        try:
            states, actions, cam_imgs, img_keys, lang = process_trajectory(traj_folder, state_keys, action_keys, lang_keys, img_keys, rot_repr, subsample_rate, human)
   
            len_traj = len(actions[next(iter(actions))])
            
            for step_idx in range(1, len_traj):
                try:
                    # Prepare image observations
                    obs = {}
                    for i, img_key in enumerate(img_keys):
                        img_files = cam_imgs[img_key]
                        img = Image.open(img_files[step_idx - 1])
                        img = np.array(img)
                        if img.shape[-1] == 4:
                            img = img[:, :, :3]
                        obs[f'cam{i}'] = img
                        
                    processed_actions = process_action(actions, step_idx, rot_repr)
                    
                    for key, value in processed_actions.items():
                        if check_nan_inf(f"action:{key}", value, traj_folder, step_idx):
                            raise ValueError(f"NaN/Inf in action:{key}")

                    # Check for NaNs/Infs in states
                    for key, value in states.items():
                        state_val = value[step_idx - 1]
                        if check_nan_inf(f"state:{key}", state_val, traj_folder, step_idx):
                            raise ValueError(f"NaN/Inf in state:{key}")
                    
                    for key in action_logger.keys():
                        action_logger[key].append(processed_actions[key])
                    for key in state_logger.keys():
                        state_logger[key].append(states[key][step_idx-1])
                    
                    obs_hand_images = {"state": {}}
                    # normalize the state
                    if len(state_keys) > 0:
                        for state_key in states.keys():
                            obs_hand_images["state"][state_key] = states[state_key][step_idx-1]
                    else:
                        for action_key in actions.keys():
                            obs_hand_images["state"][action_key] = actions[action_key][step_idx-1]
                        
                    if len(lang) > 0:
                        obs_hand_images['lang'] = lang[step_idx-1]
                        
                    obs_hand_images.update(obs)
                    obs_hand_images = ObsWrapper(obs_hand_images)
                            
                    # Add transition to buffer
                    is_first = (step_idx == (1))
                    is_last = (step_idx == len_traj - 1)
                    safe_add(rb_mixed_hand_images, obs_hand_images, processed_actions, is_last, is_first, traj_folder, step_idx) # add safely with checks
                    
                    # rb_mixed_hand_images.add(Transition(obs=obs_hand_images, action=processed_actions, reward=is_last), is_first=is_first)
                except Exception as e:
                    print(f"Error processing {traj_folder} at step {step_idx}: {e}")
                    continue
            
            len_traj = len(actions[next(iter(actions))]) # update length after processing
            
            for _ in range(upsample_multiplier):
                print("Upsampling...")
                if len(upsample_pcts) % 2 != 0:
                    raise ValueError("upsample_pcts must contain pairs of start and end percentages.")
                
                actual_indices = []
                start_idxs = []
                end_idxs = []
                for i in range(0, len(upsample_pcts), 2): # Iterate through pairs
                    startpct = upsample_pcts[i]
                    endpct = upsample_pcts[i+1]
                    assert 0 <= startpct < endpct <= 1, f"Invalid percentage range: {startpct}, {endpct}"
                    start_idx = int(startpct * len_traj) + 1
                    end_idx = int(endpct * len_traj)
                    start_idxs.append(start_idx)
                    end_idxs.append(end_idx)
                    actual_indices.extend(list(range(start_idx, end_idx)))
                    
                actual_indices = sorted(set(actual_indices))
                
                for step_idx in actual_indices:
                    obs = {}
                    for i, img_key in enumerate(img_keys):
                        img_files = cam_imgs[img_key]
                        img = Image.open(img_files[step_idx - 1])
                        img = np.array(img)
                        if img.shape[-1] == 4:
                            img = img[:, :, :3]
                        obs[f'cam{i}'] = img
                        
                    processed_actions = process_action(actions, step_idx, rot_repr)
                    
                    for key, value in processed_actions.items():
                        if check_nan_inf(f"action:{key}", value, traj_folder, step_idx):
                            raise ValueError(f"NaN/Inf in action:{key}")

                    # Check for NaNs/Infs in states
                    for key, value in states.items():
                        state_val = value[step_idx - 1]
                        if check_nan_inf(f"state:{key}", state_val, traj_folder, step_idx):
                            raise ValueError(f"NaN/Inf in state:{key}")
                    
                    for key in action_logger.keys():
                        action_logger[key].append(processed_actions[key])
                    for key in state_logger.keys():
                        state_logger[key].append(states[key][step_idx-1])
                    
                    obs_hand_images = {"state": {}}
                    # state should already be normalized
                    if len(state_keys) > 0:
                        for state_key in states.keys():
                            obs_hand_images["state"][state_key] = states[state_key][step_idx-1]
                        
                    else:
                        for action_key in actions.keys():
                            obs_hand_images["state"][action_key] = actions[action_key][step_idx-1]
                    
                    if len(lang) > 0:
                        obs_hand_images['lang'] = lang[step_idx-1]
                        
                    obs_hand_images.update(obs)
                    obs_hand_images = ObsWrapper(obs_hand_images)
                        
                    is_first = (step_idx in start_idxs)
                    is_last = (step_idx in end_idxs)
                    
                    safe_add(rb_mixed_hand_images, obs_hand_images, processed_actions, is_last, is_first, traj_folder, step_idx) # add safely with checks
                    
        except Exception as e:
            print(f"Error processing {traj_folder}: {e}")
            continue
    
    # get the means, stds, and  max and min values of the actions
    for key in action_logger.keys():
        print("action key:", key)
        action_data = np.array(action_logger[key])
        print(f"max value in {key}: {np.max(action_data, axis=0)}")
        print(f"min value in {key}: {np.min(action_data, axis=0)}")
        print(f"mean value in {key}: {np.mean(action_data, axis=0)}")
    
    for key in state_logger.keys():
        print("state key:", key)
        state_data = np.array(state_logger[key])
        print(f"max value in {key}: {np.max(state_data, axis=0)}")
        print(f"min value in {key}: {np.min(state_data, axis=0)}")
        print(f"mean value in {key}: {np.mean(state_data, axis=0)}")
        
    print("Saving Buffer size:", len(rb_mixed_hand_images))
    # Save buffer and normalization stats
    buffer_dir = os.path.join(output_dir, "buffer")
    os.makedirs(buffer_dir, exist_ok=True)
    
    # save the buffer size in a file
    with open(os.path.join(buffer_dir, "buffer_size.txt"), "w") as f:
        f.write(str(len(rb_mixed_hand_images)))
        
    traj_list = rb_mixed_hand_images.to_traj_list()
    print("Converted to list— length:", len(traj_list))

    # Save all at once
    with open(os.path.join(buffer_dir, "buffer.pkl"), "wb") as f:
        pkl.dump(traj_list, f)
    print(colored("Buffer saved in a single file.", "green"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process robot and human trajectory data.")
    
    parser.add_argument('--retain_pct', type=float, default=1.0, help='Percentage of trajectory to retain')
    parser.add_argument('--upsample_pcts', type=float, nargs='+', default=[0.2, 0.5], help='Percentages for upsampling (start end pairs)')
    parser.add_argument('--upsample_multiplier', type=int, default=0, help='Multiplier for upsampling')
    parser.add_argument('--mode', type=str, choices=['delta', 'rel', 'abs', 'rel_intergripper_action', 'hybrid'], default='rel', help='Mode for processing actions')
    parser.add_argument('--data', type=str, default="", help='Path to data')
    parser.add_argument('--buf_name', type=str, default="robot_only", help='Name for the output buffer directory')
    parser.add_argument("--rot_repr", type=str, default="euler", choices=["quat", "euler", "rot6d"], help="Rotation representation to use")
    parser.add_argument('--subsample_rate', type=float, default=1.0, help='Subsample rate for the data')
    parser.add_argument("--human", action='store_true', help="Whether to process human data")
    parser.add_argument('--left_hand', action='store_true', help="Process left hand data")
    parser.add_argument('--right_hand', action='store_true', help="Process right hand data")
    parser.add_argument('--language', action='store_true', help="Whether to include language data")

    args = parser.parse_args()
    retain_pct = args.retain_pct
    upsample_pcts = args.upsample_pcts
    upsample_multiplier = args.upsample_multiplier
    MODE = args.mode
    data = args.data
    buffer_name = args.buf_name
    human = args.human
    rot_repr = args.rot_repr    
    subsample_rate = args.subsample_rate
    left_hand = args.left_hand
    right_hand = args.right_hand
    language = args.language
    
    if human:
        if left_hand and right_hand:
            action_keys = ["right_leapv2", "right_tracker", "left_leapv2", "left_tracker"]
            state_keys = ["right_tracker", "left_tracker", "intergripper"]
            img_keys = ["right_pinky_cam", "right_thumb_cam", "left_pinky_cam", "left_thumb_cam"]
        elif left_hand:
            action_keys = ["left_leapv2", "left_tracker"]
            state_keys = ["left_tracker"]
            img_keys = ["left_pinky_cam", "left_thumb_cam"]
        elif right_hand:
            action_keys = ["right_leapv2", "right_tracker"]
            state_keys = ["right_tracker"]
            img_keys = ["right_pinky_cam", "right_thumb_cam"]
    if not human:
        if left_hand and right_hand:
            action_keys = ["right_leapv2", "right_arm_eef", "left_leapv2", "left_arm_eef"]
            state_keys = ["right_arm_eef", "left_arm_eef", "intergripper"]
            img_keys = ["right_pinky_cam", "right_thumb_cam", "left_pinky_cam", "left_thumb_cam"]
        elif left_hand:
            action_keys = ["left_leapv2", "left_arm_eef"]
            state_keys = ["left_arm_eef"]
            img_keys = ["left_pinky_cam", "left_thumb_cam"]
        elif right_hand:
            action_keys = ["right_leapv2", "right_arm_eef"]
            state_keys = ["right_arm_eef"]
            img_keys = ["right_pinky_cam", "right_thumb_cam"]
    
    if language:
        lang_keys = ["language"]
    else:
        lang_keys = []
    
    main(data, retain_pct, upsample_pcts, upsample_multiplier, MODE, buffer_name, action_keys, state_keys, lang_keys, img_keys, rot_repr, subsample_rate, human)