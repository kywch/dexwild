# Copyright (c) 2025 Tony Tao
# Licensed under the MIT License. See LICENSE file in the project root for details.

"""
Replay and Diffusion Policy Inference for Dexterous Manipulation
=================================================================

This script defines two main classes for running action inference on demonstration data:
1. `ReplayPolicy` – A simple policy that replays pre-recorded demonstration actions.
2. `DiTInference` – A learned policy interface using a Diffusion Transformer (DiT) model 
   for action generation based on multi-view images, state history, and optional language inputs.

Key Features:
-------------
- Handles both single-arm and bimanual setups, with support for Leap hand inputs.
- Decodes and visualizes camera frames for inspection.
- Reconstructs absolute pose sequences from relative transformations (e.g., `replay_rel` mode).
- Integrates action normalization and thresholding for hardware safety and consistency.
- Uses weighted ensembles of predictions over a temporal horizon to stabilize inference.
- Converts between different pose representations: Euler, quaternion, and 6D.
- Supports hybrid control modes combining relative and absolute actions.
"""

import os
import cv2
import json
import yaml
import torch
import pickle
import hydra
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from collections import deque
from scipy.spatial.transform import Rotation as R

from dexwild_utils.pose_utils import (
    plot_one_path3d, plot_paths3d, average_quaternions,
    pose6d_to_mat, mat_to_pose6d,
    mat_to_pose_func, pose_to_mat_func,
    pose_to_pose6d, pose6d_to_pose
)

class ReplayPolicy:
    def __init__(self, buffer_path, id, mode, normalized = False, isensemble = False, replay_id = 0, rot_repr = "euler"):
        with open(buffer_path, "rb") as f:
            buffer = pickle.load(f)
        
        self.id = id
        
        self.mode = mode
        self.normalized = normalized
        self.isensemble = isensemble
        
        self.is_bimanual = "bimanual" in self.id
        
        self.rot_repr = rot_repr
        
        action_keys = []
        if "leapv1" in self.id:
            raise NotImplementedError
        else:
            if "bimanual" in self.id:
                action_keys.append("right_leapv2")
                action_keys.append("right_arm_eef")
                action_keys.append("left_leapv2")
                action_keys.append("left_arm_eef")
            elif "right" in self.id:
                action_keys.append("right_leapv2")
                action_keys.append("right_arm_eef")
            elif "left" in self.id:
                action_keys.append("left_leapv2")
                action_keys.append("left_arm_eef")
        
        if self.normalized == True:
            with open(os.path.join(os.path.dirname(buffer_path), "ac_norm.json"), "r") as f:
                ac_norm_dict = json.load(f)
                loc, scale = ac_norm_dict["loc"], ac_norm_dict["scale"]
                self.loc = np.array(loc).astype(np.float32)
                self.scale = np.array(scale).astype(np.float32)
        
        trajectory_idx = replay_id
        
        self.trajectory = buffer[trajectory_idx]
        
        print("Length of Replay Buffer:", len(self.trajectory), "| Time In Seconds:", len(self.trajectory)/30)
        
        first_timestep = self.trajectory[0][0] #timestep # state
        
        first_image = first_timestep["enc_cam_0"]
        second_image = first_timestep["enc_cam_1"]  # encoded in jpeg
        first_image = cv2.imdecode(first_image, cv2.IMREAD_COLOR)
        second_image = cv2.imdecode(second_image, cv2.IMREAD_COLOR)
        first_image_rgb = cv2.cvtColor(first_image, cv2.COLOR_BGR2RGB)
        second_image_rgb = cv2.cvtColor(second_image, cv2.COLOR_BGR2RGB)
        
        if self.is_bimanual:
            third_image = first_timestep["enc_cam_2"]
            fourth_image = first_timestep["enc_cam_3"]
            third_image =  cv2.imdecode(third_image, cv2.IMREAD_COLOR)
            fourth_image = cv2.imdecode(fourth_image, cv2.IMREAD_COLOR)
            third_image_rgb = cv2.cvtColor(third_image, cv2.COLOR_BGR2RGB)
            fourth_image_rgb = cv2.cvtColor(fourth_image, cv2.COLOR_BGR2RGB) 
            combined_image = np.concatenate((first_image_rgb, second_image_rgb, third_image_rgb, fourth_image_rgb), axis=1)
        else:
            combined_image = np.concatenate((first_image_rgb, second_image_rgb), axis=1)
        
        plt.imshow(combined_image)
        
        self.ncams = None
        
        self.delay = 0
        
        self.actions = {}
        for key in action_keys:
            self.actions[key] = []
            
        # self.actions is an dictionary of arrays
        for t in range(len(self.trajectory)):
            for key in action_keys:
                self.actions[key].append(self.trajectory[t][1][key])
                
        for key in action_keys:
            self.actions[key] = np.array(self.actions[key])
            print("action key:", key, "shape:", self.actions[key].shape)
        
        self.actions = self.threshold_hands(self.actions)
        
        # show the paths
        if "bimanual" in self.id:
            plot_paths3d([self.actions["right_arm_eef"], self.actions["left_arm_eef"]])
        elif "right" in self.id:
            plot_one_path3d(self.actions["right_arm_eef"])
        elif "left" in self.id:
            plot_one_path3d(self.actions["left_arm_eef"])

        if self.mode == 'replay_rel':     
            for key in action_keys:               
                if "eef" in key:
                    prev_pose = self.actions[key][0].copy()
                    for t in range(1, self.actions[key].shape[0]):
                        trans = self.actions[key][t]
                        prev_pose_mat = pose_to_mat_func(self.rot_repr)(prev_pose)
                        trans_mat = pose_to_mat_func(self.rot_repr)(trans)
                        new_pose_mat = prev_pose_mat @ trans_mat
                        new_pose = mat_to_pose_func(self.rot_repr)(new_pose_mat)
                        
                        prev_pose = new_pose.copy()
                        
                        self.actions[key][t] = new_pose
                    
            if "bimanual" in self.id:
                plot_paths3d([self.actions["right_arm_eef"], self.actions["left_arm_eef"]]) 
            else:
                plot_one_path3d(self.actions["right_arm_eef"])
            
        self.curr_idx = 0
        self.chunk_size = 30

        self.pred_horizon = 24 #30 #24 #16
        self.exp_weight = 0.4 #0.8 #0.4
        self.act_history = deque(maxlen=self.pred_horizon)
        
        # convert actions to pose6d
        vectorized_actions = []
        for key in action_keys:
            action = self.actions[key]
            if "eef" in key:
                new_actions = []
                for i  in range(len(action)):
                    new_actions.append(pose_to_pose6d(self.rot_repr)(action[i]))
            else:
                new_actions = action
            vectorized_actions.append(new_actions)
        
        vectorized_actions = np.concatenate(vectorized_actions, axis=1)
        self.actions = vectorized_actions
    
    def update_pose(self, curr_pose):
        if "right_arm_eef" in curr_pose.keys() and "left_arm_eef" in curr_pose.keys():
            self.curr_pose = [curr_pose["right_arm_eef"], curr_pose["left_arm_eef"]]
            
    def forward(self, obs, lang=None):
        
        if self.curr_idx < len(self.actions) and not self.isensemble:
            ac = self.actions[self.curr_idx]
            self.curr_idx += 1
            return ac
        
        elif (self.curr_idx + self.chunk_size) < len(self.actions) and self.isensemble:
            actions = self.actions[self.curr_idx:self.curr_idx+self.chunk_size]
            self.curr_idx += 1
            self.act_history.append(actions)
            
            num_actions = len(self.act_history)
            curr_act_preds = np.stack(
                [
                    pred_actions[i]
                    for (i, pred_actions) in zip(
                        range(num_actions - 1, -1, -1), self.act_history
                    )
                ]
            )
            # more recent predictions get exponentially *less* weight than older predictions
            weights = np.exp(-self.exp_weight * np.arange(num_actions))
            weights = weights / weights.sum()
            # compute the weighted average across all predictions for this timestep
            ac = np.sum(weights[:, None] * curr_act_preds, axis=0)
            return ac
        else:
            return None
    
    
    def threshold_hands(self, actions):
        
        ###### HAND THRESHOLDING #########
        # thresholding stuff
        self.hand_threshold = None
        
        self.hand_scaler = 1.6
        
        ###### HAND THRESHOLDING #########
        
        motors_side =    [0,3,6,9,12]
        motors_forward = [1,4,7,10,13]
        motors_curl =    [2,5,8,11,14]
        motors_palm =    [15,16]  # 15 is for the thumb, 16 is between the 4 fingers,
        
        motors_close = motors_forward + motors_curl + motors_palm
        
        actions_copy = actions.copy()
        
        if self.hand_threshold is not None:
            if "bimanual" in self.id:
                right_hand_actions = actions_copy["right_leapv2"]
                left_hand_actions = actions_copy["left_leapv2"]
                
                left_hand_actions[:, motors_close] = np.where(left_hand_actions[:, motors_close] > self.hand_threshold, left_hand_actions[:, motors_close] * self.hand_scaler, left_hand_actions[:, motors_close])
                right_hand_actions[:, motors_close] = np.where(right_hand_actions[:, motors_close] > self.hand_threshold, right_hand_actions[:, motors_close] * self.hand_scaler, right_hand_actions[:, motors_close])
                
                actions_copy["right_leapv2"] = right_hand_actions
                actions_copy["left_leapv2"] = left_hand_actions
            elif "right" in self.id:
                right_hand_actions = actions_copy["right_leapv2"]
                right_hand_actions[:, motors_close] = np.where(right_hand_actions[:, motors_close] > self.hand_threshold, right_hand_actions[:, motors_close] * self.hand_scaler, right_hand_actions[:, motors_close])
                actions_copy["right_leapv2"] = right_hand_actions
            elif "left" in self.id:
                left_hand_actions = actions_copy["left_leapv2"]
                left_hand_actions[:, motors_close] = np.where(left_hand_actions[:, motors_close] > self.hand_threshold, left_hand_actions[:, motors_close] * self.hand_scaler, left_hand_actions[:, motors_close])
                actions_copy["left_leapv2"] = left_hand_actions
        
        # additional retargeting
        if "bimanual" in self.id:
            actions_copy["right_leapv2"][:, 12] = 0.25
            actions_copy["left_leapv2"][:, 12] = 0.25
        elif "right" in self.id:
            actions_copy["right_leapv2"][:, 12] = 0.25
        elif "left" in self.id:
            actions_copy["left_leapv2"][:, 12] = 0.25
            
        return actions_copy
    

class DiTInference:
    def __init__(self, agent_path, model_name, isensemble = False, id = "none", mode = "", openloop_length = 30, pred_horizon = 24, exp_weight = 0.4, rot_repr = "euler", buffer_size = 1, skip_first_actions=0):
        with open(Path(agent_path, "agent_config.yaml"), "r") as f:
            config_yaml = f.read()
            agent_config = yaml.safe_load(config_yaml)
        with open(os.path.expanduser("~/dexwild/train/obs_config.yaml", "r")) as f:
            config_yaml = f.read()
            obs_config = yaml.safe_load(config_yaml)
        with open(Path(agent_path, "exp_config.yaml"), "r") as f:
            config_yaml = f.read()
            exp_config = yaml.safe_load(config_yaml)
        
        self.rot_repr = rot_repr
        self.ncams = agent_config["n_cams"]
        self.obs_dim = agent_config["odim"]
        try:
            self.obs_keys = exp_config["params"]["task"]["train_buffer"]["obs_keys"]
            self.act_keys = exp_config["params"]["task"]["train_buffer"]["act_keys"]
            hybrid_action = exp_config["params"]["task"]["train_buffer"]["hybrid_action"]
            self.img_hist_frames = exp_config["params"]["task"]["train_buffer"]["img_hist_frame_indices"]
            self.state_hist_frames = exp_config["params"]["task"]["train_buffer"]["state_hist_frame_indices"]
        except:
            self.obs_keys = exp_config["params"]["task"]["buffer_defaults"]["obs_keys"]
            self.act_keys = exp_config["params"]["task"]["buffer_defaults"]["act_keys"]
            hybrid_action = exp_config["params"]["task"]["buffer_defaults"]["hybrid_action"]
            self.img_hist_frames = exp_config["params"]["task"]["buffer_defaults"]["img_hist_frame_indices"]
            self.state_hist_frames = exp_config["params"]["task"]["buffer_defaults"]["state_hist_frame_indices"]
            
        self.buffer_size = buffer_size
        self.skip_first_actions = skip_first_actions
        
        if hybrid_action:
            assert mode == "hybrid"
        
        self.ac_norm_stats = {}
        self.state_norm_stats = {}
        
        # find the stats
        files = os.listdir(agent_path)
        for file in files:
            if "state_norm" in file:
                state_norm_file = file
            elif "ac_norm" in file:
                ac_norm_file = file
        
        with open(Path(agent_path, ac_norm_file), "r") as f:
            self.ac_norm_stats = json.load(f)
    
        if agent_config["use_obs"] != False: # using state information
            with open(Path(agent_path, state_norm_file), "r") as f:
                self.state_norm_stats = json.load(f)

        self.mode = mode
        self.id = id
        self.isensemble = isensemble
        
        if self.rot_repr == "euler":
            if "leapv1" in self.id:
                self.right_hand_idx = slice(0, 16)
                self.right_arm_idx = slice(16, 22)
            else:
                self.right_hand_idx = slice(0, 17)
                self.right_arm_idx = slice(17, 23)
                self.left_hand_idx = slice(23, 40)
                self.left_arm_idx = slice(40, 46)
        elif self.rot_repr == "rot6d":
            if "leapv1" in self.id:
                raise NotImplementedError("rot6d not supported for leapv1")
            else:
                self.right_hand_idx = slice(0, 17)
                self.right_arm_idx = slice(17, 26)
                self.left_hand_idx = slice(26, 43)
                self.left_arm_idx = slice(43, 52)
        else:
            raise NotImplementedError("Only euler is supported right now")
        
        self.step = 0
        
        agent = hydra.utils.instantiate(agent_config)
        save_dict = torch.load(Path(agent_path, model_name), map_location="cpu")
        
        agent.load_state_dict(save_dict["model"])
        self.agent = agent.eval().cuda()
        
        self.raw_pred_length = agent_config["ac_chunk"]
        self.use_obs = agent_config["use_obs"]
        self.openloop_length = openloop_length

        self.transform = hydra.utils.instantiate(obs_config["transform"])

        print(f"loaded agent from {agent_path}, at step: {save_dict['global_step']}")
        
        self._last_time = None
        
        self.pred_horizon = pred_horizon 
        self.exp_weight = exp_weight 
        self.sparse_ensemble_steps = self.raw_pred_length // self.pred_horizon 
        
        assert self.sparse_ensemble_steps * self.pred_horizon <= self.raw_pred_length
        
        self.act_history = deque(maxlen=self.pred_horizon)
        
        self.act_history_logger = []
        
        self.actions = []
        
        self.arm_log = []
        
        self.obs_diff = []
        
        self.applied_actions = None
        
        self.started = False
        
    def preprocess_image(self, bgr_img, size=(256, 256)):
        bgr_img = cv2.resize(bgr_img, size, interpolation=cv2.INTER_AREA)
        rgb_img = bgr_img[:, :, ::-1].copy()
        rgb_img = torch.from_numpy(rgb_img).float().permute((2, 0, 1)) / 255
        return self.transform(rgb_img)

    def _proc_image(self, imgs, size=(256, 256)):
        
        cam_dict = {"cam0": [], "cam1": [], "cam2": [], "cam3": [], "cam4": []}
        buffer_size = self.buffer_size     
        if "bimanual" in self.id:
            for frame_id in self.img_hist_frames:
                if self.ncams == 4:
                    cam_dict["cam0"].insert(0, self.preprocess_image(imgs["cam0"][buffer_size - frame_id - 1], size))
                    cam_dict["cam1"].insert(0, self.preprocess_image(imgs["cam1"][buffer_size - frame_id - 1], size))
                    cam_dict["cam2"].insert(0, self.preprocess_image(imgs["cam2"][buffer_size - frame_id - 1], size))
                    cam_dict["cam3"].insert(0, self.preprocess_image(imgs["cam3"][buffer_size - frame_id - 1], size))
                elif self.ncams == 1:
                    cam_dict["cam0"].insert(0, self.preprocess_image(imgs["cam4"][buffer_size - frame_id - 1], size)) # just take the third person
                elif self.ncams == 3:
                    cam_dict["cam0"].insert(0, self.preprocess_image(imgs["cam1"][buffer_size - frame_id - 1], size))
                    cam_dict["cam1"].insert(0, self.preprocess_image(imgs["cam3"][buffer_size - frame_id - 1], size))
                    cam_dict["cam2"].insert(0, self.preprocess_image(imgs["cam4"][buffer_size - frame_id - 1], size))
                else:
                    raise NotImplementedError
        else:
            for frame_id in self.img_hist_frames:
                if self.ncams == 1:
                    cam_dict[f"cam0"].insert(0, self.preprocess_image(imgs["cam0"][buffer_size - frame_id - 1], size))
                elif self.ncams == 2:
                    cam_dict[f"cam0"].insert(0, self.preprocess_image(imgs["cam0"][buffer_size - frame_id - 1], size))
                    cam_dict[f"cam1"].insert(0, self.preprocess_image(imgs["cam1"][buffer_size - frame_id - 1], size))
                elif self.ncams == 3:
                    cam_dict[f"cam0"].insert(0, self.preprocess_image(imgs["cam0"][buffer_size - frame_id - 1], size))
                    cam_dict[f"cam1"].insert(0, self.preprocess_image(imgs["cam1"][buffer_size - frame_id - 1], size))
                    cam_dict[f"cam2"].insert(0, self.preprocess_image(imgs["cam2"][buffer_size - frame_id - 1], size))
        for key in cam_dict.keys():
            if cam_dict[key] != []:
                cam_dict[key] = torch.stack(cam_dict[key], dim=0)[None].cuda() # add batch dim
        
        return cam_dict

    def _proc_state(self, obs):
        state = obs.astype(np.float32)
            
        state = torch.from_numpy(state).cuda()
        return state
    
    def update_pose(self, curr_pose):
        if "bimanual" in self.id and "right_arm_eef" in curr_pose.keys() and "left_arm_eef" in curr_pose.keys():
            self.curr_pose = [curr_pose["right_arm_eef"], curr_pose["left_arm_eef"]]
        elif "right" in self.id and "right_arm_eef" in curr_pose.keys():
            self.curr_pose = [curr_pose["right_arm_eef"]]
        elif "left" in self.id and "left_arm_eef" in curr_pose.keys():
            self.curr_pose = [curr_pose["left_arm_eef"]]
        
    def forward(self, obs, lang=None):
        img = self._proc_image(obs["images"])
        # hand_state = self._proc_state(obs["leapv2"])
        pose_state = self._proc_state(obs["pose"])

        state = []
        if self.use_obs != False:  
            for key in self.obs_keys:
                if key == "intergripper":
                    # normalize the intergripper state using the saved loc and scale
                    tmp_obs = []
                    for i in range(len(obs["intergripper"])):
                        tmp_obs.append((pose6d_to_pose(self.rot_repr)(obs["intergripper"][i]) - self.state_norm_stats[key]["loc"]) / self.state_norm_stats[key]["scale"])
                    
                    curr_state = self._proc_state(np.array(tmp_obs)[None]) # add batch dimension
                elif "eef" in key:
                    arm_obs = []

                    if "bimanual" in self.id:
                        if key == "right_arm_eef":
                            arm_poses = obs["pose"][0, :, :]
                        elif key == "left_arm_eef":
                            arm_poses = obs["pose"][1, :, :]
                    else:
                        arm_poses = obs["pose"]
                    
                    inv_curr_arm_pose_mat = np.linalg.inv(pose6d_to_mat(arm_poses[-1]))
                    
                    for i in range(arm_poses.shape[0]):
                        pose = pose6d_to_mat(arm_poses[i])
                        if "hybrid" in self.mode:
                            unnormed_obs = mat_to_pose_func(self.rot_repr)(inv_curr_arm_pose_mat @ pose)
                        else:
                            unnormed_obs = mat_to_pose_func(self.rot_repr)(pose)
                        normed_obs = (unnormed_obs - self.state_norm_stats[key]["loc"]) / self.state_norm_stats[key]["scale"]
                        arm_obs.append(normed_obs)
                    arm_obs = np.array(arm_obs)
                    
                    curr_state = self._proc_state(arm_obs[None])
                    
                state.append(curr_state)
            
            # cat all of them together
            state = torch.cat(state, dim=2)  # Concatenate along the feature dimension
            
            tmp_state = []
            buffer_size = state.shape[1]
            for frame_idx in self.state_hist_frames:
                tmp_state.append(state[:, buffer_size - frame_idx - 1, :])
            
            state = torch.cat(tmp_state)[None]
        else:   
            state = torch.empty(0)[None].cuda()
        
        # run inference either when actions are depleted or when ensemble is enabled
        if len(self.actions) <= (self.raw_pred_length - self.openloop_length) or (self.isensemble and self.step % self.sparse_ensemble_steps == 0):
            with torch.no_grad():

                ac = self.agent.get_actions(img, state, lang)
            
            self.actions = ac[0].cpu().numpy().astype(np.float32)
            
            action_dict = {}
            for key in self.act_keys:
                if key == "right_arm_eef":
                    action_dict[key] = self.actions[:, self.right_arm_idx]
                elif key == "left_arm_eef":
                    action_dict[key] = self.actions[:, self.left_arm_idx]
                elif key == "right_leapv2":
                    action_dict[key] = self.actions[:, self.right_hand_idx]
                elif key == "left_leapv2":
                    action_dict[key]= self.actions[:, self.left_hand_idx]
                    
                action_dict[key] = (action_dict[key] * self.ac_norm_stats[key]["scale"]) + self.ac_norm_stats[key]["loc"] # unnormalize the actions

            action_dict = self.threshold_hands(action_dict)

            current_pose = self.curr_pose
            
            if  self.mode == 'abs':
                self.applied_actions = action_dict
            elif self.mode == "hybrid":
                self.applied_actions = {}
                for act_key in action_dict.keys():
                    temp = []
                    if "eef" in act_key:
                        for i in range(self.raw_pred_length):
                            if act_key == "right_arm_eef":
                                target_rel_pose = action_dict[act_key][i, :]
                                current_pose_mat = pose6d_to_mat(current_pose[0])
                            elif act_key == "left_arm_eef":
                                target_rel_pose = action_dict[act_key][i, :]
                                current_pose_mat = pose6d_to_mat(current_pose[1])

                            target_rel_pose_mat = pose_to_mat_func(self.rot_repr)(target_rel_pose)
                            target_pose_mat = current_pose_mat @ target_rel_pose_mat
                            target_pose = mat_to_pose6d(target_pose_mat)
                            
                            temp.append(target_pose)
                        
                        if self.skip_first_actions > 0:
                            copy_start = temp[self.skip_first_actions]
                            for i in range(self.skip_first_actions):
                                temp[i] = copy_start
                    else:
                        temp = action_dict[act_key]
                        
                    self.applied_actions[act_key] = np.array(temp)
                        
            elif self.mode == 'rel_mixed': # relative eef but absolute hands
                #convert relative transformations to absolute poses
                self.applied_actions = {}

                if "bimanual" in self.id:
                    for act_key in action_dict.keys():
                        temp = []
                        if "eef" in act_key:
                            for i in range(self.raw_pred_length):
                                    if act_key == "right_arm_eef":
                                        target_rel_pose = action_dict[act_key][i, :]
                                        current_pose_mat = pose6d_to_mat(current_pose[0])
                                    elif act_key == "left_arm_eef":
                                        target_rel_pose = action_dict[act_key][i, :]
                                        current_pose_mat = pose6d_to_mat(current_pose[1])

                                    target_rel_pose_mat = pose_to_mat_func(self.rot_repr)(target_rel_pose)
                                    target_pose_mat = current_pose_mat @ target_rel_pose_mat
                                    target_pose = mat_to_pose6d(target_pose_mat)
                                    
                                    temp.append(target_pose)
                                    
                                    if act_key == "right_arm_eef":
                                        current_pose[0] = target_pose
                                    elif act_key == "left_arm_eef":
                                        current_pose[1] = target_pose
                                        
                            # if skip actions > 0, take the action from skip_frame and copy it to 
                            if self.skip_first_actions > 0:
                                copy_start = temp[self.skip_first_actions]
                                for i in range(self.skip_first_actions):
                                    temp[i] = copy_start
                        else:
                            temp = action_dict[act_key]
                        
                        self.applied_actions[act_key] = np.array(temp)   

            self.actions = np.concatenate(list(self.applied_actions.values()), axis=1)
                    
            self.act_history_logger.append(self.actions)
        
        if not self.isensemble:
            ac = self.actions[0]
            self.actions = self.actions[1:]
        else:
            if "bimanual" in self.id:
                self.act_history.append(self.actions)
                num_actions = (len(self.act_history))
                curr_act_preds = []
                for j, pred_actions in zip(range(num_actions - 1, -1, -1), self.act_history):
                    try:
                        curr_act_preds.append(pred_actions[j])
                    except:
                        breakpoint()
                    
                curr_act_preds = np.array(curr_act_preds)
                weights = np.exp(-self.exp_weight * np.arange(num_actions))
                weights = weights[::-1]
                weights = weights / weights.sum()
                half_point = curr_act_preds.shape[1] // 2
                right_ac = self.average_actions(weights, curr_act_preds[:, :half_point], use_median=False)
                left_ac = self.average_actions(weights, curr_act_preds[:, half_point:], use_median=False)
                ac = np.concatenate((right_ac, left_ac))
        
        self.step += 1 
        
        return ac
    
    def average_actions(self, weights, curr_act_preds, use_median = False):
        """
        Averages the action predictions properly, handling rotations using quaternions.

        Parameters:
        - weights: A 1D numpy array of weights for each prediction.
        - curr_act_preds: A 2D numpy array of shape (num_predictions, action_dimension),
                        where each row is a predicted action.

        Returns:
        - ac: A 1D numpy array representing the averaged action.
        """

        # Indices for hand joints, rotations, and translations
        hand_joint_indices = self.right_hand_idx
        
        translation_indices = list(range(self.right_arm_idx.start or 0, self.right_arm_idx.stop or 0, self.right_arm_idx.step or 1))[:3]
        rotation_indices = list(range(self.right_arm_idx.start or 0, self.right_arm_idx.stop or 0, self.right_arm_idx.step or 1))[3:]
    
        # Extract components
        hand_joints = curr_act_preds[:, hand_joint_indices]      # Shape: (N, 17)
        translations = curr_act_preds[:, translation_indices]    # Shape: (N, 3)
        rotations = curr_act_preds[:, rotation_indices]          # Shape: (N, 3)
        
        if use_median == True:
            median_hand_joints = np.median(hand_joints, axis=0)
            median_translations = np.median(translations, axis=0)
            median_rotations = np.median(rotations, axis=0)
            ac = np.concatenate((median_hand_joints, median_translations, median_rotations))
            return ac

        # Average hand joints directly
        mean_hand_joints = np.sum(weights[:, None] * hand_joints, axis=0)  # Shape: (17,)

        # Convert rotations to quaternions
        quaternions = np.array([R.from_euler('xyz', rot).as_quat() for rot in rotations])  # Shape: (N, 4)

        # Average quaternions
        mean_quaternion = average_quaternions(quaternions, weights)  # Shape: (4,)

        # Convert mean quaternion back to Euler angles
        mean_rotation = R.from_quat(mean_quaternion).as_euler("xyz")  # Shape: (3,)

        # Average translations directly
        mean_translation = np.sum(weights[:, None] * translations, axis=0)  # Shape: (3,)

        # Combine components
        ac = np.concatenate((mean_hand_joints, mean_translation, mean_rotation))  # Shape: (23,)

        return ac
    
    def threshold_hands(self, actions):
        
        ###### HAND THRESHOLDING #########
        self.hand_threshold = None
    
        self.hand_scaler = 1.2
        
        ###### HAND THRESHOLDING #########
        
        motors_side =    [0,3,6,9,12]
        motors_forward = [1,4,7,10,13]
        motors_curl =    [2,5,8,11,14]
        motors_palm =    [15,16]  # 15 is for the thumb, 16 is between the 4 fingers,
        
        motors_close = motors_forward + motors_curl + motors_palm
        
        actions_copy = actions.copy()
        
        if self.hand_threshold is not None:
            if "bimanual" in self.id:
                right_hand_actions = actions_copy["right_leapv2"]
                left_hand_actions = actions_copy["left_leapv2"]
  
                left_hand_actions[:, motors_close] = np.where(left_hand_actions[:, motors_close] > self.hand_threshold, left_hand_actions[:, motors_close] * self.hand_scaler, left_hand_actions[:, motors_close])
                right_hand_actions[:, motors_close] = np.where(right_hand_actions[:, motors_close] > self.hand_threshold, right_hand_actions[:, motors_close] * self.hand_scaler, right_hand_actions[:, motors_close])
                
                actions_copy["right_leapv2"] = right_hand_actions
                actions_copy["left_leapv2"] = left_hand_actions

            elif "right" in self.id:
                right_hand_actions = actions_copy["right_leapv2"]
                right_hand_actions[:, motors_close] = np.where(right_hand_actions[:, motors_close] > self.hand_threshold, right_hand_actions[:, motors_close] * self.hand_scaler, right_hand_actions[:, motors_close])
                actions_copy["right_leapv2"] = right_hand_actions
                
        # additional retargeting
        if "bimanual" in self.id:
            actions_copy["right_leapv2"][:, 12] = 0.25
            actions_copy["left_leapv2"][:, 12] = 0.25     
        elif "right" in self.id:
            actions_copy["right_leapv2"][:, 12] = 0.25
        elif "left" in self.id:
            actions_copy["left_leapv2"][:, 12] = 0.25
            
        return actions_copy