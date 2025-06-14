#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch_ros.actions import Node
import numpy as np

deg2rad = np.pi/180

left_hand_pos1 = [1.17208822e-01, 3.60038023e-01, 2.26476593e-01, 3.02883962e-01, 1.60047096e-01, 
                                1.56461728e-01, 4.33847452e-01, 2.93576544e-01, 1.65315670e-01, 4.63769688e-01, 
                                1.63201087e-01, 2.25485884e-01, 1.07365078e+00, 5.19101894e-01, 3.82364111e-02, 
                                4.81756478e-01, 1.14412508e-01] # start pose 1

right_hand_pos1 = [1.17208822e-01, 3.60038023e-01, 2.26476593e-01, 3.02883962e-01, 1.60047096e-01, 
                                1.56461728e-01, 4.33847452e-01, 2.93576544e-01, 1.65315670e-01, 4.63769688e-01, 
                                1.63201087e-01, 2.25485884e-01, 1.07365078e+00, 5.19101894e-01, 3.82364111e-02, 
                                4.81756478e-01, 1.14412508e-01] # start pose 1 ##open


right_arm_pos = [0.285, 0.0, 0.392, -168*deg2rad, 0.5*deg2rad, -92*deg2rad]
left_arm_pos = [ 0.30018143, -0.10876129,  0.35720853,  2.98594601, -0.20576644, -1.80375634]

def generate_launch_description():
    
    config_file = os.path.join(
        os.path.expanduser('~/dexwild/dexwild_ros2'),
        'configs',
        'bimanual_robot_wideview_deploy.yaml'
    )
    
    assert os.path.exists(config_file), f"Config file not found at {config_file}"
    

    left_xarm_node = Node(
        package='xarm_control',
        executable='xarm_node',
        name='left_xarm_node',
        output='screen',
        parameters=[config_file],
    )

    right_xarm_node = Node(
        package='xarm_control',        
        executable='xarm_node',        
        name='right_xarm_node',          
        output='screen',               
        parameters=[config_file],
    )
    
    leap_v2_node_left = Node(
        package='leap_v2',
        executable='leapv2_node',
        name='leap_v2_node_left',
        output='screen',
        parameters=[config_file]
    )
    
    leap_v2_node_right = Node(
        package='leap_v2',
        executable='leapv2_node',
        name='leap_v2_node_right',
        output='screen',
        parameters=[config_file]
    )
    
    palm_camera_node = Node(
        package='dexwild_cameras',          
        executable='read_palm_cameras',         
        name='palm_camera_node',          
        output='screen',                
        parameters=[config_file]
    )
    
    data_sync_node = Node(
        package='dexwild_data_collection',     
        executable='sync_data',        
        name='data_sync_node',        
        output='screen',              
        parameters=[config_file]
    )

    ticker_node = Node(
        package='dexwild_data_collection',        
        executable='ticker',        
        name='ticker_node',      
        output='screen',                
        parameters=[config_file]
    )

    policy_node = Node(
        package='dexwild_deploy',          # Replace with your package name
        executable='deploy_policy',         # The entry point defined in setup.py
        name='deploy_policy_node',          # Name of the node
        output='screen',                 # Output logs to screen
        parameters=[
            
            {
            "checkpoint_path": "Path to your model checkpoint",  # Replace with your model path
            
            "replay_path": "Path to your replay data",  # Replace with your replay path
            "replay_id": 46,
            
            "id": "bimanual",
            
            "observation_keys": ['right_pose', 'left_pose', 'right_hand', 'left_hand', 'right_thumb_cam', 'right_pinky_cam', 'left_thumb_cam', 'left_pinky_cam'],
            
            "isensemble": False,
            "openloop_length": 48,
            "skip_first_actions":8,
            
            "freq": 20,
            "rot_repr": "rot6d", # "euler", "quat", "rot6d"
            
            # HISTORY 
            "buffer_size": 19,
            
            # SMOOTHING
            "ema_amount": 0.5,
            "use_rmp": True,
            
            "start_poses": 
            np.array([right_arm_pos, left_arm_pos]).flatten().tolist(),

            "start_hand_poses":
            np.array([right_hand_pos1, left_hand_pos1]).flatten().tolist(),
            
            "pred_horizon": 16,
            "exp_weight": 0.4, # increase to give more weight to older predictions
            "mode": 
                # "rel_mixed"
                # "abs"
                "hybrid",
                # "rel_intergripper"
                
                # "replay_rel"
                # "replay_abs"
            }]
    )
    
    return LaunchDescription([
        left_xarm_node,
        right_xarm_node,
        
        # zed_node,
        ticker_node,

        leap_v2_node_left,
        leap_v2_node_right,
        policy_node,
        
        palm_camera_node,
        data_sync_node,
    ]
    )

