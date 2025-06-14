#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch_ros.actions import Node
import numpy as np

deg2rad = np.pi/180

hand_pose1 = [1.17208822e-01, 3.60038023e-01, 2.26476593e-01, 3.02883962e-01, 1.60047096e-01, 
                1.56461728e-01, 4.33847452e-01, 2.93576544e-01, 1.65315670e-01, 4.63769688e-01, 
                1.63201087e-01, 2.25485884e-01, 1.07365078e+00, 5.19101894e-01, 3.82364111e-02, 
                4.81756478e-01, 1.14412508e-01]

toy_lab_pose = [0.300, -0.300, 0.350, -158*deg2rad, 1.1*deg2rad, -84*deg2rad]

def generate_launch_description():
    
    config_file = os.path.join(
    os.path.expanduser('~/dexwild/dexwild_ros2'),
    'configs',
    'right_robot_wideview_deploy.yaml'
    )
    
    assert os.path.exists(config_file), f"Config file not found at {config_file}"
    

    right_xarm_node = Node(
        package='xarm_control',        
        executable='xarm_node',        
        name='right_xarm_node',          
        output='screen',               
        parameters=[config_file],
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
        package='dexwild_deploy',         
        executable='deploy_policy',       
        name='deploy_policy_node',       
        output='screen',               
        parameters=[{"checkpoint_path":
            "/home/alfredo/telekinesis_3/train/remote_models/final_models/toy_cotrain_2to1_48ac_0_9_18_eef/toy_cotrain_2to1_48ac_0_9_18_eef_500000.pth",
                
            "replay_path": "/home/alfredo/telekinesis_3/all_data/test_buffers/florist_bimanual_human_abs_not_normalized_all_apr19/small_buffer.pkl",
            "replay_id": 46,
            
            "id": "right_mobile",
            
            "observation_keys": ['right_pose', 'right_hand', 'right_thumb_cam', 'right_pinky_cam'],
            
            "isensemble": False
            ,
            "openloop_length": 48,
            "skip_first_actions":0,
            
            "freq": 30,
            "rot_repr": "rot6d", # "euler", "quat", "rot6d"
            
            "buffer_size": 19,
        
            "mask": False,
            
            "ema_amount": 0.5,
            "use_rmp": False,
            
            "start_poses": 
            np.array(toy_lab_pose).flatten().tolist(),
            
            "start_hand_poses":
            np.array([hand_pose1]).flatten().tolist(),
            
            "pred_horizon": 16,
            "exp_weight": 0.4, # increase to give more weight to older predictions
            "mode": 
                # "rel_mixed"
                # "abs"
                "hybrid",
                # "rel_intergripper"
                
                # "replay_rel"
                # "replay_rel_abs"
                # "replay_abs"
                # "replay_rel_intergripper"
        }]
    )
    
    return LaunchDescription([
        right_xarm_node,
        ticker_node,
        leap_v2_node_right,
        
        palm_camera_node,
        data_sync_node,
        
        policy_node,
    ]
    )

