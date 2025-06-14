#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch_ros.actions import Node

mirrored = False

def generate_launch_description():
    
    config_file = os.path.join(
        os.path.expanduser('~/dexwild/dexwild_ros2'),
        'configs',
        'right_robot_leapv2.yaml'
    )
    
    assert os.path.exists(config_file), f"Config file not found at {config_file}"
    
    # Create the ReadGello node
    right_mobile_gello_node = Node(
        package='dexwild_gello',          
        executable='read_gello',        
        name='right_mobile_gello_node',        
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
    
    glove_node = Node(
        package='glove',         
        executable='read_and_send_zmq',         
        name='glove_node',          
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
    
    leap_v2_ik_right = Node(
            package='retargeting',
            executable='leap_v2_ik',
            name='leap_v2_ik_right',
            output='screen',
            emulate_tty=True,
            parameters=[config_file]
        )

    palm_camera_node = Node(
        package='dexwild_cameras',          
        executable='read_palm_cameras',         
        name='palm_camera_node',          
        output='screen',                
        parameters=[config_file]
    )
    
    zed_node = Node(
        package = 'dexwild_cameras',
        executable = 'read_zed',
        name = 'zed_node',
        output = 'screen',
        parameters = [config_file]
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

    return LaunchDescription([
        right_xarm_node,
        right_mobile_gello_node,
        glove_node,
        leap_v2_ik_right,
        leap_v2_node_right,
        ticker_node,
        palm_camera_node,
        data_sync_node,
    ])



