#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction
import time


def generate_launch_description():

    config_file = os.path.join(
        os.path.expanduser('~/dexwild/dexwild_ros2'),
        'configs',
        'human_bimanual_jetson2.yaml'
    )
    
    assert os.path.exists(config_file), f"Config file not found at {config_file}"
    
    
    glove_node = Node(
        package='glove',          # Replace with your package name
        executable='read_and_send_zmq',         # The entry point defined in setup.py
        name='glove_node',          # Name of the node
        output='screen',                 # Output logs to screen
        parameters= [config_file]
    )
    
    palm_camera_node = Node(
        package='dexwild_cameras',          # Replace with your package name
        executable='read_palm_cameras',         # The entry point defined in setup.py
        name='palm_camera_node',          # Name of the node
        output='screen',                 # Output logs to screen
        parameters=[config_file]
    )
    
    zed_node = Node(
        package = 'dexwild_cameras',          # Replace with your package name
        executable = 'read_zed',
        name = 'zed_node',
        output = 'screen',
        parameters = [config_file]
    )
    
    delayed_palm = TimerAction(
        period=10.0,             # Wait 5 seconds
        actions=[palm_camera_node]
    )

    return LaunchDescription([
        glove_node,  
        zed_node,   
        delayed_palm,
    ])



