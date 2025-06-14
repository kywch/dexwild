#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    config_file = os.path.join(
        os.path.expanduser('~/dexwild/dexwild_ros2'),
        'configs',
        'human_bimanual_jetson2.yaml'
    )
    
    assert os.path.exists(config_file), f"Config file not found at {config_file}"
    
    data_sync_node = Node(
        package='dexwild_data_collection',          # Replace with your package name
        executable='sync_data',         # The entry point defined in setup.py
        name='data_sync_node',          # Name of the node
        output='screen',                 # Output logs to screen
        parameters=[config_file]
    )

    return LaunchDescription([    
        data_sync_node,
    ])



