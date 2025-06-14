#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction

mirrored = False

def generate_launch_description():
    
    config_file = os.path.join(
        os.path.expanduser('~/dexwild/dexwild_ros2'),
        'configs',
        'bimanual_robot_wideview.yaml'
    )
    
    assert os.path.exists(config_file), f"Config file not found at {config_file}"
    
    left_mobile_gello_node = Node(
        package='dexwild_gello',
        executable='read_gello',
        name='left_mobile_gello_node',
        output='screen',
        parameters=[config_file],
    )
    
    right_mobile_gello_node = Node(
        package='dexwild_gello',          
        executable='read_gello',        
        name='right_mobile_gello_node',        
        output='screen',              
        parameters=[config_file],
    )
    
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
    
    glove_node = Node(
        package='glove',         
        executable='read_and_send_zmq',         
        name='glove_node',          
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
    
    leap_v2_ik_left = Node(
        package='retargeting',
        executable='leap_v2_ik',
        name='leap_v2_ik_left',
        output='screen',
        emulate_tty=True,
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
    
    delayed_arm_nodes = TimerAction(
        period=5.0,             # Wait 5 seconds
        actions=[left_xarm_node,
        left_mobile_gello_node,
        leap_v2_ik_left,
        leap_v2_node_left,
        right_xarm_node,
        right_mobile_gello_node,
        leap_v2_ik_right,
        leap_v2_node_right]
    )

    delayed_palm_camera_node = TimerAction(
        period=5.0,             # Wait 5 seconds
        actions=[palm_camera_node]
    )
    
    delayed_datasync = TimerAction(
        period=7.5,             # Wait 5 seconds
        actions=[data_sync_node]
    )

    return LaunchDescription([
        glove_node,
        ticker_node,
        
        delayed_arm_nodes,
        delayed_palm_camera_node,
        delayed_datasync,
    ])



