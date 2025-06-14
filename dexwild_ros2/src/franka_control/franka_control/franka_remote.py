# Copyright (c) 2025 Tony Tao
# Licensed under the MIT License. See LICENSE file in project root for details.

"""
ROS 2 node for remote control and observation of a Franka robot arm via ZeroMQ.

This script enables communication between a remote policy or simulator and a physical or virtual Franka robot arm.
It subscribes to desired end-effector poses (as `PoseStamped` messages) over a ROS topic and sends them to a remote
server using ZeroMQ. It also receives the arm’s actual end-effector pose from a remote source via ZeroMQ and publishes
it back to ROS for visualization or logging.

Key features:
- Receives target end-effector poses via `/arm/<id>/cmd_eef_pose` (ROS topic) and sends them to a remote system.
- Receives the actual end-effector pose from the remote system over ZeroMQ and publishes it to `/arm/<id>/obs_eef_pose`.
- Listens to `/policy_start` to control when remote actions should be sent.
- Provides a test function (`test_function`) to simulate sending random poses.

Parameters:
- `server_IP`: IP address of the remote server.
- `id`: Identifier for the robot arm (e.g., 'franka_right').

ZMQ configuration:
- Subscribes to incoming poses from `tcp://172.16.0.9:2097`
- Publishes outgoing poses to `tcp://172.16.0.11:3097`
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
import asyncio
import random
from dexwild_utils.zmq_utils import ZMQPublisher, ZMQSubscriber


class FrankaRemote(Node):
    def __init__(self):
        super().__init__('franka_remote_node')

        # Parameters
        self.declare_parameter('server_IP', '172.26.9.131')
        self.declare_parameter('id', 'franka_right')
        
        server_IP = self.get_parameter('server_IP').value
        id = self.get_parameter('id').value
        
        self.get_logger().info(f"server_IP: {server_IP}")
        self.get_logger().info(f"ID: {id}")
        
        self.policy_start = False
        
        self.eef_tracker = self.create_subscription(
            PoseStamped,
            f"/arm/{id}/cmd_eef_pose",
            self._receive_eef_pose,
            10
        )
        
        self.arm_eef_pose_pub = self.create_publisher(
            PoseStamped,
            f"/arm/{id}/obs_eef_pose",
            10
        )
        
        otherIP = "172.16.0.9"
        myIP = "172.16.0.11"
        
        self.zmq_subscriber = ZMQSubscriber(f"tcp://{otherIP}:2097")
        
        self.zmq_publisher = ZMQPublisher(f"tcp://{myIP}:3097")
        
        self.policy_start_tracker = self.create_subscription(Bool, "/policy_start", self._receive_policy, 10)

        self.get_logger().info("FRANKA ARM READY!!!")
        
        self.timer = self.create_timer(1/30, self.publish_eef_pose)

    def test_function(self):
        dummy_posemsg = PoseStamped()
        # give random pose
        dummy_posemsg.pose.position.x = random.uniform(-0.5, 0.5)
        dummy_posemsg.pose.position.y = random.uniform(-0.5, 0.5)
        dummy_posemsg.pose.position.z = random.uniform(0.0, 1.0)
        self.get_logger().info(f"Sending: {dummy_posemsg}")
        self._receive_eef_pose(dummy_posemsg)
    
    def _receive_policy(self, data: Bool):
        self.policy_start = data.data

    def _receive_eef_pose(self, data: PoseStamped):
        # receives the end effector poses
        
        target_position = data.pose.position
        target_orientation = data.pose.orientation
        
        target_pose = np.array([target_position.x, target_position.y, target_position.z, target_orientation.w, target_orientation.x, target_orientation.y, target_orientation.z])
        
        self.zmq_publisher.send_message(target_pose)
    
    def publish_eef_pose(self):
        # recieve the pose using websockets
        current_pose = self.zmq_subscriber.message
        
        if current_pose is not None:
            current_pose = current_pose.astype(float)
            # publish the pose
            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            
            
            pose_msg.pose.position.x = current_pose[0]
            pose_msg.pose.position.y = current_pose[1]
            pose_msg.pose.position.z = current_pose[2]
            
            pose_msg.pose.orientation.w = current_pose[3]
            pose_msg.pose.orientation.x = current_pose[4]
            pose_msg.pose.orientation.y = current_pose[5]
            pose_msg.pose.orientation.z = current_pose[6]
            
            
            self.arm_eef_pose_pub.publish(pose_msg)
        
    def run_asyncio_loop(self):
        """Run the asyncio event loop in a dedicated thread."""
        # Start the server
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.server)
        self.loop.run_forever()

    def destroy_node(self):
        # Shutdown procedure
        self.get_logger().info("Shutting down Franka Remote Node")
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    franka_node = FrankaRemote()
    try:
        rclpy.spin(franka_node)
    except KeyboardInterrupt:
        pass
    finally:
        franka_node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
