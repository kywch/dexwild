# Copyright (c) 2025 Tony Tao
# Licensed under the MIT License. See LICENSE file in project root for details.

"""
ROS2 node that reads from a GelloAgent and publishes matching joint commands
for an XArm tracker, with optional RMP output and hybrid control.
Preserves original behavior with clearer structure.
"""

import glob
import numpy as np
import termcolor

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String, Bool
from dexwild_utils.zmq_utils import ZMQPublisher
from gello.agents.gello_agent import GelloAgent
from scipy.spatial.transform import Rotation as Rot

# Constants
DEFAULT_RATE = 120.0  # Hz
MATCH_THRESHOLD = 0.4
USB_GLOB = "/dev/serial/by-id/*"
LOCAL_IP = "127.0.0.1"


def print_color(*args, color=None, attrs=(), **kwargs):
    """Print colored text to console."""
    colored = [termcolor.colored(str(a), color=color, attrs=attrs) for a in args]
    print(*colored, **kwargs)

class ReadGello(Node):
    def __init__(self):
        super().__init__('read_gello')
        self._declare_parameters()
        self._init_agent()
        self._setup_publishers()
        self._setup_subscribers_and_services()
        print_color(f"Start Gello with id='{self.tracker_id}'", color="green", attrs=("bold",))
        self.matched = False
        self.control_state = None
        self.timer = self.create_timer(1.0/DEFAULT_RATE, self._on_timer)

    def _declare_parameters(self):
        self.declare_parameter('port', None)
        self.declare_parameter('id', '')
        self.declare_parameter('use_rmp', False)

        self.gello_port = self.get_parameter('port').value
        self.tracker_id = self.get_parameter('id').value
        self.use_rmp = self.get_parameter('use_rmp').value
        self.get_logger().info(f"Using RMP: {self.use_rmp}")

        if self.gello_port is None:
            ports = glob.glob(USB_GLOB)
            self.get_logger().info(f"Found {len(ports)} USB ports")
            if not ports:
                raise RuntimeError("No Gello port found; please specify one.")
            self.gello_port = ports[0]
            self.get_logger().info(f"Using port {self.gello_port}")

    def _init_agent(self):
        start_joints = np.deg2rad([0]*6)
        self.agent = GelloAgent(port=self.gello_port, start_joints=start_joints)
        self.arm_joint = None
        self.match_threshold = MATCH_THRESHOLD

    def _setup_publishers(self):
        # publish desired tracker commands
        self.pub_tracker = self.create_publisher(
            JointState, f"/arm/{self.tracker_id}/cmd_joint_angles", 10)
        # publish match state
        self.pub_match = self.create_publisher(
            Bool, f"/gello/{self.tracker_id}/match_state", 10)
        # optional RMP output
        if self.use_rmp:
            port_map = {'left_mobile': 5096, 'right_mobile': 4096}
            rp = port_map.get(self.tracker_id)
            self.zmq_pub = ZMQPublisher(f"tcp://{LOCAL_IP}:{rp}")

    def _setup_subscribers_and_services(self):
        # arm feedback
        self.create_subscription(
            JointState, f"/arm/{self.tracker_id}/obs_joint_angles",
            self._on_arm_joint, 10)
        if self.tracker_id=='left_mobile':
            self.right_matched = False
            self.create_subscription(
                Bool, "/gello/right_mobile/match_state",
                self._on_right_match, 10)
        # control state
        self.create_subscription(
            String, "/control_state", self._on_control_state, 10)

    def _on_timer(self):
        # publish match state
        self._publish_match_state()
        # get action
        action = self.agent.act([])
        # send to RMP if matched
        if self.use_rmp and self.matched:
            self.zmq_pub.send_message(action)
        # check arm_joint available
        if self.arm_joint is None:
            return
        # check matching
        if not self.matched:
            diff = np.abs(action - self.arm_joint)
            msg = f"{diff[0]}, {diff[1]}, {diff[2]}, {diff[3]}, {diff[4]}, {diff[5]}"
            # self.get_logger().info(f"Diff: {msg}")
            
            if diff.max() < self.match_threshold:
                self.matched = True
                self.get_logger().info("MATCHED!!")
            return
        # publish tracker joint command
        msg = JointState()
        msg.position = [float(x) for x in action]
        if (self.matched and not self.use_rmp):
            # return
            # self.get_logger().info(f"publishing angles {msg.position}")
            self.pub_tracker.publish(msg)

    def _publish_match_state(self):
        m = Bool(data=self.matched)
        self.pub_match.publish(m)

    def _on_arm_joint(self, msg: JointState):
        self.arm_joint = np.array(msg.position)

    def _on_right_match(self, msg: Bool):
        self.right_matched = msg.data

    def _on_control_state(self, msg: String):
        new = msg.data
        if self.control_state!='teleop' and new=='teleop':
            self.matched = False
        self.control_state = new

    def destroy_node(self):
        super().destroy_node()


def main():
    rclpy.init()
    node = ReadGello()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__=='__main__':
    main()
