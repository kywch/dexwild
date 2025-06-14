# Copyright (c) 2025 Tony Tao
# Licensed under the MIT License. See LICENSE file in project root for details.

"""
ROS2 node for controlling an XArm robot.
"""

import math
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, PoseStamped, Quaternion, TransformStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from tf2_ros import TransformBroadcaster

from dexwild_utils.zmq_utils import ZMQPublisher, ZMQSubscriber
from xarm.wrapper import XArmAPI

# Constants
WORLD_FRAME = "world"
DEFAULT_IP = "192.168.1.1"
ARM_Y_OFFSET = 0.46  # meters between left/right arms
JOINT_WRAP_THRESHOLD = 5.0  # radians
JOINT_DELTA_MAX = 0.01  # radians

# Default joint limits (radians)
JT_LIMITS_MIN = np.deg2rad([-359, -117, -224, -359, -96, -359])
JT_LIMITS_MAX = np.deg2rad([359, 119, 10, 359, 179, 359])
RESET_JOINTS = np.deg2rad([0, -30, -40, 0, 70, 0])

class XArm(Node):
    def __init__(self):
        super().__init__('xarm_node')
        self._declare_and_get_params()
        self._configure_arm()
        self._initialize_state()
        self._setup_ros_interface()
        self.get_logger().info("ARM READY!!!")

    def _declare_and_get_params(self):
        self.declare_parameter('xarm_ip', DEFAULT_IP)
        self.declare_parameter('arm_id', 'left_mobile')
        self.declare_parameter('teleop', False)
        self.declare_parameter('use_rmp', False)

        self.xarm_ip = self.get_parameter('xarm_ip').value
        self.arm_id = self.get_parameter('arm_id').value
        self.teleop = self.get_parameter('teleop').value
        self.use_rmp = self.get_parameter('use_rmp').value

        self.freq = 120
        self.safe = True
        self.policy_start = False

        self.get_logger().info(f"XARM_IP: {self.xarm_ip}")
        self.get_logger().info(f"ARM_ID: {self.arm_id}")

    def _configure_arm(self):
        self.arm = XArmAPI(self.xarm_ip)
        self.arm.set_simulation_robot(False)
        self.arm.motion_enable(enable=True)
        self.arm.set_tcp_load(2.5, [0, 0, 0])
        self.arm.set_tcp_offset([0, 0, 150, 0, 0, np.pi/2], is_radian=True) # leap v2
        # self.arm.set_tcp_offset([0, 0, 70, 0, 0, np.pi/2], is_radian=True) # original leap hand
        self.arm.set_mode(6)
        self.arm.set_state(0)
        self.arm.set_collision_sensitivity(1)
        self.arm.set_tcp_jerk(10000)
        self.arm.set_joint_jerk(500, is_radian=True)
        self.arm.save_conf()
        self.get_logger().info(f"WORLD OFFSET {self.arm.world_offset}")

    def _initialize_state(self):
        # Movement parameters
        self.reset_speed = 0.5
        self.reset_mvacc = 2.0
        self.speed = 3.14
        self.mvacc = 19.98

        # Joint limits and reset pose
        self.jt_min = JT_LIMITS_MIN
        self.jt_max = JT_LIMITS_MAX
        self.reset_joints = RESET_JOINTS if self.arm_id in ('left_mobile', 'right_mobile') else None

        # Joint history
        self.jts = None
        self.jts_prev = None

        if self.teleop and self.reset_joints is not None:
            self._homing_sequence()

    def _setup_ros_interface(self):
        # Subscribers
        self.create_subscription(JointState, f"/arm/{self.arm_id}/cmd_joint_angles", self._on_joint_cmd, 10)
        self.create_subscription(PoseStamped, f"/arm/{self.arm_id}/cmd_eef_pose", self._on_eef_cmd, 10)
        self.create_subscription(Bool, "/is_safe", self._on_safe, 10)
        self.create_subscription(Bool, "/policy_start", self._on_policy_start, 10)

        # Publishers
        self.arm_joint_pub = self.create_publisher(JointState, f"/arm/{self.arm_id}/obs_joint_angles", 10)
        self.arm_eef_pose_pub = self.create_publisher(PoseStamped, f"/arm/{self.arm_id}/obs_eef_pose", 10)
        self.cmd_eef_pose_pub = self.create_publisher(PoseStamped, f"/arm/{self.arm_id}/cmd_eef_pose_fk", 10)

        # TF broadcaster
        self.tf_br = TransformBroadcaster(self)

        # Timers
        self.create_timer(1/self.freq, self._publish_joint_state)
        if self.use_rmp:
            self._setup_rmp_interface()

    def _setup_rmp_interface(self):
        base_ip = "127.0.0.1"
        ports = {
            'left_mobile': (5099, 5097),
            'right_mobile': (4099, 4097)
        }
        pub_port, sub_port = ports.get(self.arm_id, (None, None))
        self.xarm_joint_publisher = ZMQPublisher(f"tcp://{base_ip}:{pub_port}")
        self.rmp_joint_subscriber = ZMQSubscriber(f"tcp://{base_ip}:{sub_port}")
        self.create_timer(1/self.freq, self._cmd_rmp_joints)

    def _homing_sequence(self):
        current = self.arm.get_servo_angle(is_radian=True)[1][:6]
        delta = np.abs(current - self.reset_joints).max()
        steps = min(int(delta / JOINT_DELTA_MAX), 100)
        self.get_logger().info("Performing teleop homing...")
        for j in np.linspace(current, self.reset_joints, steps):
            self.arm.set_servo_angle(angle=j, speed=self.reset_speed, mvacc=self.reset_mvacc, wait=False, is_radian=True)
            time.sleep(1/30)
        

    # --- Callback handlers ---
    def _on_safe(self, msg: Bool): self.safe = msg.data

    def _on_policy_start(self, msg: Bool): self.policy_start = msg.data

    def _on_joint_cmd(self, msg: JointState):
        self.jts_prev, self.jts = self.jts, np.array(msg.position)
        # self.get_logger().info(f"Joint command: {self.jts}")    
        self._unwrap_joints()
        if self.safe:
            result = self.arm.set_servo_angle(angle=self.jts, speed=self.speed, mvacc=self.mvacc, wait=False, is_radian=True)
            if result != 0:
                self.safe = False

    def _on_eef_cmd(self, msg: PoseStamped):
        pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]) * 1000
        rot = R.from_quat([msg.pose.orientation.x, msg.pose.orientation.y,
                            msg.pose.orientation.z, msg.pose.orientation.w])
        target = np.concatenate((pos, rot.as_euler('xyz')))
        if self._is_eef_safe(target) and self.safe:
            res, joints = self.arm.get_inverse_kinematics(target, input_is_radian=True, return_is_radian=True)
            if res == 0:
                self.jts_prev, self.jts = self.jts, np.array(joints[:6])
                self._unwrap_joints()
                speed, mvacc = (self.speed, self.mvacc) if self.policy_start else (self.reset_speed, self.reset_mvacc)
                self.arm.set_servo_angle(angle=self.jts, speed = speed, mvacc=mvacc, wait=False, is_radian=True)
            else:
                self.get_logger().warn(f"IK failure for target {target}")

    # --- Helpers ---
    def _unwrap_joints(self):
        if self.jts_prev is None:
            return
        diff = self.jts - self.jts_prev
        over = diff > JOINT_WRAP_THRESHOLD
        under = diff < -JOINT_WRAP_THRESHOLD
        self.jts[over] -= 2*math.pi
        self.jts[under] += 2*math.pi
        if np.any(self.jts < self.jt_min) or np.any(self.jts > self.jt_max):
            self.jts = self.jts_prev

    def _is_eef_safe(self, eef):
        x = eef[0]
        y = eef[1]
        z = eef[2]
        if z < 166:
            self.get_logger().warn(f"Unsafe EEF z={eef[2]}")
            return False
        return True

    # --- Publishing ---
    def _publish_joint_state(self):
        state = self.arm.get_servo_angle(is_radian=True)[1][:6]
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.position = list(state)
        self.arm_joint_pub.publish(msg)
        self._publish_eef_pose()
        if self.use_rmp:
            code, (pos, vel, _) = self.arm.get_joint_states(is_radian=True)
            data = np.concatenate((pos, vel))
            self.xarm_joint_publisher.send_message(data)

    def _publish_eef_pose(self):
        res, pose = self.arm.get_position_aa(is_radian=True)
        if res != 0:
            self.get_logger().warn("FK position failed")
            return

        pose = np.array(pose)           # Convert to NumPy array
        pose[:3] /= 1000                # Convert mm to meters

        msg = PoseStamped()
        msg.header.frame_id = self.arm_id
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = pose[0]
        msg.pose.position.y = pose[1]
        msg.pose.position.z = pose[2]

        quat = R.from_rotvec(pose[3:]).as_quat()
        msg.pose.orientation.x = quat[0]
        msg.pose.orientation.y = quat[1]
        msg.pose.orientation.z = quat[2]
        msg.pose.orientation.w = quat[3]
        
        self.arm_eef_pose_pub.publish(msg)

        tf = TransformStamped()
        tf.header.stamp = msg.header.stamp
        tf.header.frame_id = WORLD_FRAME
        tf.child_frame_id = self.arm_id
        tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = (
            pose[0], pose[1] + (ARM_Y_OFFSET if self.arm_id == 'left_mobile' else 0), pose[2]
        )
        tf.transform.rotation.x = quat[0]
        tf.transform.rotation.y = quat[1]
        tf.transform.rotation.z = quat[2]
        tf.transform.rotation.w = quat[3]
        self.tf_br.sendTransform(tf)

    def _cmd_rmp_joints(self):
        data = self.rmp_joint_subscriber.message
        if data is not None:
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.position = data.tolist()
            self._on_joint_cmd(msg)

    def destroy_node(self):
        self.get_logger().info("Shutting down XArm node...")
        if self.teleop and self.reset_joints is not None:
            self._homing_sequence()
        self.arm.set_state(4)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = XArm()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
