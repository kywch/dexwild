# Copyright (c) 2025 Tony Tao
# Licensed under the MIT License. See LICENSE file in project root for details.

"""
Leapv2PybulletIK simulates inverse kinematics (IK) for a LeapV2 robotic hand using PyBullet.
This code supports both GUI and headless (DIRECT) simulation, ROS integration for real-time operation,
and offline data processing for human glove demonstrations.
"""

import os
import time
import numpy as np
import pybullet as p
import ray
import pickle
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseArray

from dexwild_utils.data_processing import load_pickle

BASE_PATH = os.path.expanduser("~/dexwild")

class Leapv2PybulletIKNoROS:
    """
    Standalone PyBullet-based LeapV2 inverse kinematics solver.
    This class does not depend on ROS and is used for both real-time and batch processing.
    """
    def __init__(self, process=False, is_left=False, use_viewer=False):
        # Connect to PyBullet
        if use_viewer:
            clid = p.connect(p.GUI)
        else:
            clid = p.connect(p.DIRECT)
        if clid < 0:
            p.connect(p.DIRECT, options="--egl")

        self.process = process
        self.is_left = is_left
        self.glove_to_leap_mapping_scale = 1.6

        # Set gravity and simulation settings
        p.setGravity(0, 0, 0)
        p.setRealTimeSimulation(0)

        # Define PIP-DIP joint pairs for post-processing smoothing
        self.pip_dip_pairs = [(9,10), (15,16), (21,22), (27,28)] if is_left else [(3,4), (9,10), (15,16), (21,22)]

        # Define end-effector indices (tip and middle joints)
        self.leapEndEffectorIndex = [
            4, 5, 11, 12, 17, 18, 23, 24, 29, 30
        ] if is_left else [
            29, 30, 5, 6, 11, 12, 17, 18, 23, 24
        ]

        # Load robot URDF and apply fixed transform
        urdf_path = os.path.join(BASE_PATH, "dexwild_ros2", "src", "leap_v2", "leap_v2", "leap_v2_left" if is_left else "leap_v2_right", "robot.urdf")
        
        pyb_xyz = [-0.02, 0.15, -0.2] if is_left else [0.02, 0.15, -0.2]
        pyb_euler = [1.57, 0, 3.14]

        self.LeapId = p.loadURDF(
            urdf_path,
            pyb_xyz,
            p.getQuaternionFromEuler(pyb_euler),
            useFixedBase=True
        )

        self.numJoints = p.getNumJoints(self.LeapId)
        self.create_target_vis()

    def create_target_vis(self):
        """Creates visual markers for fingertips and palm."""
        radius = 0.01
        shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
        mass = 0.001
        base_pos = [0.25, 0.25, 0]

        self.ballMbt = []
        for _ in range(5):
            ball = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=shape, basePosition=base_pos)
            p.setCollisionFilterGroupMask(ball, -1, 0, 0)
            self.ballMbt.append(ball)

        dip_ball = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=shape, basePosition=base_pos)
        p.setCollisionFilterGroupMask(dip_ball, -1, 0, 0)
        self.ballMbt.append(dip_ball)

        # Assign colors for visualization
        colors = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
        for i, color in enumerate(colors):
            p.changeVisualShape(self.ballMbt[i], -1, rgbaColor=color)

    def update_target_vis(self, hand_pos):
        """Updates visual marker positions for hand targets."""
        indices = [3, 5, 7, 9, 1]  # thumb, index, middle, ring, pinky (mid joints)
        for i, idx in enumerate(indices):
            _, ori = p.getBasePositionAndOrientation(self.ballMbt[i])
            p.resetBasePositionAndOrientation(self.ballMbt[i], hand_pos[idx], ori)
        # palm marker
        _, ori = p.getBasePositionAndOrientation(self.ballMbt[5])
        p.resetBasePositionAndOrientation(self.ballMbt[5], [0, 0, 0], ori)

    def get_glove_data(self, pose):
        """
        Convert incoming glove pose data into scaled fingertip positions,
        compute IK, and visualize the targets.
        """
        hand_pos = []
        x_scale = 0.7
        pinky_scale = self.glove_to_leap_mapping_scale * 1.2
        thumb_scale = self.glove_to_leap_mapping_scale

        if self.process:
            poses = pose
            for i in range(0, 2):
                hand_pos.append([poses[i, 0] * thumb_scale, poses[i, 1] * thumb_scale, -poses[i, 2] * thumb_scale])
            for i in range(2, 8):
                hand_pos.append([poses[i, 0] * x_scale * self.glove_to_leap_mapping_scale,
                                 poses[i, 1] * self.glove_to_leap_mapping_scale,
                                 -poses[i, 2] * self.glove_to_leap_mapping_scale])
            for i in range(8, 10):
                hand_pos.append([poses[i, 0] * x_scale * pinky_scale,
                                 poses[i, 1] * pinky_scale,
                                 -poses[i, 2] * pinky_scale])
        else:
            poses = pose.poses
            for i in range(0, 2):
                hand_pos.append([poses[i].position.x * thumb_scale,
                                 poses[i].position.y * thumb_scale,
                                 -poses[i].position.z * thumb_scale])
            for i in range(2, 8):
                hand_pos.append([poses[i].position.x * x_scale * self.glove_to_leap_mapping_scale,
                                 poses[i].position.y,
                                 -poses[i].position.z * self.glove_to_leap_mapping_scale])
            for i in range(8, 10):
                hand_pos.append([poses[i].position.x * x_scale * pinky_scale,
                                 poses[i].position.y,
                                 -poses[i].position.z * pinky_scale])

        leap_joints = self.compute_IK(hand_pos)
        self.update_target_vis(hand_pos)
        return leap_joints

    def compute_IK(self, hand_pos):
        """
        Compute inverse kinematics for the LeapV2 hand using fingertip positions.
        Also applies smoothing to DIP-PIP pairs and maps the results to real robot joint angles.
        """
        p.stepSimulation()

        # Reorganize hand_pos to match expected end-effector ordering
        leapEndEffectorPos = [
            hand_pos[1], hand_pos[0],  # thumb_pos, thumb_middle_pos
            hand_pos[3], hand_pos[2],  # index
            hand_pos[5], hand_pos[4],  # middle
            hand_pos[7], hand_pos[6],  # ring
            hand_pos[9], hand_pos[8]   # pinky
        ]

        jointPoses = p.calculateInverseKinematics2(
            self.LeapId,
            self.leapEndEffectorIndex,
            leapEndEffectorPos,
            solver=p.IK_DLS,
            maxNumIterations=30,
            residualThreshold=0.001
        )

        # Fill unused joints with zeros
        if self.is_left:
            combined_jointPoses = (
                jointPoses[0:4] + (0.0, 0.0,) +
                jointPoses[4:9] + (0.0, 0.0,) +
                jointPoses[9:13] + (0.0, 0.0,) +
                jointPoses[13:17] + (0.0, 0.0,) +
                jointPoses[17:21] + (0.0, 0.0,)
            )
        else:
            combined_jointPoses = (
                jointPoses[0:5] + (0.0, 0.0,) +
                jointPoses[5:9] + (0.0, 0.0,) +
                jointPoses[9:13] + (0.0, 0.0,) +
                jointPoses[13:17] + (0.0, 0.0,) +
                jointPoses[17:21] + (0.0, 0.0,)
            )

        combined_jointPoses = list(combined_jointPoses)

        # Average DIP and PIP joint pairs for better smoothness
        for pip_id, dip_id in self.pip_dip_pairs:
            avg = (combined_jointPoses[pip_id] + combined_jointPoses[dip_id]) / 2
            combined_jointPoses[pip_id] = avg
            combined_jointPoses[dip_id] = avg

        # Apply IK results to simulation
        for i in range(min(len(combined_jointPoses), 31)):
            p.setJointMotorControl2(
                bodyIndex=self.LeapId,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=combined_jointPoses[i],
                targetVelocity=0,
                force=500,
                positionGain=0.3,
                velocityGain=1.0
            )

        # Map joint poses to the 17-DOF real robot format
        real_q = np.zeros(17)
        jp = np.array(combined_jointPoses)

        if self.is_left:
            real_q[12:15] = [-jp[2], jp[1], jp[3]]  # thumb
            real_q[0:3] = [-jp[8], jp[7], jp[9]]    # index
            real_q[3:6] = [-jp[14], jp[13], jp[15]] # middle
            real_q[6:9] = [-jp[20], jp[19], jp[21]] # ring
            real_q[9:12] = [-jp[26], jp[25], jp[27]]# pinky
            real_q[15] = jp[0]                      # palm thumb
            real_q[16] = jp[6]                      # palm fingers
        else:
            real_q[12:15] = [jp[27], jp[26], jp[28]]
            real_q[0:3] = [jp[2], jp[1], jp[3]]
            real_q[3:6] = [jp[8], jp[7], jp[9]]
            real_q[6:9] = [jp[14], jp[13], jp[15]]
            real_q[9:12] = [jp[20], jp[19], jp[21]]
            real_q[15] = jp[25]
            real_q[16] = jp[0]

        # Add offset to MCP forward joints for better closure
        real_q[[1, 4, 7, 10]] += 0.2

        return real_q.tolist()

    def disconnect(self):
        """Disconnect the PyBullet simulation."""
        p.disconnect()

class Leapv2PybulletIK(Node):
    """
    ROS2 Node wrapper for real-time inference and command publishing using PyBullet-based IK.
    """
    def __init__(self, is_left=False, use_viewer=False):
        super().__init__('leapv2_pyb_ik')
        self.is_left = self.declare_parameter('isLeft', False).get_parameter_value().bool_value
        self.pybulletIK = Leapv2PybulletIKNoROS(False, self.is_left, use_viewer)
        self.get_logger().info(f"Is left: {self.is_left}")

        if self.is_left:
            self.pub_hand = self.create_publisher(JointState, '/leapv2_node/cmd_raw_leap_l', 10)
            self.sub_skeleton = self.create_subscription(PoseArray, '/glove/l_short', self.get_glove_data, 10)
        else:
            self.pub_hand = self.create_publisher(JointState, '/leapv2_node/cmd_raw_leap_r', 10)
            self.sub_skeleton = self.create_subscription(PoseArray, '/glove/r_short', self.get_glove_data, 10)

        self.curr_stater = None
        self.freq = 60  # Hz
        self.timer = self.create_timer(1.0 / self.freq, self.timer_callback)

    def timer_callback(self):
        if self.curr_stater is not None:
            self.curr_stater.header.stamp = self.get_clock().now().to_msg()
            self.pub_hand.publish(self.curr_stater)

    def get_glove_data(self, pose):
        """Callback for incoming glove pose data; runs IK and publishes joint state."""
        leap_joints = self.pybulletIK.get_glove_data(pose)
        stater = JointState()
        stater.header.stamp = self.get_clock().now().to_msg()
        stater.position = leap_joints
        self.curr_stater = stater
        return leap_joints

    def destroy_node(self):
        self.pybulletIK.disconnect()
        super().destroy_node()


def main(args=None):
    """Main ROS2 entry point if used as a node."""
    rclpy.init(args=args)
    leapv2pybulletik = Leapv2PybulletIK()
    rclpy.spin(leapv2pybulletik)
    rclpy.shutdown()


# ========== Offline Processing API ==========

def process_glove_data(glove_path, leap_v2_path, is_left, use_viewer):
    """
    Batch processes glove trajectories and saves corresponding LeapV2 joint commands.
    """
    try:
        start = time.time()
        ik_solver = Leapv2PybulletIKNoROS(True, is_left, use_viewer)

        glove_poses = load_pickle(glove_path)
        if np.isnan(glove_poses).any():
            print(f"NaNs found in glove poses: {glove_path}")
            ik_solver.disconnect()
            if os.path.exists(leap_v2_path):
                os.remove(leap_v2_path)
            return 0

        glove_times = glove_poses[:, 0]
        glove_poses = glove_poses[:, 1:].reshape(-1, 10, 7)

        leap_v2_joints = []
        for pose in glove_poses:
            joints = ik_solver.get_glove_data(pose)
            leap_v2_joints.append(joints)
            if use_viewer:
                time.sleep(1 / 30.0)

        ik_solver.disconnect()
        stamped = np.hstack((glove_times.reshape(-1, 1), np.array(leap_v2_joints)))
        with open(leap_v2_path, 'wb') as f:
            pickle.dump(stamped, f, protocol=pickle.HIGHEST_PROTOCOL)

        return time.time() - start

    except Exception as e:
        print(f"Error processing glove data: {e}")
        return 0


@ray.remote
def ray_process_glove_data(glove_path, leap_v2_path, is_left, use_viewer):
    return process_glove_data(glove_path, leap_v2_path, is_left, use_viewer)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Process Manus Short PKL files")
    parser.add_argument('--episiode_path', '-p', type=str, required=False)
    parser.add_argument('--isleft', '-l', action='store_true')
    parser.add_argument('--use_viewer', '-v', action='store_true')
    args = parser.parse_args()

    if args.episiode_path is None:
        main()
    else:
        base = args.episiode_path
        if args.isleft:
            glove_path = os.path.join(base, 'left_manus', 'left_manus.pkl')
            leap_v2_path = os.path.join(base, 'left_leapv2', 'left_leapv2.pkl')
        else:
            glove_path = os.path.join(base, 'right_manus', 'right_manus.pkl')
            leap_v2_path = os.path.join(base, 'right_leapv2', 'right_leapv2.pkl')

        process_glove_data(glove_path, leap_v2_path, args.isleft, args.use_viewer)