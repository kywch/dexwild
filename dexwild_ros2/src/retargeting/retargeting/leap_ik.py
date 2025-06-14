# #!/usr/bin/env python3
# import pybullet as p
# import numpy as np
# import rclpy
# import os

# from rclpy.node import Node
# from geometry_msgs.msg import PoseArray
# from sensor_msgs.msg import JointState
# import sys
# from ament_index_python.packages import get_package_share_directory

# from dexwild_utils.data_processing import load_pickle

# import argparse
# import time
# import ray
# import pickle

# class Leapv1PybulletIKNoROS():
#     def __init__(self, process = False, is_left=False, use_viewer=False):   
#         # start pybullet
#         #clid = p.connect(p.SHARED_MEMORY)
#         if use_viewer:
#             clid = p.connect(p.GUI)
#         else:
#             clid = p.connect(p.DIRECT)
#         if clid < 0:
#             p.connect(p.DIRECT, options="--egl")
            
#         path_src = os.path.abspath(__file__)
#         path_src = os.path.dirname(path_src)
        
#         self.process = process
#         self.is_left = is_left
        
#         self.glove_to_leap_mapping_scale = 2.0
#         self.leapEndEffectorIndex = [3, 4, 8, 9, 13, 14, 18, 19]
        
#         if self.is_left:
#             path_src = os.path.join(path_src, "leap_hand_mesh_left/robot_pybullet.urdf")
#             ##You may have to set this path for your setup on ROS2
#             self.LeapId = p.loadURDF(
#                 path_src,
#                 [-0.05, -0.03, -0.25],
#                 p.getQuaternionFromEuler([0, 1.57, 1.57]),
#                 useFixedBase = True
#             )
#         else:
#             path_src = os.path.join(path_src, "leap_hand_mesh_right/robot_pybullet.urdf")
#             ##You may have to set this path for your setup on ROS2
#             self.LeapId = p.loadURDF(
#                 path_src,
#                 [-0.05, -0.03, -0.25],
#                 p.getQuaternionFromEuler([0, 1.57, 1.57]),
#                 useFixedBase = True
#             )
        
#         self.numJoints = p.getNumJoints(self.LeapId)
#         p.setGravity(0, 0, 0)
#         useRealTimeSimulation = 0
#         p.setRealTimeSimulation(useRealTimeSimulation)
#         self.create_target_vis()
        
#     def create_target_vis(self):
#         # load balls
#         small_ball_radius = 0.01
#         small_ball_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=small_ball_radius)
#         ball_radius = 0.01
#         ball_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=ball_radius)
#         baseMass = 0.001
#         basePosition = [0.25, 0.25, 0]
        
#         self.ballMbt = []
#         for i in range(0,4):
#             self.ballMbt.append(p.createMultiBody(baseMass=baseMass, baseCollisionShapeIndex=ball_shape, basePosition=basePosition)) # for base and finger tip joints    
#             no_collision_group = 0
#             no_collision_mask = 0
#             p.setCollisionFilterGroupMask(self.ballMbt[i], -1, no_collision_group, no_collision_mask)
#         p.changeVisualShape(self.ballMbt[0], -1, rgbaColor=[1, 0, 0, 1]) 
#         p.changeVisualShape(self.ballMbt[1], -1, rgbaColor=[0, 1, 0, 1]) 
#         p.changeVisualShape(self.ballMbt[2], -1, rgbaColor=[0, 0, 1, 1])  
#         p.changeVisualShape(self.ballMbt[3], -1, rgbaColor=[1, 1, 1, 1])
    
#     def update_target_vis(self, hand_pos):
#         _, current_orientation = p.getBasePositionAndOrientation( self.ballMbt[0])
#         p.resetBasePositionAndOrientation(self.ballMbt[0], hand_pos[3], current_orientation)
#         _, current_orientation = p.getBasePositionAndOrientation(self.ballMbt[1])
#         p.resetBasePositionAndOrientation(self.ballMbt[1], hand_pos[2], current_orientation)
#         _, current_orientation = p.getBasePositionAndOrientation(self.ballMbt[2])
#         p.resetBasePositionAndOrientation(self.ballMbt[2], hand_pos[7], current_orientation)
#         _, current_orientation = p.getBasePositionAndOrientation(self.ballMbt[3])
#         p.resetBasePositionAndOrientation(self.ballMbt[3], hand_pos[1], current_orientation)
    
#     def get_glove_data(self, pose):
#         #gets the data converts it and then computes IK and visualizes
#         hand_pos = []  
#         if self.process:
#             pose = pose
#             for i in range(0,10):
#                 hand_pos.append([pose[i, 0] * self.glove_to_leap_mapping_scale * 1.15, pose[i, 1] * self.glove_to_leap_mapping_scale, -pose[i, 2] * self.glove_to_leap_mapping_scale])
#         else:
#             poses = pose.poses
#             for i in range(0,10):
#                 hand_pos.append([poses[i].position.x * self.glove_to_leap_mapping_scale * 1.15, poses[i].position.y * self.glove_to_leap_mapping_scale, -poses[i].position.z * self.glove_to_leap_mapping_scale])
        
#         # hand_pos[2][0] = hand_pos[2][0] - 0.02  this isn't great because they won't oppose properly
#         # hand_pos[3][0] = hand_pos[3][0] - 0.02    
#         # hand_pos[6][0] = hand_pos[6][0] + 0.02
#         # hand_pos[7][0] = hand_pos[7][0] + 0.02
#         #hand_pos[2][1] = hand_pos[2][1] + 0.002
#         hand_pos[4][1] = hand_pos[4][1] + 0.002
#         hand_pos[6][1] = hand_pos[6][1] + 0.002
#         leap_joints = self.compute_IK(hand_pos)
#         self.update_target_vis(hand_pos)
        
#         return leap_joints
    
#     def compute_IK(self, hand_pos):
#         p.stepSimulation()     

#         rightHandIndex_middle_pos = hand_pos[2]
#         rightHandIndex_pos = hand_pos[3]
        
#         rightHandMiddle_middle_pos = hand_pos[4]
#         rightHandMiddle_pos = hand_pos[5]
        
#         rightHandRing_middle_pos = hand_pos[6]
#         rightHandRing_pos = hand_pos[7]
        
#         rightHandThumb_middle_pos = hand_pos[0]
#         rightHandThumb_pos = hand_pos[1]
        
#         leapEndEffectorPos = [
#             rightHandIndex_middle_pos,
#             rightHandIndex_pos,
#             rightHandMiddle_middle_pos,
#             rightHandMiddle_pos,
#             rightHandRing_middle_pos,
#             rightHandRing_pos,
#             rightHandThumb_middle_pos,
#             rightHandThumb_pos
#         ]

#         jointPoses = p.calculateInverseKinematics2(
#             self.LeapId,
#             self.leapEndEffectorIndex,
#             leapEndEffectorPos,
#             solver=p.IK_DLS,
#             maxNumIterations=50,
#             residualThreshold=0.0001,
#         )
        
#         combined_jointPoses = (jointPoses[0:4] + (0.0,) + jointPoses[4:8] + (0.0,) + jointPoses[8:12] + (0.0,) + jointPoses[12:16] + (0.0,))
#         combined_jointPoses = list(combined_jointPoses)

#         # update the hand joints
#         for i in range(20):
#             p.setJointMotorControl2(
#                 bodyIndex=self.LeapId,
#                 jointIndex=i,
#                 controlMode=p.POSITION_CONTROL,
#                 targetPosition=combined_jointPoses[i],
#                 targetVelocity=0,
#                 force=500,
#                 positionGain=0.3,
#                 velocityGain=1,
#             )

#         # map results to real robot
#         real_robot_hand_q = np.array([float(0.0) for _ in range(16)])
#         #real_left_robot_hand_q = np.array([0.0 for _ in range(16)])

#         real_robot_hand_q[0:4] = jointPoses[0:4]
#         real_robot_hand_q[4:8] = jointPoses[4:8]
#         real_robot_hand_q[8:12] = jointPoses[8:12]
#         real_robot_hand_q[12:16] = jointPoses[12:16]
#         real_robot_hand_q[12] = 3 * np.pi / 2 - np.pi # fix the thumb position
#         real_robot_hand_q[0:2] = real_robot_hand_q[0:2][::-1]
#         real_robot_hand_q[4:6] = real_robot_hand_q[4:6][::-1]
#         real_robot_hand_q[8:10] = real_robot_hand_q[8:10][::-1]
        
#         return [float(i) for i in real_robot_hand_q]
        
#     def disconnect(self):
#         p.disconnect()

# class LeapPybulletIK(Node):
#     def __init__(self, is_left=False, use_viewer=False):
#         super().__init__('leap_pyb_ik')  

#         # load right leap hand      
#         self.is_left = self.declare_parameter('isLeft', False).get_parameter_value().bool_value
#         use_viewer = self.declare_parameter('useViewer', True).get_parameter_value().bool_value
        
#         self.pybulletIK = Leapv1PybulletIKNoROS(False, self.is_left, use_viewer)
        
#         self.get_logger().info(f"Is left: {self.is_left}")
        
#         if self.is_left:
#             self.pub_hand = self.create_publisher(JointState, '/leaphand_node/cmd_allegro_left', 10)
#             self.sub_skeleton = self.create_subscription(PoseArray, "/glove/l_short", self.get_glove_data, 10)
        
#         else:
#             self.pub_hand = self.create_publisher(JointState, '/leaphand_node/cmd_allegro_right', 10)
#             self.sub_skeleton = self.create_subscription(PoseArray, "/glove/r_short", self.get_glove_data, 10)
        
    
#     def get_glove_data(self, pose):
#         #gets the data converts it and then computes IK and visualizes
#         leap_joints = self.pybulletIK.get_glove_data(pose)
        
#         stater = JointState()
#         stater.header.stamp = self.get_clock().now().to_msg()
#         stater.position = leap_joints
#         self.pub_hand.publish(stater)
        
#         return leap_joints

#     def destroy_node(self):
#         self.pybulletIK.disconnect()
#         super().destroy_node()
         
# def main(args=None):
#     rclpy.init(args=args)
#     leappybulletik = LeapPybulletIK()
#     rclpy.spin(leappybulletik)
#     # Destroy the node explicitly
#     # (optional - otherwise it will be done automatically
#     # when the garbage collector destroys the node object)
#     leappybulletik.destroy_node()
#     rclpy.shutdown()

# @ray.remote
# def ray_process_glove_data_v1(glove_path, leap_v1_path, is_left, use_viewer):
#     return process_glove_data_v1(glove_path, leap_v1_path, is_left, use_viewer)

# def process_glove_data_v1(glove_path, leap_v1_path, is_left, use_viewer):
#     try:
#         start = time.time()

#         leapv1_pybullet_ik = Leapv1PybulletIKNoROS(True, is_left, use_viewer)
        
#         try:
#             glove_poses = load_pickle(glove_path)
#         except Exception as e:
#             print(f"Error loading glove poses: {e}")
#             return 0
        
#         if np.isnan(glove_poses).any():
#             print(f"NAN found in glove poses {glove_path}")
#             leapv1_pybullet_ik.disconnect()
#             if os.path.exists(leap_v1_path):
#                 os.remove(leap_v1_path)
#             return 0
        
#         glove_times = glove_poses[:, 0]
#         glove_poses  = glove_poses[:, 1:] #remove the time column
        
#         leap_v1_joints = []
        
#         for t in range(glove_poses.shape[0]):
#             curr_glove_pose = glove_poses[t]
#             curr_glove_pose = curr_glove_pose.reshape(-1, 7)
#             ik_hand_pose = leapv1_pybullet_ik.get_glove_data(curr_glove_pose)
#             # quit()
#             leap_v1_joints.append(ik_hand_pose)
#             if use_viewer:
#                 time.sleep(1/30)
        
#         leapv1_pybullet_ik.disconnect()
        
#         leap_v1_joints = np.array(leap_v1_joints)
        
#         stamped = np.hstack((glove_times.reshape(-1, 1), leap_v1_joints))
        
#         with open(leap_v1_path, 'wb') as f:
#             pickle.dump(stamped, f, protocol=pickle.HIGHEST_PROTOCOL)
        
#         elapsed = time.time() - start
#         return elapsed
    
#     except Exception as e:
#         print(f"Error processing glove data: {e}")
#         return 0

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Process Manus Short PKL files")
#     parser.add_argument(
#         '--episiode_path', 
#         '-p',
#         type=str, 
#         required=False, 
#         help="Path to the Manus file to process."
#     )
#     parser.add_argument("--isleft",
#                         "-l", 
#                         action="store_true", 
#                         help="Is the glove on the left hand?")
    
#     parser.add_argument(
#         '--use_viewer',
#         '-v', 
#         action='store_true', 
#         help="Enable 3D visualization with the Pybullet."
#     )
    
#     args = parser.parse_args()
    
#     episode_path = args.episiode_path
#     isleft = args.isleft
#     use_viewer = args.use_viewer
    
#     if episode_path is None:
#         main()
#     else:
#         if isleft:
#             glove_path = os.path.join(episode_path, 'left_manus', 'left_manus.pkl')
#             leap_v1_path= os.path.join(episode_path, 'left_leapv1', 'left_leapv1.pkl')
#         else:
#             glove_path = os.path.join(episode_path, 'right_manus', 'right_manus.pkl')
#             leap_v1_path= os.path.join(episode_path, 'right_leapv1', 'right_leapv1.pkl')
    
#         leap_v2_joints = process_glove_data_v1(glove_path, leap_v1_path, isleft, use_viewer)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025 Tony Tao
# Licensed under the MIT License. See LICENSE file in project root for details.

"""
Leapv1PybulletIK simulates inverse kinematics (IK) for a Leap V1 robotic hand using PyBullet.
Supports both headless (DIRECT) and GUI modes, ROS2 integration for live control,
and offline batch processing for glove demonstration data.
"""

import os
import sys
import time
import argparse
import pickle

import numpy as np
import pybullet as p
import ray
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray
from sensor_msgs.msg import JointState


# Utility to load glove pose data
from dexwild_utils.data_processing import load_pickle

BASE_PATH = os.path.expanduser("~/dexwild")
class Leapv1PybulletIKNoROS:
    """
    Standalone PyBullet IK solver for Leap V1 hand.
    """
    def __init__(self, process=False, is_left=False, use_viewer=False):
        # Connect to PyBullet (GUI or DIRECT)
        if use_viewer:
            clid = p.connect(p.GUI)
        else:
            clid = p.connect(p.DIRECT)
        if clid < 0:
            # Fallback to EGL headless
            p.connect(p.DIRECT, options="--egl")

        # Base directory for URDF and assets
        self.process = process
        self.is_left = is_left
        self.glove_to_leap_mapping_scale = 2.0

        # Indices of Leap end-effectors for IK solver
        self.leapEndEffectorIndex = [3, 4, 8, 9, 13, 14, 18, 19]

        # Choose URDF based on hand side
        urdf_path = os.path.join(BASE_PATH, "dexwild_ros2", "src", "leap_v1", "leap_v1", "leap_hand_left" if is_left else "leap_hand_right", "robot_pybullet.urdf")

        # Load the hand URDF at a fixed pose
        init_pos = [-0.05, -0.03, -0.25]
        init_ori = p.getQuaternionFromEuler([0, 1.57, 1.57])
        self.LeapId = p.loadURDF(urdf_path, init_pos, init_ori, useFixedBase=True)

        # Configure simulation
        p.setGravity(0, 0, 0)
        p.setRealTimeSimulation(0)

        # Prepare visualization markers
        self.create_target_vis()

    def create_target_vis(self):
        """
        Create colored spheres at fingertip and base locations for visualization.
        """
        # Sphere collision shape and basic properties
        radius = 0.01
        col_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
        mass = 0.001
        base_pos = [0.25, 0.25, 0]

        # Instantiate one sphere per finger landmark
        self.ballMbt = []
        for _ in range(4):
            ball = p.createMultiBody(baseMass=mass,
                                      baseCollisionShapeIndex=col_shape,
                                      basePosition=base_pos)
            # disable collisions for markers
            p.setCollisionFilterGroupMask(ball, -1, 0, 0)
            self.ballMbt.append(ball)

        # Assign distinct colors
        colors = [[1,0,0,1], [0,1,0,1], [0,0,1,1], [1,1,1,1]]
        for idx, ball in enumerate(self.ballMbt):
            p.changeVisualShape(ball, -1, rgbaColor=colors[idx])

    def update_target_vis(self, hand_pos):
        """
        Reposition visual markers based on current hand landmark positions.
        """
        for idx, ball in enumerate(self.ballMbt):
            pos = hand_pos[[3,2,7,1][idx]]  # map ball index to hand_pos index
            _, ori = p.getBasePositionAndOrientation(ball)
            p.resetBasePositionAndOrientation(ball, pos, ori)

    def get_glove_data(self, pose):
        """
        Convert incoming PoseArray or numpy array to hand landmark positions,
        run IK, update markers, and return computed joint angles.
        """
        # Build list of scaled landmark positions
        hand_pos = []
        if self.process:
            # pose is a numpy array shape (N,7)
            for i in range(10):
                x, y, z = pose[i, 0], pose[i, 1], pose[i, 2]
                hand_pos.append([x * self.glove_to_leap_mapping_scale * 1.15,
                                 y * self.glove_to_leap_mapping_scale,
                                 -z * self.glove_to_leap_mapping_scale])
        else:
            # pose is a ROS PoseArray
            for pmsg in pose.poses:
                hand_pos.append([pmsg.position.x * self.glove_to_leap_mapping_scale * 1.15,
                                 pmsg.position.y * self.glove_to_leap_mapping_scale,
                                 -pmsg.position.z * self.glove_to_leap_mapping_scale])

        # Minor adjustments for more realistic closure
        hand_pos[4][1] += 0.002
        hand_pos[6][1] += 0.002

        # Compute IK and visualize
        joints = self.compute_IK(hand_pos)
        self.update_target_vis(hand_pos)
        return joints

    def compute_IK(self, hand_pos):
        """
        Run PyBullet inverse kinematics and map to real robot joint layout.
        """
        # Step simulation to update internal state
        p.stepSimulation()

        # Assign named variables for clarity
        rip, rip_mid = hand_pos[3], hand_pos[2]
        rpm, rpm_mid = hand_pos[5], hand_pos[4]
        rrp, rrp_mid = hand_pos[7], hand_pos[6]
        rtp, rtp_mid = hand_pos[1], hand_pos[0]

        # Define end-effector targets in required order
        ee_targets = [rip_mid, rip, rpm_mid, rpm, rrp_mid, rrp, rtp_mid, rtp]

        # Compute joint angles via DLS solver
        jp = p.calculateInverseKinematics2(
            self.LeapId,
            self.leapEndEffectorIndex,
            ee_targets,
            solver=p.IK_DLS,
            maxNumIterations=50,
            residualThreshold=1e-4
        )

        # Fill missing joint slots with zeros
        cj = list(jp[0:4] + (0.0,) + jp[4:8] + (0.0,) + jp[8:12] + (0.0,) + jp[12:16] + (0.0,))

        # Apply to sim joints
        for i in range(20):
            p.setJointMotorControl2(
                bodyIndex=self.LeapId,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=cj[i],
                force=500,
                positionGain=0.3,
                velocityGain=1
            )

        # Map to real 16-DOF robot format
        real_q = np.zeros(16)
        real_q[0:4] = jp[0:4]
        real_q[4:8] = jp[4:8]
        real_q[8:12] = jp[8:12]
        real_q[12:16] = jp[12:16]
        # Thumb fix and axis reversals
        real_q[12] = 3*np.pi/2 - np.pi
        real_q[0:2] = real_q[0:2][::-1]
        real_q[4:6] = real_q[4:6][::-1]
        real_q[8:10] = real_q[8:10][::-1]

        return [float(v) for v in real_q]

    def disconnect(self):
        """
        Cleanly disconnect from the PyBullet simulation.
        """
        p.disconnect()


class LeapPybulletIK(Node):
    """
    ROS2 Node wrapper for real-time Leap V1 IK.
    """
    def __init__(self, is_left=False, use_viewer=False):
        super().__init__('leap_pyb_ik')

        # Read parameters
        self.is_left = self.declare_parameter('isLeft', False).get_parameter_value().bool_value
        uv = self.declare_parameter('useViewer', True).get_parameter_value().bool_value

        # Instantiate IK solver
        self.pybulletIK = Leapv1PybulletIKNoROS(False, self.is_left, uv)
        self.get_logger().info(f"Leap V1 IK (is_left={self.is_left}) initialized.")

        # Publishers and subscriptions based on side
        topic = '/leaphand_node/cmd_allegro_left' if self.is_left else '/leaphand_node/cmd_allegro_right'
        glove_topic = '/glove/l_short' if self.is_left else '/glove/r_short'

        self.pub_hand = self.create_publisher(JointState, topic, 10)
        self.sub_skel = self.create_subscription(PoseArray, glove_topic, self.get_glove_data, 10)

    def get_glove_data(self, pose):
        """
        Handle incoming PoseArray, run IK, publish JointState.
        """
        joints = self.pybulletIK.get_glove_data(pose)
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.position = joints
        self.pub_hand.publish(msg)
        return joints

    def destroy_node(self):
        # Disconnect sim before cleanup
        self.pybulletIK.disconnect()
        super().destroy_node()


def main(args=None):
    """ROS2 entry point if launched as a standalone node."""
    rclpy.init(args=args)
    node = LeapPybulletIK()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


# ========== Offline Batch API ==========
def process_glove_data_v1(glove_path, leap_v1_path, is_left, use_viewer):
    """
    Batch-process glove PKL files to generate Leap V1 joint outputs.
    """
    try:
        start = time.time()
        ik = Leapv1PybulletIKNoROS(True, is_left, use_viewer)

        glove_poses = load_pickle(glove_path)
        if np.isnan(glove_poses).any():
            print(f"NAN in {glove_path}")
            ik.disconnect()
            os.remove(leap_v1_path) if os.path.exists(leap_v1_path) else None
            return 0

        # Separate time and reshape pose data
        times = glove_poses[:,0]
        raw = glove_poses[:,1:]

        results = []
        for i in range(raw.shape[0]):
            frame = raw[i].reshape(-1,7)
            results.append(ik.get_glove_data(frame))
            if use_viewer:
                time.sleep(1/30)

        ik.disconnect()
        stamped = np.hstack((times.reshape(-1,1), np.array(results)))
        with open(leap_v1_path,'wb') as f:
            pickle.dump(stamped, f, protocol=pickle.HIGHEST_PROTOCOL)

        return time.time() - start
    except Exception as e:
        print(f"Error in batch processing: {e}")
        return 0


@ray.remote
def ray_process_glove_data_v1(glove_path, leap_v1_path, is_left, use_viewer):
    return process_glove_data_v1(glove_path, leap_v1_path, is_left, use_viewer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Process Manus Short PKL files")
    parser.add_argument('-p','--episiode_path', type=str, help="Base folder path.")
    parser.add_argument('-l','--isleft', action='store_true', help="Left-hand flag.")
    parser.add_argument('-v','--use_viewer', action='store_true', help="Enable GUI.")
    args = parser.parse_args()

    if args.episiode_path:
        base = args.episiode_path
        side = 'left' if args.isleft else 'right'
        glove = os.path.join(base,f'{side}_manus',f'{side}_manus.pkl')
        leap = os.path.join(base,f'{side}_leapv1',f'{side}_leapv1.pkl')
        process_glove_data_v1(glove, leap, args.isleft, args.use_viewer)
    else:
        main()
