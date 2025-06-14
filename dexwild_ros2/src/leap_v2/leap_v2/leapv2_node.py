# Copyright (c) 2025 Tony Tao
# Licensed under the MIT License. See LICENSE file in project root for details.

import math
import time
import csv
import copy

import rclpy
from rclpy.node import Node
import numpy as np

from sensor_msgs.msg import JointState
from std_msgs.msg import String
from dexwild_interfaces.srv import LeapPosition, LeapVelocity, LeapEffort

import leap_v2_utils.dynamixel_client as dxl
import leap_v2_utils.leap_v2_utils as lv2u


class LeapvtwoNode(Node):
    def __init__(self, device=None):
        super().__init__('leapv2_node')

        port = self.declare_parameter('port', "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT78LTIG-if00-port0").get_parameter_value().string_value
        isLeft = self.declare_parameter('isLeft', False).get_parameter_value().bool_value
        self.curr_pos = np.zeros(17)
        self.get_logger().info(f"{port}")

        from ament_index_python.packages import get_package_share_directory
        path_src = get_package_share_directory("leap_v2") + "/../../../../src/leap_v2/aligner/alignments"
        path_src += "/test_L.csv" if isLeft else "/test_R.csv"

        self.disable_palm = False

        # Motor groupings
        self.motors_side =    [0, 3, 6, 9, 12]
        self.motors_forward = [1, 4, 7, 10, 13]
        self.motors_curl =    [2, 5, 8, 11, 14]
        self.motors_palm =    [15, 16]
        self.all_motors = list(range(17))

        self.dxl_client = dxl.DynamixelClient(self.all_motors, port, 4000000)
        self.dxl_client.connect()

        self.strength_scale = 1

        # Side Motors (XC330)
        self._configure_motors(self.motors_side, 5, 100, mode=11)

        # Forward Motors (XM430)
        self.open_align, self.close_align = self._read_limits(path_src)
        self.limits_min_max = np.sort(np.stack([self.open_align, self.close_align]), axis=0)
        self._configure_motors(self.motors_forward, 5, 650 / 2.69, mode=11)

        # Curl Motors (XM430)
        self._configure_motors(self.motors_curl, 5, 700 / 2.69, mode=11)

        # Palm Motors (XM430)
        self._configure_motors(self.motors_palm, 5, 700 / 2.69, mode=11)

        self._compute_home_pose_offset()

        self.dxl_client.set_torque_enabled(self.all_motors, True)
        self._setup_ros_interfaces(isLeft)

    def _configure_motors(self, motors, mode_val, current_val, mode):
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * mode_val, 11, 1)
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * current_val * self.strength_scale, 102, 2)
        self.dxl_client.set_torque_enabled(motors, True)

    def _compute_home_pose_offset(self):
        curr_pos = self.dxl_client.read_pos()
        self.home_pose_offset = np.zeros(17)
        for i in range(17):
            if (curr_pos[i] + self.home_pose_offset[i]) < (self.limits_min_max[0][i] - 2):
                self.home_pose_offset[i] += 6.28
            elif (curr_pos[i] + self.home_pose_offset[i]) > (self.limits_min_max[1][i] + 2):
                self.home_pose_offset[i] -= 6.28

    def _setup_ros_interfaces(self, isLeft):
        prefix = 'l' if isLeft else 'r'
        self.create_subscription(JointState, f"cmd_leap_{prefix}", self._receive_pose, 10)
        self.create_subscription(JointState, f"/leapv2_node/cmd_raw_leap_{prefix}", self._receive_raw_joints, 10)
        self.create_subscription(JointState, f"cmd_ones_{prefix}", self._receive_ones, 10)
        self.create_service(LeapPosition, 'leap_position', self.pos_srv)
        self.create_service(LeapVelocity, 'leap_velocity', self.vel_srv)
        self.create_service(LeapEffort, 'leap_effort', self.eff_srv)

    def _receive_pose(self, pose):
        pose = pose.position
        self.prev_pos = copy.deepcopy(self.curr_pos)
        pose = np.array(pose)
        self.curr_input = np.array(lv2u.compress(pose[0:4]) + lv2u.compress(pose[4:8]) + lv2u.compress(pose[8:12]) + lv2u.compress(pose[12:16]) + pose[16:20].tolist() + [0])

        zero_to_one = lv2u.unscale(self.curr_input[self.motors_side], -1.57, 1.57)
        self.curr_pos[self.motors_side] = lv2u.scale(zero_to_one, self.open_align[self.motors_side], self.close_align[self.motors_side])

        if not self.disable_palm:
            min_forward = np.min(list(self.curr_input[self.motors_forward[0:4]]) + [0.8])
            self.curr_input[self.motors_forward[0:4]] -= min_forward
            zero_to_one = lv2u.unscale(min_forward, 0, 1.22)
            zero_to_one = np.clip(zero_to_one, 0.005, 0.995)
            self.curr_pos[16] = lv2u.scale(zero_to_one, self.open_align[16], self.close_align[16])

        zero_to_one = lv2u.unscale(self.curr_input[self.motors_forward], np.zeros(len(self.motors_forward)), [1.57]*5)
        zero_to_one = np.clip(zero_to_one, 0.005, 0.995)
        self.curr_pos[self.motors_forward] = lv2u.scale(zero_to_one, self.open_align[self.motors_forward], self.close_align[self.motors_forward])

        zero_to_one = lv2u.unscale(self.curr_input[self.motors_curl], 0, 1.57)
        zero_to_one = np.clip(zero_to_one, 0.005, 0.995)
        self.curr_pos[self.motors_curl] = lv2u.scale(zero_to_one, self.open_align[self.motors_curl], self.close_align[self.motors_curl])

        zero_to_one = lv2u.unscale(self.curr_input[14], 0, 2.28)
        zero_to_one = np.clip(zero_to_one, 0.005, 0.995)
        self.curr_pos[13] = lv2u.scale(zero_to_one, self.open_align[13], self.close_align[13])

        zero_to_one = lv2u.unscale(self.curr_input[15], 0, 1.57)
        zero_to_one = np.clip(zero_to_one, 0.005, 0.995)
        self.curr_pos[14] = lv2u.scale(zero_to_one, self.open_align[14], self.close_align[14])

        zero_to_one = lv2u.unscale(self.curr_input[13], 0, 1.04)
        zero_to_one = np.clip(zero_to_one, 0.005, 0.995)
        self.curr_pos[15] = lv2u.scale(zero_to_one, self.open_align[15], self.close_align[15])

        self.dxl_client.write_desired_pos(self.all_motors, self.curr_pos)

    '''
    This receives 17 dof motor joints from 0 -> closed angle.
    Finger Order: index = [0:3], middle = [3:6], ring = [6:9], pinky = [9:12], thumb = [12:15], palm_thumb = [15], palm_fingers = [16]
    Joint order: MCP side[0,3,6,9,12], MCP forward[1,4,7,10,(13 actually thumb mcp forward)], PIP/DIP [2,5,8,11,(14 actually MCP thumb tendon)] palm_thumb = [15], palm_4_fingers = [16]
    '''
    def _receive_raw_joints(self, pose):
        pose = pose.position
        self.prev_pos = copy.deepcopy(self.curr_pos)
        self.curr_input = np.array(pose)

        zero_to_one = lv2u.unscale(self.curr_input[self.motors_side], -1.57, 1.57)
        self.curr_pos[self.motors_side] = lv2u.scale(zero_to_one, self.open_align[self.motors_side], self.close_align[self.motors_side])

        zero_to_one = lv2u.unscale(self.curr_input[self.motors_forward], np.zeros(len(self.motors_forward)), [1.57]*5)
        zero_to_one = np.clip(zero_to_one, 0.005, 0.995)
        self.curr_pos[self.motors_forward] = lv2u.scale(zero_to_one, self.open_align[self.motors_forward], self.close_align[self.motors_forward])

        zero_to_one = lv2u.unscale(self.curr_input[self.motors_curl], 0, 1.57)
        zero_to_one = np.clip(zero_to_one, 0.005, 0.995)
        self.curr_pos[self.motors_curl] = lv2u.scale(zero_to_one, self.open_align[self.motors_curl], self.close_align[self.motors_curl])

        zero_to_one = lv2u.unscale(self.curr_input[15], 0, 1.04)
        zero_to_one = np.clip(zero_to_one, 0.005, 0.995)
        self.curr_pos[15] = lv2u.scale(zero_to_one, self.open_align[15], self.close_align[15])

        zero_to_one = lv2u.unscale(self.curr_input[16], 0, 1.22)
        zero_to_one = np.clip(zero_to_one, 0.005, 0.995)
        self.curr_pos[16] = lv2u.scale(zero_to_one, self.open_align[16], self.close_align[16])

        self.dxl_client.write_desired_pos(self.all_motors, self.curr_pos - self.home_pose_offset)

    def _receive_ones(self, pose):
        raise NotImplementedError

    def pos_srv(self, request, response):
        response.position = (self.dxl_client.read_pos() + self.home_pose_offset).tolist()
        return response

    def vel_srv(self, request, response):
        response.velocity = self.dxl_client.read_vel().tolist()
        return response

    def eff_srv(self, request, response):
        response.effort = self.dxl_client.read_cur().tolist()
        return response

    def _read_limits(self, file_name):
        lower, upper = [], []
        with open(file_name, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in reader:
                lower.append(math.radians(float(row[0])))
                upper.append(math.radians(float(row[1])))
        return np.array(lower), np.array(upper)

    def destroy_node(self):
        curl_target = lv2u.scale([0.2]*5, self.open_align[self.motors_curl], self.close_align[self.motors_curl])
        self.dxl_client.write_desired_pos(self.motors_curl, curl_target - self.home_pose_offset[self.motors_curl])
        time.sleep(1)
        self.dxl_client.set_torque_enabled(self.motors_curl, False)
        self.dxl_client.sync_write(self.motors_curl, np.zeros(len(self.motors_curl)), 11, 1)
        self.dxl_client.sync_write(self.motors_curl, np.zeros(len(self.motors_curl)), 102, 2)
        self.dxl_client.set_torque_enabled(self.motors_curl, True)
        self.get_logger().info("WAIT, LEAP V2 SHUTTING DOWN!!!")
        for _ in range(10):
            self.dxl_client.set_torque_enabled(self.all_motors, False)
            time.sleep(0.05)


def main(args=None):
    rclpy.init(args=args)
    leapv2_node = LeapvtwoNode()
    try:
        rclpy.spin(leapv2_node)
    except KeyboardInterrupt:
        pass
    finally:
        leapv2_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
