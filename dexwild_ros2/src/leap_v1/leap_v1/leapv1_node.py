#!/usr/bin/env python3

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

import dexwild_utils.dynamixel_client as dxl
import dexwild_utils.leap_hand_utils as lhu
from dexwild_interfaces.srv  import LeapPosition, LeapVelocity, LeapEffort, LeapPosVelEff
import math
import csv

#LEAP hand conventions:
#180 is flat out home pose for the index, middle, ring, finger MCPs.
#Applying a positive angle closes the joints more and more to curl closed.
#The MCP is centered at 180 and can move positive or negative to that.

#The joint numbering goes from Index (0-3), Middle(4-7), Ring(8-11) to Thumb(12-15) and from MCP Side, MCP Forward, PIP, DIP for each finger.
#For instance, the MCP Side of Index is ID 0, the MCP Forward of Ring is 9, the DIP of Ring is 11

#I recommend you only query when necessary and below 90 samples a second.  Used the combined commands if you can to save time.  Also don't forget about the USB latency settings in the readme.
#The services allow you to always have the latest data when you want it, and not spam the communication lines with unused data.

class LeapNode(Node):
    def __init__(self):
        super().__init__('leaphand_node')
        # Some parameters to control the hand
        self.kP = self.declare_parameter('kP', 800.0).get_parameter_value().double_value
        self.kI = self.declare_parameter('kI', 0.0).get_parameter_value().double_value
        self.kD = self.declare_parameter('kD', 200.0).get_parameter_value().double_value
        self.curr_lim = self.declare_parameter('curr_lim', 350.0).get_parameter_value().double_value
        
        isLeft = self.declare_parameter('isLeft',False).get_parameter_value().bool_value
        
        self.port = self.declare_parameter('port',"").get_parameter_value().string_value
        
        self.get_logger().info(f"Using port: {self.port}")
        
        self.ema_amount = 0.2
        self.prev_pos = self.pos = self.curr_pos = lhu.allegro_to_LEAPhand(np.zeros(16))

        # Subscribes to a variety of sources that can command the hand
        self.create_subscription(JointState, 'cmd_leap', self._receive_pose, 10)
        
        if isLeft:
            self.create_subscription(JointState, '/leaphand_node/cmd_allegro_left', self._receive_allegro, 10)
        else:
            self.create_subscription(JointState, '/leaphand_node/cmd_allegro_right', self._receive_allegro, 10)
            self.create_subscription(JointState, '/leapv2_node/cmd_raw_leap_r', self._receive_leapv2, 10)        

        # Creates services that can give information about the hand out
        self.create_service(LeapPosition, 'leap_position', self.pos_srv)
        self.create_service(LeapVelocity, 'leap_velocity', self.vel_srv)
        self.create_service(LeapEffort, 'leap_effort', self.eff_srv)
        self.create_service(LeapPosVelEff, 'leap_pos_vel_eff', self.pos_vel_eff_srv)
        self.create_service(LeapPosVelEff, 'leap_pos_vel', self.pos_vel_srv)
        # You can put the correct port here or have the node auto-search for a hand at the first 3 ports.
        # For example ls /dev/serial/by-id/* to find your LEAP Hand. Then use the result.  
        # For example: /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT7W91VW-if00-port0
        self.motors = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        
        self.dxl_client = dxl.DynamixelClient(self.motors, self.port, 4000000)
        
        self.dxl_client.connect()

        # Enables position-current control mode and the default parameters
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * 5, 11, 1)
        self.dxl_client.set_torque_enabled(self.motors, True)
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * self.kP, 84, 2)  # Pgain stiffness     
        self.dxl_client.sync_write([0,4,8], np.ones(3) * (self.kP * 0.75), 84, 2)  # Pgain stiffness for side to side should be a bit less
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * self.kI, 82, 2)  # Igain
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * self.kD, 80, 2)  # Dgain damping
        self.dxl_client.sync_write([0,4,8], np.ones(3) * (self.kD * 0.75), 80, 2)  # Dgain damping for side to side should be a bit less
        # Max at current (in unit 1ma) so don't overheat and grip too hard
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * self.curr_lim, 102, 2)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)
        
    def _receive_leapv2(self, msg):
        
        '''
        LEAPV2
        This receives 17 dof motor joints from 0 -> closed angle.
        Finger Order: index = [0:3], middle = [3:6], ring = [6:9], pinky = [9:12], thumb = [12:15], palm_thumb = [15], palm_fingers = [16]
        Joint order: MCP side[0,3,6,9,12], MCP forward[1,4,7,10,(13 actually thumb mcp forward)], PIP/DIP [2,5,8,11,(14 actually MCP thumb tendon)] palm_thumb = [15], palm_4_fingers = [16]
        '''
        v2_motors_side =    [0,3,6,9,12]
        v2_motors_forward = [1,4,7,10,13]
        v2_motors_curl =    [2,5,8,11,14]
        v2_motors_palm =    [15,16]  # 15 is for the thumb, 16 is between the 4 fingers, 
        v2_all_motors = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
        
        pose = msg.position
        leapv2_joints = np.array(pose)
        leapv2_joints_normed = np.zeros(17)
        
        #for MCP side, set 0 to be straight forward by adding 180 to map to robot space  (including motor 12 for thumb)
        zero_to_one = lhu.unscale(leapv2_joints[v2_motors_side], -1.57, 1.57)
        leapv2_joints_normed[v2_motors_side] = zero_to_one
        
        #set the mcp forward motors
        zero_to_one = lhu.unscale(leapv2_joints[v2_motors_forward],np.zeros(len(v2_motors_forward)),[1.57, 1.57, 1.57, 1.57, 1.57]) #[2.28, 2.059, 2.059, 2.28,2.28]
        zero_to_one = np.clip(zero_to_one, 0.005, 0.995)
        leapv2_joints_normed[v2_motors_forward] = zero_to_one
        
        #for curl, make it so that you control motor from 0 to 1.  Then we assume the soft printed finger can move 1.57 rad at each joint.  We then map angle input to this script to 0,1
        zero_to_one = lhu.unscale(leapv2_joints[v2_motors_curl], 0, 1.57) 
        zero_to_one = np.clip(zero_to_one, 0.005, 0.995)
        leapv2_joints_normed[v2_motors_curl] = zero_to_one
        
        ##thumb palm forward
        zero_to_one = lhu.unscale(leapv2_joints[15], 0, 1.04)  
        zero_to_one = np.clip(zero_to_one, 0.005, 0.995)
        leapv2_joints_normed[15] = zero_to_one
        
        ##4 fingers palm forward
        zero_to_one = lhu.unscale(leapv2_joints[16], 0, 1.22)  
        zero_to_one = np.clip(zero_to_one, 0.005, 0.995)
        leapv2_joints_normed[16] = zero_to_one
        
        leapv1_joints = np.zeros(16)
        
        self.nominal_open = [-3.42902317e-01, -4.13555283e-01, -1.62902839e-01,
                            4.44754828e-01, -5.98419838e-03, -4.23198776e-01, -1.73425600e-01,
                            4.45409983e-01,  3.16566506e-01, -3.91376854e-01, -2.06551253e-01,
                            5.48279969e-01,  2.05204916e-02, -1.18676704e-01,  3.90739471e-01,
                            -9.40697476e-03
                            ]
        
        self.nominal_closed =  [4.98404110e-02, 1.76172334e+00, 5.71870863e-01,
                            1.47162994e+00, 4.52223428e-01, 2.00516025e+00, 4.97504339e-01,
                            1.51511221e+00, 2.83616293e-01, 2.06586776e+00, 4.33664997e-01,
                            1.52440990e+00, 8.41671372e-01, 7.92767966e-01, 2.95032727e-01,
                            1.26972666e+00]
        
        #LEAPV1: The joint numbering goes from Index (0-3), Middle(4-7), Ring(8-11) to Thumb(12-15) and from MCP Side, MCP Forward, PIP, DIP for each finger.
        leapv1_joints[0:3] = leapv2_joints_normed[0:3] * self.nominal_closed[0:3] + (1 - leapv2_joints_normed[0:3]) * self.nominal_open[0:3]
        leapv1_joints[1] += leapv2_joints_normed[16] * self.nominal_closed[1] + (1 - leapv2_joints_normed[16]) * self.nominal_open[1] # account for palm forward
        leapv1_joints[3] = leapv2_joints_normed[2] * self.nominal_closed[3] + (1 - leapv2_joints_normed[2]) * self.nominal_open[3] + 1
        
        leapv1_joints[4:7] = leapv2_joints_normed[3:6] * self.nominal_closed[4:7] + (1 - leapv2_joints_normed[3:6]) * self.nominal_open[4:7]
        leapv1_joints[5] += leapv2_joints_normed[16] * self.nominal_closed[5] + (1 - leapv2_joints_normed[16]) * self.nominal_open[5] # account for palm forward
        leapv1_joints[7] = leapv2_joints_normed[5] * self.nominal_closed[7] + (1 - leapv2_joints_normed[5]) * self.nominal_open[7] + 1
        
        leapv1_joints[8:11] = leapv2_joints_normed[6:9] * self.nominal_closed[8:11] + (1 - leapv2_joints_normed[6:9]) * self.nominal_open[8:11]
        leapv1_joints[9] += leapv2_joints_normed[16] * self.nominal_closed[9] + (1 - leapv2_joints_normed[16]) * self.nominal_open[9] # account for palm forward
        leapv1_joints[11] = leapv2_joints_normed[8] * self.nominal_closed[11] + (1 - leapv2_joints_normed[8]) * self.nominal_open[11] + 1
        
        leapv1_joints[12:15] = leapv2_joints_normed[12:15] * self.nominal_closed[12:15] + (1 - leapv2_joints_normed[12:15]) * self.nominal_open[12:15]
        leapv1_joints[12] += leapv2_joints_normed[15] * self.nominal_closed[12] + (1 - leapv2_joints_normed[15]) * self.nominal_open[12] # account for thumb palm forward
        leapv1_joints[14] = leapv2_joints_normed[15] * self.nominal_closed[14] + (1 - leapv2_joints_normed[15]) * self.nominal_open[14]
    
        leapv1_joints[15] = leapv2_joints_normed[11] * self.nominal_closed[15] + (1 - leapv2_joints_normed[11]) * self.nominal_open[15]
        
        # use ema
        
        pose = lhu.allegro_to_LEAPhand(leapv1_joints)
        self.curr_pos = np.array(pose)
        # use ema
        self.curr_pos = self.prev_pos * (1 - self.ema_amount) + self.curr_pos * self.ema_amount
        self.prev_pos = self.curr_pos
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    # Receive LEAP pose and directly control the robot.  Fully open here is 180 and increases in this value closes the hand.
    def _receive_pose(self, msg):
        pose = msg.position
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    #Allegro compatibility, first read the allegro publisher and then convert to leap
    #It adds 180 to the input to make the fully open position at 0 instead of 180.
    def _receive_allegro(self, msg):
        pose = lhu.allegro_to_LEAPhand(msg.position, zeros=False)
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.curr_pos = self.prev_pos * (1 - self.ema_amount) + self.curr_pos * self.ema_amount
        self.prev_pos = self.curr_pos
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    # Sim compatibility, first read the sim publisher and then convert to leap
    #Sim compatibility for policies, it assumes the ranges are [-1,1] and then convert to leap hand ranges.
    def _receive_ones(self, msg):
        pose = lhu.sim_ones_to_LEAPhand(np.array(msg.position))
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    # Service that reads and returns the pos of the robot in regular LEAP Embodiment scaling.
    def pos_srv(self, request, response):
        response.position = self.dxl_client.read_pos().tolist()
        return response

    # Service that reads and returns the vel of the robot in LEAP Embodiment
    def vel_srv(self, request, response):
        response.velocity = self.dxl_client.read_vel().tolist()
        return response

    # Service that reads and returns the effort/current of the robot in LEAP Embodiment
    def eff_srv(self, request, response):
        response.effort = self.dxl_client.read_cur().tolist()
        return response
    #Use these combined services to save a lot of latency if you need multiple datapoints
    def pos_vel_srv(self, request, response):
        output = self.dxl_client.read_pos_vel()
        response.position = output[0].tolist()
        response.velocity = output[1].tolist()
        response.effort = np.zeros_like(output[1]).tolist()
        return response
    #Use these combined services to save a lot of latency if you need multiple datapoints
    def pos_vel_eff_srv(self, request, response):
        output = self.dxl_client.read_pos_vel_cur()
        response.position = output[0].tolist()
        response.velocity = output[1].tolist()
        response.effort = output[2].tolist()
        return response

    def _read_limits(self, file_name):
        file_path = file_name
        lower = [] #home position
        upper = [] #closed position
        with open(file_path, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in reader:
                lower.append(math.radians(float(row[0])))
                upper.append(math.radians(float(row[1])))
        return np.array(lower), np.array(upper)

def main(args=None):
    rclpy.init(args=args)
    leaphand_node = LeapNode()
    rclpy.spin(leaphand_node)
    leaphand_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
