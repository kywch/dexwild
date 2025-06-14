import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseArray, Point, Pose, Quaternion
from sensor_msgs.msg import JointState
import zmq
import copy

'''
This reads from websockets from Manus SDK and republishes to each glove topic

The joint level data is what Manus estimates your skeleton as in the order of thumb to pinky and MCP side, MCP forward, PIP, DIP.

The full skeleton is the xyz quaternion of every single 
'''

IP_ADDRESS = "tcp://localhost:8000"
LEFT_GLOVE_SN =  "60f3738b" #"45a7fc8f" 
RIGHT_GLOVE_SN =  '431ea0a1' #"8569617b"

DEBUG = False

class GloveReader(Node):
    def __init__(self):
        super().__init__('glove_reader')
        #Connect to Server
        context = zmq.Context()
        self.socket = context.socket(zmq.PULL)
        self.socket.setsockopt(zmq.CONFLATE, True)
        
        self.socket.connect(IP_ADDRESS)

        self.pub_left = self.create_publisher(JointState, "/glove/l_joints", 10)
        self.pub_right = self.create_publisher(JointState, "/glove/r_joints",  10)

        self.pub_skeleton_right_full = self.create_publisher(PoseArray, '/glove/r_full',  10)
        self.pub_skeleton_left_full = self.create_publisher(PoseArray, '/glove/l_full',  10)

        self.pub_skeleton_right_short = self.create_publisher(PoseArray, '/glove/r_short',  10)
        self.pub_skeleton_left_short = self.create_publisher(PoseArray, '/glove/l_short',  10)
        
        self.declare_parameter('left_glove_sn', LEFT_GLOVE_SN) #replace with your gloves (all lowercase letters)
        self.declare_parameter('right_glove_sn', RIGHT_GLOVE_SN)
        
        #replace with your gloves (all lowercase letters)
        self.left_glove_sn = self.get_parameter('left_glove_sn').value
        self.right_glove_sn = self.get_parameter('right_glove_sn').value
        
        self.freq = 60
        
        self.timer1 = self.create_timer(1/self.freq, self.timer_callback)
        
        self.l_short_msg = None
        self.r_short_msg = None
        self.l_full_msg = None
        self.r_full_msg = None

        # For preventing redundant publishing
        self.last_l_full_msg = None
        self.last_r_full_msg = None
        self.last_l_short_msg = None
        self.last_r_short_msg = None
        
        self.same_count_l_full = 0
        self.same_count_r_full = 0
        self.same_count_l_short = 0
        self.same_count_r_short = 0
        
        self.same_count_threshold = 100
    
    def pose_array_content_equal(self, msg1, msg2, tol=1e-6):
        """
        Compare two PoseArray messages ignoring header stamps.
        Returns True if they have the same number of poses and each pose's
        position and orientation values are equal within a tolerance.
        """
        if len(msg1.poses) != len(msg2.poses):
            return False
        for p1, p2 in zip(msg1.poses, msg2.poses):
            if abs(p1.position.x - p2.position.x) > tol or \
               abs(p1.position.y - p2.position.y) > tol or \
               abs(p1.position.z - p2.position.z) > tol or \
               abs(p1.orientation.x - p2.orientation.x) > tol or \
               abs(p1.orientation.y - p2.orientation.y) > tol or \
               abs(p1.orientation.z - p2.orientation.z) > tol or \
               abs(p1.orientation.w - p2.orientation.w) > tol:
                return False
        return True
        
    def timer_callback(self):
        current_time = self.get_clock().now().to_msg()
        
        # Check and publish left full skeleton if needed
        if self.l_full_msg is not None:
            if self.last_l_full_msg is not None and \
                self.pose_array_content_equal(self.l_full_msg, self.last_l_full_msg):
                self.same_count_l_full += 1
            else:
                self.same_count_l_full = 0
            # Store a deep copy of current message for next comparison
            self.last_l_full_msg = copy.deepcopy(self.l_full_msg)
            if self.same_count_l_full < self.same_count_threshold:
                self.l_full_msg.header.stamp = current_time
                self.pub_skeleton_left_full.publish(self.l_full_msg)
            else:
                self.get_logger().info(f"Skipping publishing left full skeleton; unchanged for {self.same_count_threshold} cycles.")
                    
        # Check and publish right full skeleton if needed
        if self.r_full_msg is not None:
            if self.last_r_full_msg is not None and \
                self.pose_array_content_equal(self.r_full_msg, self.last_r_full_msg):
                self.same_count_r_full += 1
            else:
                self.same_count_r_full = 0
            self.last_r_full_msg = copy.deepcopy(self.r_full_msg)
            if self.same_count_r_full < self.same_count_threshold:
                self.r_full_msg.header.stamp = current_time
                self.pub_skeleton_right_full.publish(self.r_full_msg)
            else:
                self.get_logger().info(f"Skipping publishing right full skeleton; unchanged for {self.same_count_threshold} cycles.")
                    
        # Check and publish left short skeleton if needed
        if self.l_short_msg is not None:
            if self.last_l_short_msg is not None and \
                self.pose_array_content_equal(self.l_short_msg, self.last_l_short_msg):
                self.same_count_l_short += 1
            else:
                self.same_count_l_short = 0
            self.last_l_short_msg = copy.deepcopy(self.l_short_msg)
            if self.same_count_l_short < self.same_count_threshold:
                self.l_short_msg.header.stamp = current_time
                self.pub_skeleton_left_short.publish(self.l_short_msg)
            else:
                self.get_logger().info(f"Skipping publishing left short skeleton; unchanged for {self.same_count_threshold} cycles.")
                    
        # Check and publish right short skeleton if needed
        if self.r_short_msg is not None:
            if self.last_r_short_msg is not None and \
                self.pose_array_content_equal(self.r_short_msg, self.last_r_short_msg):
                self.same_count_r_short += 1
            else:
                self.same_count_r_short = 0
            self.last_r_short_msg = copy.deepcopy(self.r_short_msg)
            if self.same_count_r_short < self.same_count_threshold:
                self.r_short_msg.header.stamp = current_time
                self.pub_skeleton_right_short.publish(self.r_short_msg)
            else:
                self.get_logger().info(f"Skipping publishing right short skeleton; unchanged for {self.same_count_threshold} cycles.")

    #If you set a flag in the C++ code, you can send all the data that comes from the raw skeleton of the glove.  This data is from thumb to pinky, across all joints from palm to fingertip.   This can slow things down though
    def parse_full_skeleton_and_send(self, data, timestamp):
        skeleton_list = []
        for i in range(0,25):
            position = Point(x=float(data[1 + i*7]), y=float(data[2 + i*7]), z=float(data[3 + i*7]))  #the first ID is right or left glove don't forget
            orientation = Quaternion(x=float(data[4 + i*7]), y=float(data[5 + i*7]), z=float(data[6 + i*7]), w=float(data[7 + i*7]))
            pose = Pose(position=position, orientation=orientation)
            skeleton_list.append(pose)
        output_array_msg = PoseArray()
        output_array_msg.header.stamp = timestamp
        output_array_msg.poses = skeleton_list
        if data[0] == self.left_glove_sn:
            if DEBUG:
                self.get_logger().info("Sending full left")
            self.l_full_msg = output_array_msg
        elif data[0] == self.right_glove_sn:
            if DEBUG:
                self.get_logger().info("Sending full right")
            self.r_full_msg = output_array_msg
        else:
            print("Glove serial number incorrect!")
            print(data[0])
            
    #This the dexcap style data, you only get the fingertip and the previous joint xyz as the data and then you can send that.  It goes from thumb_middle, thumb_tip, index_middle, index_tip etc.etc.
    def parse_short_skeleton_and_send(self, data, timestamp):
        output_array_msg = PoseArray()
        output_array_msg.header.stamp = timestamp
        #short_idx = [3, 4, 8, 9, 13, 14, 18, 19, 23, 24] 
        ##Right now the integrated mode is in a different ordering, pinky, thumb, index, ring, middle
        ##Will be fixed to match the SDK in a future release 
        short_idx = [23, 24, 4, 5, 9, 10, 19, 20, 14, 15] 
        
        for i in short_idx:
            position = Point(x=float(data[1 + i*7]), y=float(data[2 + i*7]), z=float(data[3 + i*7]))  #the first ID is right or left glove don't forget
            orientation = Quaternion(x=float(data[4 + i*7]), y=float(data[5 + i*7]), z=float(data[6 + i*7]), w=float(data[7 + i*7]))
            pose = Pose(position=position, orientation=orientation)
            output_array_msg.poses.append(pose)
        if data[0] == self.left_glove_sn:
            if DEBUG:
                self.get_logger().info("Sending left")
            self.l_short_msg = output_array_msg
        elif data[0] == self.right_glove_sn:
            if DEBUG:
                self.get_logger().info("Sending right")
            self.r_short_msg = output_array_msg
        else:
            print("Glove serial number incorrect!")  
            print(data[0])      

def main(args=None):
    rclpy.init(args=args)
    glove_reader = GloveReader()
    while rclpy.ok():
        rclpy.spin_once(glove_reader, timeout_sec=0)  
        message = glove_reader.socket.recv()
        #receive the message from the socket
        message = message.decode('utf-8')
        data = message.split(",")  
        if data is not None:
            try:
                timestamp = glove_reader.get_clock().now().to_msg()
                # glove_reader.get_logger().info(f"Received data from glove: {len(data)}")
                #If joint level data
                if len(data) == 40:
                    # glove_reader.get_logger().info(f"Received joint level data from glove: {len(data)}")
                    stater_msg = JointState()
                    stater_msg.header.stamp = timestamp
                    stater_msg.position = list(map(float,data[0:20]))
                    glove_reader.pub_left.publish(stater_msg)
                    stater_msg.position = list(map(float,data[20:40]))
                    glove_reader.pub_right.publish(stater_msg)
                #If full skeleton data two hands
                elif len(data) == 352:
                    # glove_reader.get_logger().info(f"Received full skeleton data from glove: {len(data)}")
                    glove_reader.parse_full_skeleton_and_send(data[0:176], timestamp)
                    glove_reader.parse_full_skeleton_and_send(data[176:352], timestamp)
                    glove_reader.parse_short_skeleton_and_send(data[0:176], timestamp)
                    glove_reader.parse_short_skeleton_and_send(data[176:352], timestamp)
                        
                #If full skeleton data one hand
                elif len(data) == 176:
                    glove_reader.parse_full_skeleton_and_send(data[0:176], timestamp)
                    glove_reader.parse_short_skeleton_and_send(data[0:176], timestamp)
                #     # glove_reader.get_logger().info(f"Time taken: {time.time() - start}")    
                
            except KeyboardInterrupt as e:
                return
            except Exception as e:
                print(e)
                pass
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    glove_reader.destroy_node()
    rclpy.shutdown()
    

if __name__ == '__main__':
    main()