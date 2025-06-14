# Copyright (c) 2025 Tony Tao
# Licensed under the MIT License. See LICENSE file in project root for details.

import cv2
import pyzed.sl as sl
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool  # You can customize the message type
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from dexwild_interfaces.msg import StampedInt
from dexwild_utils.aruco_utils import CubeTracker
from cv_bridge import CvBridge
import threading
import os
import time
import numpy as np
import pickle

class ZedNode(Node):
    def __init__(self):
        super().__init__('zed_node')

        self.declare_parameter('serial', 16352271)
        self.declare_parameter("visualize", False)
        
        self.zed_serial = self.get_parameter('serial').value
        self.visualize = self.get_parameter('visualize').value
        
        self.cameras = {}
        
        self.output_dir = os.path.expanduser("~/dexwild/data/")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.all_dirs = os.listdir(self.output_dir)
        
        self.camera_fps = 60
        
        self.initialize_cameras()
        
        self.is_recording = False
        
        self.record_sub = self.create_subscription(Bool, '/zed_recording', self.recording_callback, 10)
        self.ep_sub = self.create_subscription(StampedInt, '/curr_ep', self.episode_callback, 10)
        
        self.episodes = []
        
        self.zed_pub = self.create_publisher(StampedInt, '/zed_ts', 10)
        
        camera_info = self.zed.get_camera_information()
        calibration_params = camera_info.camera_configuration.calibration_parameters.left_cam

        fx = calibration_params.fx
        fy = calibration_params.fy
        cx = calibration_params.cx
        cy = calibration_params.cy
        #disto: [k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4]
        disto = calibration_params.disto

        k1 = disto[0]
        k2 = disto[1]
        p1 = disto[2]
        p2 = disto[3]
        k3 = disto[4]
        
        print(f"Camera intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
        print(f"Camera distortion coefficients: k1={k1}, k2={k2}, p1={p1}, p2={p2}, k3={k3}")
        
        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        
        dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)

        self.bridge = CvBridge()
        if self.visualize:
            self.image_pub = self.create_publisher(Image, '/zed/im', 10) 
    
        self.timer = self.create_timer(1/self.camera_fps, self.timer_callback) # might go back to 30hz -> see if 100 works better
        
    def initialize_cameras(self):
        
        init_params = sl.InitParameters()
        
        init_params.camera_resolution = sl.RESOLUTION.HD720 #sl.RESOLUTION.VGA
        init_params.camera_fps = self.camera_fps
        init_params.coordinate_units = sl.UNIT.METER
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE #sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP
        init_params.sensors_required = True
        
        self.output_dir = os.path.join(self.output_dir, 'zed_recordings')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        tries = 5

        for i in range(tries):
            all_detected_cameras = sl.Camera.get_device_list()
            self.get_logger().info(f"Trying to open Zed for the {i} time")
            self.get_logger().info(f"Detected {len(all_detected_cameras)} Cameras")
            for camera in all_detected_cameras:
                self.get_logger().info(f"Found camera with serial number {camera.serial_number}")
                if camera.serial_number == self.zed_serial:
                    self.zed = sl.Camera()
                    status = self.zed.open(init_params)
                    if status != sl.ERROR_CODE.SUCCESS:
                        print("Camera Open : "+repr(status)+". Exit program.")
                        exit()
                    else:
                        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.LED_STATUS, True)
                        self.get_logger().info(f"ZED Opened Successfully on try {i}")
                        break
                
            if self.zed is not None:
                break

            time.sleep(5)
                    
        if self.zed is None:
            self.get_logger().error("Could not find the ZED camera")
            exit()
                    
    def episode_callback(self, msg):
        self.episodes.append(msg.data)
    
    def recording_callback(self, msg):
        if not self.is_recording:
            output_file = os.path.join(self.output_dir, "output.svo2")
            record_params = sl.RecordingParameters(output_file, compression_mode = sl.SVO_COMPRESSION_MODE.H264)
            err = self.zed.enable_recording(record_params)
            if err == sl.ERROR_CODE.SUCCESS:
                self.get_logger().info(f"Recording started for zed")
            self.is_recording = True
        else:
            self.zed.disable_recording()
            self.zed.set_camera_settings(sl.VIDEO_SETTINGS.LED_STATUS, False)
            start_ep = min(self.episodes)
            end_ep = max(self.episodes)
            os.rename(os.path.join(self.output_dir, "output.svo2"), os.path.join(self.output_dir, f"output_{start_ep}_{end_ep}.svo2"))
            self.get_logger().info(f"Recording stopped for zed")
            self.episodes = []
            
            self.is_recording = False

    def timer_callback(self):
        start = time.time()
        image = sl.Mat()
        err = self.zed.grab()
        if err != sl.ERROR_CODE.SUCCESS:
            self.get_logger().error(f"Failed to grab frame: {err}")
        timestamp = self.zed.get_timestamp(sl.TIME_REFERENCE.IMAGE)
        timestamp_nsec = timestamp.get_nanoseconds()
        
        current_time =  self.get_clock().now().to_msg()
        
        self.zed.retrieve_image(image, sl.VIEW.LEFT)
        image_data = image.get_data()
        image_bgr = cv2.cvtColor(image_data, cv2.COLOR_BGRA2BGR)
                
        if self.visualize:
            lower_res = cv2.resize(image_bgr, (320, 240))
            image_msg = self.bridge.cv2_to_imgmsg(lower_res, encoding="bgr8")
            image_msg.header.stamp = current_time
            self.image_pub.publish(image_msg)
        
        # Create and publish the timestamp as a ROS String message
        timestamp_msg = StampedInt()
        timestamp_msg.header.stamp = current_time
        timestamp_msg.data = timestamp_nsec
        self.zed_pub.publish(timestamp_msg)
        
        effective_hertz = 1/(time.time() - start)
        
        # if effective_hertz < self.camera_fps:
        #     self.get_logger().warn(f"WARNING: Effective hz: {effective_hertz}")

    def destroy_node(self):
        self.zed.close()
        self.get_logger().info("Destroyed ZED Nodes")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    zed_node = ZedNode()
    try:
        rclpy.spin(zed_node)
    except KeyboardInterrupt:
        pass
    finally:
        zed_node.destroy_node()
        rclpy.shutdown()
        
if __name__ == '__main__': 
    main()