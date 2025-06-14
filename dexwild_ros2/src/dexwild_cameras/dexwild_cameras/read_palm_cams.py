# Copyright (c) 2025 Tony Tao
# Licensed under the MIT License. See LICENSE file in project root for details.

import cv2
import rclpy
from rclpy.node import Node
from std_msgs.msg import String  # You can customize the message type
from sensor_msgs.msg import Image
import pyudev
from cv_bridge import CvBridge

class PalmCameraNode(Node):
    def __init__(self):
        super().__init__('palm_camera_node')
        
        self.bridge = CvBridge()
        
        self.declare_parameter('cameras', [""])
        self.declare_parameter('serials', [""]) 
        self.declare_parameter('fourcc', "YUYV")
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 30)
        self.declare_parameter('superwide', False)

        self.sensor_choices = self.get_parameter('cameras').value 
        serials = self.get_parameter('serials').value
        self.fourcc = self.get_parameter('fourcc').value
        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value
        self.camera_fps = self.get_parameter('fps').value
        self.superwide = self.get_parameter('superwide').value
        
        self.cam_attrs = {}
        for i in range(len(self.sensor_choices)):
            cam_name = self.sensor_choices[i]
            self.cam_attrs[cam_name] = {}
            self.cam_attrs[cam_name]["serial"] = serials[i]
            hand = cam_name.split("_")[0]
            side = cam_name.split("_")[1]
            
            self.cam_attrs[cam_name]["publisher"] = self.create_publisher(Image, f'/{hand}/{side}_camera_im',  10)
            
        self.timer_period = 1/self.camera_fps  # seconds
        
        devices = self.discover_cameras()
        self.initialize_cameras(devices)
        
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
    
    def getSerialNumber(self, device):
        context = pyudev.Context()
        device_file = "/dev/video{}".format(device)
        device = pyudev.Devices.from_device_file(context, device_file)
        info = { item[0] : item[1] for item in device.items()}
        try:
            return info["ID_SERIAL_SHORT"]
        except:
            return None

    def discover_cameras(self, max_cameras=30):
        available_cameras = []
        for camera_index in range(max_cameras):
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                available_cameras.append(camera_index)
                cap.release()  # Release the camera after checking
        return available_cameras
    
    
    def initialize_cameras(self, devices):
        self.camera_count = 0
        self.get_logger().info(f"Devices: {devices}")
        for device in devices:
            cap = cv2.VideoCapture(device)
            
            if cap.isOpened():
                for key, value in self.cam_attrs.items():
                    if value["serial"] == self.getSerialNumber(device):
                        native_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        native_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        native_fps = int(cap.get(cv2.CAP_PROP_FPS))
                        native_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
                        fourcc_to_string = "".join([chr((native_fourcc >> 8 * i) & 0xFF) for i in range(4)])
                        self.get_logger().info(f"Camera {device} native resolution: {native_width}x{native_height} at {native_fps} FPS with FourCC: {fourcc_to_string}")

                        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*f'{self.fourcc}'))
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                        cap.set(cv2.CAP_PROP_FPS, self.camera_fps)
                        
                        self.get_logger().info(f"Camera {device} initialized with resolution {self.width}x{self.height} at {self.camera_fps} FPS with FourCC: {self.fourcc}")
                        
                        self.cam_attrs[key]["cap"] = cap
                        self.get_logger().info(f"Connected To {key} Camera")
                        self.camera_count += 1
                        break
                
        if self.camera_count == 0:
            self.get_logger().info("Cameras not found.")
            exit(0)
        self.get_logger().info(f"Connect to {self.camera_count} cameras!!")

    def timer_callback(self):
        
        if self.camera_count < len(self.cam_attrs.keys()):
            self.get_logger().info(f"Not all cameras connected yet")
            return
        
        # same timestamp for all cameras
        current_time = self.get_clock().now().to_msg()

        for key in self.cam_attrs.keys():

            ret, frame = self.cam_attrs[key]["cap"].read()
            
            if not ret:
                self.get_logger().info(f"Failed to grab frame from camera {key}")
                continue
            
            if not self.superwide:
                if key == 'left_thumb':
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif key == 'left_pinky':
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                elif key == 'right_thumb':
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif key == 'right_pinky':
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            else:
                if key == 'left_thumb':
                    pass
                elif key == 'left_pinky':
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                elif key == 'right_thumb':
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                elif key == 'right_pinky':
                    pass
                
            image_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            
            image_msg.header.stamp = current_time
            
            self.cam_attrs[key]["publisher"].publish(image_msg)

    def destroy_node(self):
        for key in self.cam_attrs.keys():
            self.cam_attrs[key]["cap"].release()
        self.get_logger().info("Destroying Camera Nodes")
        super().destroy_node()
        

def main(args=None):
    rclpy.init(args=args)
    camera_node = PalmCameraNode()
    try:
        rclpy.spin(camera_node)
    except KeyboardInterrupt:
        pass
    finally:
        camera_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
