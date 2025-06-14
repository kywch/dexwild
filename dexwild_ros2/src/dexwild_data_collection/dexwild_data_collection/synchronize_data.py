# Copyright (c) 2025 Tony Tao
# Licensed under the MIT License. See LICENSE file in project root for details.

import os
import subprocess
import datetime
import numpy as np
import cv2
import time
from dexwild_utils.RecordUI import RecordUI
from dexwild_utils.data_processing import save_pickle
import threading
import queue
import shutil
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from sensor_msgs.msg import JointState, Image, CompressedImage
from geometry_msgs.msg import PoseStamped, PoseArray
from message_filters import Subscriber, ApproximateTimeSynchronizer
from dexwild_interfaces.msg import StampedInt
from cv_bridge import CvBridge
from pynput import keyboard

# topics is dictionary with key as topic name and value as the type of message
TOPICS = {
         '/left/thumb_camera_im': {'msg_type': Image, 'save_folder' : 'left_thumb_cam', 'save_file': None},
         '/left/pinky_camera_im': {'msg_type': Image, 'save_folder' : 'left_pinky_cam', 'save_file': None},
         '/right/thumb_camera_im': {'msg_type': Image, 'save_folder' : 'right_thumb_cam', 'save_file': None},
         '/right/pinky_camera_im': {'msg_type': Image, 'save_folder' : 'right_pinky_cam', 'save_file': None},
         '/zed/im': {'msg_type': Image, 'save_folder' : 'zed_obs', 'save_file': None},
         '/head/camera_im': {'msg_type': Image, 'save_folder' : 'head_cam', 'save_file': None},
         '/head/gray_camera_im': {'msg_type': CompressedImage, 'save_folder' : 'head_gray_cam', 'save_file': None},
         '/leapv2_node/cmd_raw_leap_l': {'msg_type': JointState, 'save_folder' : 'left_leapv2', 'save_file': 'left_leapv2.pkl'},
         '/leapv2_node/cmd_raw_leap_r': {'msg_type': JointState, 'save_folder' : 'right_leapv2', 'save_file': 'right_leapv2.pkl'},
         '/leaphand_node/cmd_allegro_left': {'msg_type': JointState, 'save_folder' : 'left_leapv1', 'save_file': 'left_leapv1.pkl'},
         '/leaphand_node/cmd_allegro_right': {'msg_type': JointState, 'save_folder' : 'right_leapv1', 'save_file': 'right_leapv1.pkl'},
         '/glove/l_short': {'msg_type': PoseArray, 'save_folder' : 'left_manus', 'save_file': 'left_manus.pkl'},
         '/glove/l_full': {'msg_type': PoseArray, 'save_folder' : 'left_manus', 'save_file': 'left_manus_full.pkl'},
         '/glove/l_joints': {'msg_type': JointState, 'save_folder' : 'left_manus', 'save_file': 'left_manus_joints.pkl'},
         '/glove/r_short': {'msg_type': PoseArray, 'save_folder' : 'right_manus', 'save_file': 'right_manus.pkl'},
         '/glove/r_full': {'msg_type': PoseArray, 'save_folder' : 'right_manus', 'save_file': 'right_manus_full.pkl'},
         '/glove/r_joints': {'msg_type': JointState, 'save_folder' : 'right_manus', 'save_file': 'right_manus_joints.pkl'},
         '/arm/left_mobile/obs_eef_pose': {'msg_type': PoseStamped, 'save_folder' : 'left_arm_eef', 'save_file': 'left_arm_eef.pkl'},
         '/arm/right_mobile/obs_eef_pose': {'msg_type': PoseStamped, 'save_folder' : 'right_arm_eef', 'save_file': 'right_arm_eef.pkl'},
         '/arm/right_franka/obs_eef_pose': {'msg_type': PoseStamped, 'save_folder' : 'right_arm_eef', 'save_file': 'right_arm_eef.pkl'},
         '/zed_ts': {'msg_type': StampedInt, 'save_folder' : 'zed', 'save_file': 'zed_ts.pkl'},
         'timesteps': {'msg_type': StampedInt, 'save_folder' : 'timesteps', 'save_file': 'timesteps.txt'},
         '/ticker': {'msg_type': PoseStamped, 'save_folder': None, 'save_file': None},
      }

def ros2_time_to_ns(ros2_time):
   return int(ros2_time.sec * 1e9 + ros2_time.nanosec)

class ImageSaver:
   def __init__(self):
      # Create a thread-safe queue for image data (file_path, image)
      self.queue = queue.Queue()
      # Start the background thread as a daemon so it shuts down with your main program
      self.thread = threading.Thread(target=self.process_queue, daemon=True)
      self.thread.start()

   def process_queue(self):
      # Continuously process the queue
      while True:
         # Get image data from the queue; blocks until an item is available
         file_path, image = self.queue.get()
         try:
               # Perform the disk I/O (saving the image)
               cv2.imwrite(file_path, image)
         except Exception as e:
               print(f"Error saving image to {file_path}: {e}")
         # Signal that this queue item has been fully processed
         self.queue.task_done()

   def enqueue_image(self, file_path, image):
      # Push the file path and image into the queue
      self.queue.put((file_path, image))

class DataSync(Node):
   def __init__(self):
      super().__init__('data_sync_node') 
      
      self.declare_parameter('rosbag_collect', False)
      self.rosbag = self.get_parameter('rosbag_collect').value
      
      self.declare_parameter("collect_data_mode", False)
      self.collect_data_mode = self.get_parameter('collect_data_mode').value
      
      self.declare_parameter('topics', [""])
      self.topics = self.get_parameter('topics').value
      
      self.topics_dict = TOPICS

      self.start_recording = False
      self.last_press = None
      
      self.bridge = CvBridge()
      self.image_saver = ImageSaver()
      
      if self.rosbag:
         self.output_dir = os.path.expanduser("~/dexwild/bags/")
      else:
         self.output_dir = os.path.expanduser("~/dexwild/data/")
         self.og_output_dir = self.output_dir
      
      if not os.path.exists(self.output_dir):
         os.makedirs(self.output_dir)
      
      listener = keyboard.Listener(on_press=self.on_press_key)
      listener.start()
      
      self.subscribers = []
      self.sync_publishers = []
      
      self.safe_pub = self.create_publisher(Bool, '/is_safe', 10)
      
      for topic in self.topics:
         self.get_logger().info(f"Subscribing to {topic}")
         msg_type = self.topics_dict[topic]['msg_type']
         self.subscribers.append(Subscriber(self, msg_type, topic))
         self.sync_publishers.append(self.create_publisher(msg_type, f'/sync{topic}', 10))
      
      if ('/zed_ts' in self.topics):
         self.zed_start_recording_pub = self.create_publisher(Bool, '/zed_recording', 10)
         self.curr_ep_pub = self.create_publisher(StampedInt, '/curr_ep', 10)
         self.camera_check_subscriber = self.create_subscription(Image, '/zed/im', self.camcheck_callback, 10)
         self.zed_record = False
      else:
         self.zed_record = False
         self.zed_start_recording_pub = None

      self.sync = ApproximateTimeSynchronizer(self.subscribers,
         queue_size=30, 
         slop=0.1,
         )

      self.timer = self.create_timer(1/5, self.timer_callback)
      self.sync.registerCallback(self.sync_callback)
      
      self.episode_counter = 0
      self.start_zed_every = 5 # start zed recording every 5 episodes
      
      self.prev_time = None
      self.steps = 0
      self.hertz = 0
      self.prev_hertz = 1
      self.total_time = 0
      self.msg_count = 0
      self.time_ns = None

      self.check_camera = True
      self.already_deleted = False
      
      self.UI = RecordUI()
      self.cam_img = np.zeros((320, 240, 3), dtype=np.uint8)
   
   def timer_callback(self):
      if self.steps % 10 == 0:
         if self.prev_time is None:
            self.get_logger().warn(f"[WARNING]: Sync not working yet!!!")
            # check which topics are not being published
         elif self.hertz == self.prev_hertz:
            self.get_logger().warn(f"[WARNING]: Sync Failed!!!")
         else:
            self.get_logger().info(f"Sync Hertz: {self.hertz} | | Synced time: {self.time_ns}")

      if self.collect_data_mode and self.prev_time is not None:
         self.UI.hertz = self.hertz
         self.UI.zed_is_recording = self.zed_record
         self.UI.start_recording = self.start_recording
         self.UI.episode_counter = self.episode_counter
         self.check_camera = self.UI.check_camera
         if self.UI.delete_last_episode:
            self.delete_last_episode()
            self.UI.delete_last_episode = False
         
         self.UI.cam_img = self.cam_img
         self.UI.update_camera()
         self.UI.update_ui()
         self.UI.root.update()
      
      self.prev_hertz = self.hertz
      self.steps += 1
      
   def sync_callback(self, *args):
      current_time = time.time()
      if self.prev_time is not None:
            time_diff = current_time - self.prev_time
            self.total_time += time_diff
            self.msg_count += 1
            average_rate = self.msg_count / self.total_time
            self.hertz = average_rate
      self.prev_time = current_time

      oldest_msg = min(args, key=lambda msg: ros2_time_to_ns(msg.header.stamp)) # use the earliest message
      
      self.synced_time = oldest_msg.header.stamp
      self.time_ns = ros2_time_to_ns(self.synced_time)
      
      for idx, pub in enumerate(self.sync_publishers):  
         msg_copy = args[idx]
         msg_copy.header.stamp = self.synced_time
         pub.publish(msg_copy)
         topic_name = self.subscribers[idx].topic_name
         # self.get_logger().info(f"Publishing {topic_name} at time {self.time_ns}")
         if self.start_recording and not self.rosbag and topic_name in self.data_log.keys():
            self.process_message(topic_name, msg_copy)
            
      if self.start_recording and not self.rosbag:
         self.data_log['timesteps'].append(int(self.time_ns))
   
   def camcheck_callback(self, msg):
      if self.check_camera:
         # Convert the ROS Image to an OpenCV image
         cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
         self.cam_img = cv_image
   
   def process_message(self, topic_name, msg_copy):
      if isinstance(msg_copy, JointState):
         joint_positions = np.array(msg_copy.position)
         stamped = np.concatenate((np.array([self.time_ns]).astype(int), joint_positions))
         self.data_log[topic_name].append(stamped)
      elif isinstance(msg_copy, PoseArray):
         poses = msg_copy.poses
         pose_arr = []
         for pose in poses:
            position = np.array([pose.position.x, pose.position.y, pose.position.z])
            orientation = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
            combined = np.concatenate((position, orientation))
            pose_arr.append(combined)
         stamped = np.concatenate((np.array([self.time_ns]).astype(int), np.array(pose_arr).reshape(-1))) #make the pose array flat
         self.data_log[topic_name].append(stamped)
      elif isinstance(msg_copy, Image):
         cv_image = self.bridge.imgmsg_to_cv2(msg_copy, desired_encoding="bgr8")
         if topic_name == '/left/thumb_camera_im':
            file_path = f"{self.output_dir}/left_thumb_cam/{self.time_ns}.jpg"
         elif topic_name == '/left/pinky_camera_im':
            file_path = f"{self.output_dir}/left_pinky_cam/{self.time_ns}.jpg"
         elif topic_name == '/right/thumb_camera_im':  
            file_path = f"{self.output_dir}/right_thumb_cam/{self.time_ns}.jpg"
         elif topic_name == '/right/pinky_camera_im':
            file_path = f"{self.output_dir}/right_pinky_cam/{self.time_ns}.jpg"
         elif topic_name == '/zed/im':
            file_path = f"{self.output_dir}/zed_obs/{self.time_ns}.jpg"
         elif topic_name == '/head/camera_im':
            file_path = f"{self.output_dir}/head_cam/{self.time_ns}.jpg"
         else:
            raise ValueError(f"Unsupported topic {topic_name}")
         self.image_saver.enqueue_image(file_path, cv_image)
      elif isinstance(msg_copy, PoseStamped):
         position = np.array([msg_copy.pose.position.x, msg_copy.pose.position.y, msg_copy.pose.position.z])
         orientation = np.array([msg_copy.pose.orientation.x, msg_copy.pose.orientation.y, msg_copy.pose.orientation.z, msg_copy.pose.orientation.w])
         combined = np.concatenate((position, orientation))
         stamped = np.concatenate((np.array([self.time_ns]).astype(int), combined))
         self.data_log[topic_name].append(stamped)
      elif isinstance(msg_copy, StampedInt):
         camera_ts = msg_copy.data
         stamped =  np.concatenate((np.array([self.time_ns]).astype(int), np.array([camera_ts]).astype(int)))
         self.data_log[topic_name].append(stamped)
      else:
         raise ValueError(f"Unsupported message type {type(msg_copy)}")
   
   def delete_last_episode(self):
      if self.episode_counter > 0 and not self.start_recording and not self.rosbag and not self.already_deleted:
         assert self.output_dir == self.og_output_dir, "Output directory is not the original directory"
         
         self.episode_counter -= 1
         all_dirs = os.listdir(self.output_dir)
         all_dirs = [d for d in all_dirs if 'ep' in d]
         last_dir = max([int(d.split('_')[-1]) for d in all_dirs])
         self.deleted_dir = os.path.join(self.output_dir, f'ep_{last_dir}')
         self.get_logger().info(f"Deleting episode {last_dir}")
         shutil.rmtree(self.deleted_dir)   
         self.already_deleted = True
         
         zed_recording_path = os.path.join(self.output_dir, 'zed_recordings')
         
         if self.episode_counter % self.start_zed_every == 0 and self.zed_start_recording_pub is not None and self.zed_record:
            self.zed_record = False
            msg = Bool()
            msg.data = self.zed_record
            self.zed_start_recording_pub.publish(msg)
            # delete the zed recording
            recording = f"output_{last_dir}_{last_dir}.svo2"
            
            while not os.path.exists(os.path.join(zed_recording_path, recording)):
               time.sleep(0.1)
               self.get_logger().info(f"Waiting for {recording} to be saved so it can be deleted")

            os.remove(os.path.join(zed_recording_path, recording))
         
         if self.episode_counter % self.start_zed_every != 0 and self.zed_start_recording_pub is not None and not self.zed_record:
            self.zed_record = True
            msg = Bool()
            msg.data = self.zed_record
            self.zed_start_recording_pub.publish(msg)
            recordings = os.listdir(zed_recording_path)
            # find the recording that corresponds to the last episode
            recordings = [r for r in recordings if str(last_dir) in r]
            assert len(recordings) == 1, f"More than one recording found for episode {last_dir}"
            
            recording = recordings[0]
            changed_name = recording.replace(f"_{last_dir}", f"_{last_dir - 1}")
            
            os.rename(os.path.join(zed_recording_path, recording), os.path.join(zed_recording_path, changed_name))
            
            
   def on_press_key(self, key):
      """Callback function for key press events."""
      try:
         if hasattr(key, 'char') and key.char == 'q':
            self.get_logger().info(f"Shutting down")
            self.safe_pub.publish(Bool(data=False))
            
         if self.collect_data_mode and ((hasattr(key, 'char') and key.char == 's')):
            if self.zed_start_recording_pub is not None:
               self.zed_record = not self.zed_record
               msg = Bool()
               msg.data = self.zed_record
               self.zed_start_recording_pub.publish(msg)
         
         if self.collect_data_mode and ((hasattr(key, 'char') and key.char == 'g')):
            # delete the last episode
            self.delete_last_episode()
            
         if self.collect_data_mode and ((hasattr(key, 'char') and key.char == 'r') or (hasattr(key, 'char') and key.char == "b") or key == keyboard.Key.page_up or key == keyboard.Key.up):
            if not os.path.exists(self.output_dir):
               raise FileNotFoundError(f"Output directory {self.output_dir} does not exist")
            
            # prevent spamming
            if self.last_press == None:
               self.last_press = time.time()
            elif time.time() - self.last_press < 0.5:
               return
            else:
               self.last_press = time.time()
            
            if not self.rosbag:
               if not self.start_recording:
                  self.start_step = self.steps
                  all_dirs = os.listdir(self.output_dir)
                  all_dirs = [d for d in all_dirs if 'ep' in d]
                  if all_dirs == []:
                     last_dir = -1 # no directories
                  else:
                     last_dir = max([int(d.split('_')[-1]) for d in all_dirs])
                     
                  self.output_dir = os.path.join(self.output_dir, f'ep_{last_dir + 1}')
                  
                  self.get_logger().info(f"Output directory: {self.output_dir}") 
                  
                  if self.zed_start_recording_pub is not None and self.episode_counter % self.start_zed_every == 0:
                     assert not self.zed_record, "ZED recording already started"
                     self.zed_record = True
                     msg = Bool()
                     msg.data = self.zed_record
                     self.zed_start_recording_pub.publish(msg)
                     time.sleep(1) # wait for zed to start recording
                  
                  # publish current episode
                  if self.zed_record:
                     msg = StampedInt()
                     msg.data = last_dir + 1
                     self.curr_ep_pub.publish(msg)
                     
                  self.get_logger().info(f"Starting data recording")
                  self.data_log = {key: [] for key in self.topics}
                  
                  self.data_log['timesteps'] = [] # initialize data log for timestamps
                  logged_topics = self.data_log.keys()
                  self.get_logger().info(f"{logged_topics}")
                  
                  # create directories
                  for key in self.data_log.keys():
                     folder = self.topics_dict[key]["save_folder"]
                     if folder is not None:
                        path = os.path.join(self.output_dir, folder)
                        os.makedirs(path, exist_ok=True)
                     
                  self.already_deleted = False
               else:
                  self.get_logger().info(f"Stopping data recording") 
                  self.episode_counter += 1
                  
                  for key in self.data_log.keys():
                     file = self.topics_dict[key]["save_file"]
                     if file is not None:
                        file_ext = file.split('.')[-1]
                        path = os.path.join(self.output_dir, self.topics_dict[key]["save_folder"], file)
                        data = np.array(self.data_log[key])
                        if file_ext == 'pkl':
                           # self.get_logger().info(f"Saving {key} to {path}")
                           save_pickle(data, path)
                        elif file_ext == 'txt':
                           np.savetxt(path, data.astype(int), fmt='%d')
                     else:
                        self.get_logger().info(f"{key} does not have a save file")
         
                  self.get_logger().info(f"Data saved to {self.output_dir}")
                  self.output_dir = self.og_output_dir
                  
                  if self.zed_start_recording_pub is not None and self.episode_counter % self.start_zed_every == 0:
                     assert self.zed_record, "ZED recording is not going"
                     time.sleep(1) # wait for zed to stop recording
                     self.zed_record = False
                     msg = Bool()
                     msg.data = self.zed_record
                     self.zed_start_recording_pub.publish(msg)
                     
            else:
               assert(self.rosbag)
               now = datetime.datetime.now().isoformat(timespec='seconds')
               if not self.start_recording:
                  self.get_logger().info(f"Starting ROSBAG recording")
                  self.rosbag_process = subprocess.Popen(
                     ['ros2', 'bag', 'record', "-o", os.path.join(self.output_dir, str(now) + '_bag')] + self.topics ,
                     stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE
                  )
               elif self.start_recording:
                  self.get_logger().info(f"Stopping ROSBAG recording")
                  self.rosbag_process.terminate()
                  self.rosbag_process.wait()
            
            self.start_recording = not self.start_recording
               
      except AttributeError:
            pass


def main(args=None):
      rclpy.init(args=args)
      data_sync_node = DataSync()
      try:
         rclpy.spin(data_sync_node)
      except KeyboardInterrupt:
         pass
      finally:
         data_sync_node.destroy_node()
         rclpy.shutdown()