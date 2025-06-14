# Copyright (c) 2025 Tony Tao
# Licensed under the MIT License. See LICENSE file in project root for details.
"""
deploy_policy_node.py

This script defines the `DeployPolicy` ROS 2 node, responsible for executing
learned robot manipulation policies in real-time using visual and proprioceptive inputs.

Key Features:
- Supports single-arm and bimanual robotic setups with modular sensor configurations.
- Performs observation buffering, preprocessing, and policy inference at a configurable frequency.
- Interfaces with both standard ROS publishers and RMP (Riemannian Motion Policies) controllers for trajectory execution.
- Handles action smoothing (EMA), policy modes (absolute, relative, hybrid), and warm-start initialization.
- Supports both replay-based and transformer-based policy inference (e.g., DiTInference).
- Manages hand and end-effector command publication using learned targets.
"""

import os
import time
from collections import deque
from termcolor import colored

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from sensor_msgs.msg import JointState, Image
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

from cv_bridge import CvBridge

from dexwild_utils.zmq_utils import ZMQPublisher
from dexwild_utils.pose_utils import (
    pose6d_to_pose7d, pose7d_to_pose6d,
    pose6d_to_mat, mat_to_pose7d,
    to_intergripper_pose
)
from dexwild_utils.PolicyInference import ReplayPolicy, DiTInference

# Precision for printing
np.set_printoptions(precision=6, suppress=True)

np.set_printoptions(precision=6, suppress=True)

class DeployPolicy(Node):   
    def __init__(self):
        super().__init__('DeployPolicy')
        self.bridge = CvBridge()
        myIP = "127.0.0.1"  # used in some publishers

        self._declare_parameters()
        self._load_parameters()

        self._init_state_fields()
        self._process_nominal_poses()

        self._init_prev_action()

        self.tf_br = TransformBroadcaster(self)
        self._setup_publishers_and_subscribers(myIP)

        self._init_policy()

        self.create_timer(1/self.freq, self.inference_step)
        self.create_timer(1/120, self.pub_messages)

        self.get_logger().info(colored(f"Checkpoint path: {self.checkpoint_path}", "green"))
        self.get_logger().info(colored(f"Using Ensemble: {self.isensemble}", "green"))
        self.get_logger().info(colored(f"Mode: {self.mode}", "green"))

    def _declare_parameters(self):
        p = self.declare_parameter
        p('checkpoint_path', "")
        p('id', 'left_mobile')
        p('mode', '')
        p('isensemble', True)
        p("buffer_size", 15)
        p("replay_path", "")
        p("openloop_length", 30)
        p("pred_horizon", 24)
        p("exp_weight", 0.4)
        p("start_poses", None)
        p("start_hand_poses", None)
        p("replay_id", 0)
        p("freq", 30)
        p("ema_amount", 0.0)
        p("use_rmp", False)
        p("rot_repr", "euler")
        p("skip_first_actions", 0)
        p("observation_keys", [""])

    def _load_parameters(self):
        g = self.get_parameter
        self.checkpoint_path = g('checkpoint_path').value
        self.id              = g('id').value
        self.mode            = g('mode').value
        self.isensemble      = g('isensemble').value
        self.buffer_size     = g('buffer_size').value
        self.replay_path     = g('replay_path').value
        self.openloop_length = g('openloop_length').value
        self.pred_horizon    = g('pred_horizon').value
        self.exp_weight      = g('exp_weight').value
        self.ema_alpha       = 1 - g('ema_amount').value
        self.use_rmp         = g('use_rmp').value
        self.rot_repr        = g('rot_repr').value
        self.nominal_eef_poses  = g('start_poses').value
        self.nominal_hand_poses = g('start_hand_poses').value
        self.skip_first_actions = g('skip_first_actions').value
        self.observation_keys = g('observation_keys').value
        self.freq               = g('freq').value

    def _init_state_fields(self):
        self.ema_action = None
        self.first_poses, self.first_hands = None, None
        self.curr_target_hand_joint = None
        self.curr_target_pose       = None
        self.prev_time = None
        self.start = False
        self.step = 0
        
        # buffers
        self.obs_buffer = {}
        for key in self.observation_keys:
            self.obs_buffer[key] = deque(maxlen=self.buffer_size)
            
            
        self.curr_pose = {}
        # index ranges for hand vs eef
        if "leapv1" in self.id:
            self.hand_idx, self.eef_idx = slice(0,16), slice(16,22)
        else:
            self.hand_idx, self.eef_idx = slice(0,17), slice(17,23)
            self.eef_pos = slice(17,20)

    def _process_nominal_poses(self):
        """Turn flat start_poses into [right],[left] (or 2× halves for bimanual)."""
        def split_if_bimanual(arr):
            if arr is None: return None
            arr = np.array(arr)
            if "bimanual" in self.id:
                mid = len(arr)//2
                return np.array([arr[:mid], arr[mid:]])
            else:
                return np.array([arr])
        self.nominal_eef_poses  = split_if_bimanual(self.nominal_eef_poses)
        self.nominal_hand_poses = split_if_bimanual(self.nominal_hand_poses)
        
        

    def _init_prev_action(self):
        """Seed prev_action list from nominal poses."""
        self.prev_action = [[], []]
        if self.nominal_hand_poses is None and self.nominal_eef_poses is None:
            self.prev_action = None
            return

        def add(hand_arr, eef_arr):
            if hand_arr is not None:
                self.prev_action[0].append(hand_arr)
            if eef_arr is not None:
                self.prev_action[1].append(pose6d_to_pose7d(eef_arr))

        if "bimanual" in self.id:
            for i in range(2):
                add(self.nominal_hand_poses[i] if self.nominal_hand_poses is not None else None,
                    self.nominal_eef_poses[i]  if self.nominal_eef_poses  is not None else None)
        else:
            add(self.nominal_hand_poses[0] if self.nominal_hand_poses is not None else None,
                self.nominal_eef_poses[0]  if self.nominal_eef_poses  is not None else None)

    def _setup_publishers_and_subscribers(self, myIP):
        """Setup all the ROS pubs/subs and internal buffers per `id`."""
        # always publish policy_start
        self.policy_start_pub = self.create_publisher(Bool, '/policy_start', 10)
        self.policy_start_pub.publish(Bool(data=False))

        # now do each case
        if self.id == 'bimanual':
            self._setup_bimanual(myIP)
        elif self.id.startswith('left'):
            self._setup_single_arm('left', myIP)
        elif self.id.startswith('right'):
            self._setup_single_arm('right', myIP)

    def _setup_bimanual(self, myIP):
        left_id, right_id = 'left_mobile','right_mobile'
        # subscriptions
        self.right_arm_sub = self.create_subscription(PoseStamped, f'/sync/arm/{right_id}/obs_eef_pose', self.right_pose_callback, 10)
        self.curr_right_arm_sub = self.create_subscription(PoseStamped, f'/arm/{right_id}/obs_eef_pose', self.curr_right_pose_callback, 10)
        self.right_thumb_cam_sub = self.create_subscription(Image, '/sync/right/thumb_camera_im', self.right_thumb_cam_callback, 10)
        self.right_pinky_cam_sub = self.create_subscription(Image, '/sync/right/pinky_camera_im', self.right_pinky_cam_callback, 10)
        self.left_arm_sub = self.create_subscription(PoseStamped, f'/sync/arm/{left_id}/obs_eef_pose', self.left_pose_callback, 10)
        self.curr_left_arm_sub = self.create_subscription(PoseStamped, f'/arm/{left_id}/obs_eef_pose', self.curr_left_pose_callback, 10)
        self.left_thumb_cam_sub = self.create_subscription(Image, '/sync/left/thumb_camera_im', self.left_thumb_cam_callback, 10)
        self.left_pinky_cam_sub = self.create_subscription(Image, '/sync/left/pinky_camera_im', self.left_pinky_cam_callback, 10)
        self.zed_sub = self.create_subscription(Image, '/sync/zed/im', self.zed_callback, 10)

        # publishers (RMP vs normal)
        if self.use_rmp:
            self.right_eef_publisher = ZMQPublisher(f"tcp://{myIP}:4098")
            self.left_eef_publisher  = ZMQPublisher(f"tcp://{myIP}:5098")
        else:
            self.right_pose_pub = self.create_publisher(PoseStamped, f'/arm/{right_id}/cmd_eef_pose', 10)
            self.left_pose_pub  = self.create_publisher(PoseStamped, f'/arm/{left_id}/cmd_eef_pose', 10)

        self.right_hand_pub = self.create_publisher(JointState, '/leapv2_node/cmd_raw_leap_r', 10)
        self.left_hand_pub  = self.create_publisher(JointState, '/leapv2_node/cmd_raw_leap_l', 10)

    def _setup_single_arm(self, side, myIP):
        # side = 'left' or 'right'
        if "franka" in self.id:
            arm_id = f'{side}_franka'
        else:
            arm_id = f'{side}_mobile'
            
        self.create_subscription(PoseStamped, f'/sync/arm/{arm_id}/obs_eef_pose', getattr(self, f'{side}_pose_callback'), 10) # for observation, synchronized from synced topic
        self.create_subscription(PoseStamped, f'/arm/{arm_id}/obs_eef_pose', getattr(self, f'curr_{side}_pose_callback'), 10) # most up to date current pose (not from synchronization)
        
        self.zed_sub = self.create_subscription(Image, '/sync/zed/im', self.zed_callback, 10)
        self.create_subscription(Image, f'/sync/{side}/thumb_camera_im', getattr(self, f'{side}_thumb_cam_callback'), 10)
        self.create_subscription(Image, f'/sync/{side}/pinky_camera_im', getattr(self, f'{side}_pinky_cam_callback'), 10)
        
        # buffer
        for key in self.observation_keys:
            self.obs_buffer[key] = deque(maxlen=self.buffer_size)

        # hand pub
        if "leapv1" in self.id:
            if side == 'left':
                self.left_hand_pub = self.create_publisher(JointState, '/leaphand_node/cmd_leap_l', 10)
            else:
                self.right_hand_pub = self.create_publisher(JointState, '/leaphand_node/cmd_leap_r', 10)
        else:
            if side == 'left':
                self.left_hand_pub = self.create_publisher(JointState, '/leapv2_node/cmd_raw_leap_l', 10)
            else:
                self.right_hand_pub = self.create_publisher(JointState, '/leapv2_node/cmd_raw_leap_r', 10)

        # eef pub
        if self.use_rmp:
            port = 4098 if side=='right' else 5098
            if side == 'left':
                self.left_eef_publisher = ZMQPublisher(f"tcp://{myIP}:{port}")
            else:
                self.right_eef_publisher = ZMQPublisher(f"tcp://{myIP}:{port}")
        else:
            if side == 'left':
                self.left_pose_pub = self.create_publisher(PoseStamped, f'/arm/{arm_id}/cmd_eef_pose', 10)
            else:
                self.right_pose_pub = self.create_publisher(PoseStamped, f'/arm/{arm_id}/cmd_eef_pose', 10)

    def _init_policy(self):
        self.checkpoint_folder = os.path.dirname(self.checkpoint_path)
        model_name = os.path.basename(self.checkpoint_path)
        if 'replay' in self.mode:
            self.policy = ReplayPolicy(
                buffer_path=self.replay_path,
                id=self.id, mode=self.mode,
                normalized=False,
                isensemble=self.isensemble,
                replay_id=self.get_parameter("replay_id").value,
                rot_repr=self.rot_repr
            )
        else:
            self.policy = DiTInference(
                self.checkpoint_folder, model_name,
                isensemble=self.isensemble, id=self.id,
                mode=self.mode,
                openloop_length=self.openloop_length,
                pred_horizon=self.pred_horizon,
                exp_weight=self.exp_weight,
                rot_repr=self.rot_repr,
                buffer_size=self.buffer_size,
                skip_first_actions=self.skip_first_actions
            )
            longest_history_len = max(
                max(self.policy.img_hist_frames),
                max(self.policy.state_hist_frames)
            )
            assert self.buffer_size > longest_history_len, \
                "Buffer size must be greater than the history length used in the policy"
        
    def model_forward(self, obs):
        action = self.policy.forward(obs)
        
        # NO MORE ACTIONS
        if action is None:
            return None

        if "bimanual" in self.id:
            half_point = action.shape[0] // 2
            action = np.array([action[:half_point], action[half_point:]]) # split into right and left for bimanual
            
        return action

    def get_obs(self):
        # make a copy of obs_buffer that converts the deque to list
        obs_buffer = {key: list(value) for key, value in self.obs_buffer.items()}
        
        #NOTE always use convention # [right, left]
        def get_or_pad(key):
            return obs_buffer[key] if key in obs_buffer else [None] * self.buffer_size

        zed_hist    = get_or_pad("zed")
        if self.id == 'bimanual':
            pose_hist   = [get_or_pad("right_pose"), get_or_pad("left_pose")]
            thumb_cam_hist  = [get_or_pad("right_thumb_cam"), get_or_pad("left_thumb_cam")]
            pinky_cam_hist  = [get_or_pad("right_pinky_cam"), get_or_pad("left_pinky_cam")]
            hand_hist   = [get_or_pad("right_hand"), get_or_pad("left_hand")]
        else:
            side = 'left' if 'left' in self.id else 'right'
            pose_hist   = get_or_pad(f"{side}_pose")
            thumb_cam_hist  = get_or_pad(f"{side}_thumb_cam")
            pinky_cam_hist  = get_or_pad(f"{side}_pinky_cam")
            hand_hist   = get_or_pad(f"{side}_hand")
            
        # convert to numpy arrays
        zed_hist = np.array(zed_hist)
        pose_hist = np.array(pose_hist)
        thumb_cam_hist = np.array(thumb_cam_hist)
        pinky_cam_hist = np.array(pinky_cam_hist)
        hand_hist = np.array(hand_hist)
        
        # get the intergripper pose
        intergripper_pose_hist =  np.array(to_intergripper_pose(pose_hist[0], pose_hist[1])) if self.id == "bimanual" else None

        # convert to obs dict for the policy
        obs = {"hand": {}, "pose": {}, "zed": {}, "images": {}}
        
        obs["hand"] = hand_hist
        obs["pose"] = pose_hist
        
        if self.id == 'bimanual':
            obs["intergripper"] = intergripper_pose_hist
            if self.policy.ncams == 4:
                #print the shapes
                # right hand cameras first
                obs["images"]["cam0"] = pinky_cam_hist[0]
                obs["images"]["cam1"] = thumb_cam_hist[0]
                # left hand cameras second
                obs["images"]["cam2"] = pinky_cam_hist[1]
                obs["images"]["cam3"] = thumb_cam_hist[1]
            elif self.policy.ncams == 2:
                obs["images"]["cam1"] = thumb_cam_hist[0]
                obs["images"]["cam3"] = thumb_cam_hist[1]
            elif self.policy.ncams == 3:
                obs["images"]["cam1"] = thumb_cam_hist[0]
                obs["images"]["cam3"] = thumb_cam_hist[1]
                obs["images"]["cam4"] = zed_hist
            elif self.policy.ncams == 1:
                obs["images"]["cam4"] = zed_hist
        else:
            if self.policy.ncams == 1:
                obs["images"]["cam0"] = thumb_cam_hist
            elif self.policy.ncams == 2:
                obs["images"]["cam0"] = pinky_cam_hist
                obs["images"]["cam1"] = thumb_cam_hist
            elif self.policy.ncams == 3:
                raise NotImplementedError
            
        return obs

    def inference_step(self):
        is_bimanual = (self.id == 'bimanual')
        mode = self.mode
        
        if self.start:
            # --- Policy execution branch ---
            # 1) Signal policy start
            self.policy_start_pub.publish(Bool(data=True))

            # 2) Gather observations and current state
            obs = self.get_obs()
            
            if is_bimanual:
                current_hand = np.array([self.obs_buffer['right_hand'][-1],self.obs_buffer['left_hand'][-1]])
                current_pose = np.array([self.curr_pose['right_arm_eef'],self.curr_pose['left_arm_eef']])
            else:
                side = 'left' if 'left' in self.id else 'right'
                current_hand = self.obs_buffer[f'{side}_hand'][-1]
                current_pose = self.curr_pose[f'{side}_arm_eef']

            # 3) Inference and timing
            now = time.time()
            action = self.model_forward(obs)
            if self.prev_time is not None and self.step % 10 == 0:
                self.get_logger().info(f"Inference Time: {now - self.prev_time}")
            self.prev_time = now

            if self.ema_action is None:
                # If this is the first action, initialize EMA with it.
                if "bimanual" in self.id:
                    self.ema_action = np.array([action[0, self.eef_pos], action[1, self.eef_pos]])
                else:
                    self.ema_action = action.copy()[self.eef_pos]
            else:
                if "bimanual" in self.id:
                    # Update EMA for both hands
                    self.ema_action[0] = self.ema_alpha * action[0, self.eef_pos] + (1 - self.ema_alpha) * self.ema_action[0]
                    self.ema_action[1] = self.ema_alpha * action[1, self.eef_pos] + (1 - self.ema_alpha) * self.ema_action[1]
                else:
                    # Otherwise, update EMA: smoothed = alpha * new + (1 - alpha) * previous_smoothed
                    self.ema_action = self.ema_alpha * action[self.eef_pos] + (1 - self.ema_alpha) * self.ema_action

            # apply smoothed action
            if is_bimanual:
                action[0, self.eef_pos] = self.ema_action[0]
                action[1, self.eef_pos] = self.ema_action[1]
            else:
                action[self.eef_pos] = self.ema_action.copy()

            # 5) Compute targets based on mode
            # Prepare containers
            target_hand = None
            target_pose = None

            if mode in ('replay_rel', 'replay_rel_abs'):
                # set initial reference if needed
                if self.first_poses is None and self.first_hands is None:
                    self.first_poses = current_pose
                    self.first_hands = current_hand

                if is_bimanual:
                    target_hand = [
                        action[0, self.hand_idx],
                        action[1, self.hand_idx]
                    ]
                    # apply relative transforms
                    mats0 = pose6d_to_mat(self.first_poses[0])
                    mats1 = pose6d_to_mat(self.first_poses[1])
                    targ0 = mats0 @ pose6d_to_mat(action[0, self.eef_idx])
                    targ1 = mats1 @ pose6d_to_mat(action[1, self.eef_idx])
                    target_pose = [
                        mat_to_pose7d(targ0),
                        mat_to_pose7d(targ1)
                    ]
                else:
                    target_hand = action[self.hand_idx]
                    first_mat = pose6d_to_mat(self.first_poses)
                    targ_mat = first_mat @ pose6d_to_mat(action[self.eef_idx])
                    target_pose = mat_to_pose7d(targ_mat)

            elif mode in ('abs', 'replay_abs', 'rel_mixed', 'hybrid'):
                # absolute target modes
                if is_bimanual:
                    target_hand = [
                        action[0, self.hand_idx],
                        action[1, self.hand_idx]
                    ]
                    target_pose = [
                        pose6d_to_pose7d(action[0, self.eef_idx]),
                        pose6d_to_pose7d(action[1, self.eef_idx])
                    ]
                else:
                    target_hand = [action[self.hand_idx]]
                    target_pose = [pose6d_to_pose7d(action[self.eef_idx])]

            # remember last for replay_abs etc.
            self.prev_action = [target_hand, target_pose]

            # 6) Update shared state and increment
            self.curr_target_hand_joint = target_hand
            self.curr_target_pose = target_pose
            self.step += 1

        else:
            # --- Waiting-for-buffer branch ---
            self.policy_start_pub.publish(Bool(data=False))

            # Use nominal starting targets
            if is_bimanual:
                self.curr_target_hand_joint = self.nominal_hand_poses
                self.curr_target_pose = [pose6d_to_pose7d(self.nominal_eef_poses[0]), pose6d_to_pose7d(self.nominal_eef_poses[1])]
            else:
                self.curr_target_hand_joint = self.nominal_hand_poses
                self.curr_target_pose = [pose6d_to_pose7d(self.nominal_eef_poses[0])]

            # log buffer status
            self.get_logger().info('Waiting for buffer to fill')
            for key, dq in self.obs_buffer.items():
                self.get_logger().info(f"{key}: {len(dq)}")

            # check fullness
            full_keys = all(len(v) == self.buffer_size for v in self.obs_buffer.values())
            # minimal for certain arms
            if self.id.startswith('left'):
                minimal = (len(self.obs_buffer['left_pose']) == self.buffer_size and
                           len(self.obs_buffer['left_hand']) == self.buffer_size)
            elif self.id.startswith('right'):
                minimal = (len(self.obs_buffer['right_pose']) == self.buffer_size and
                           len(self.obs_buffer['right_hand']) == self.buffer_size)
            elif is_bimanual:
                minimal = full_keys
            else:
                minimal = False  # fallback

            # if buffers ok, check nominal pose proximity
            nominal_met = False
            if minimal or full_keys:
                if is_bimanual:
                    curr_h = np.array([self.obs_buffer['right_hand'][-1], self.obs_buffer['left_hand'][-1]])
                    curr_p = np.array([self.curr_pose['right_arm_eef'], self.curr_pose['left_arm_eef']])
                    
                    hand_distance = np.linalg.norm(curr_h.flatten() - self.nominal_hand_poses.flatten())
                    eef_distance = np.linalg.norm(curr_p[:, :3].flatten() - self.nominal_eef_poses[:, :3].flatten())
                    
                else:
                    side = 'left' if 'left' in self.id else 'right'
                    curr_h = self.obs_buffer[f'{side}_hand'][-1]
                    curr_p = self.curr_pose[f'{side}_arm_eef']
                    
                    hand_distance = np.linalg.norm(curr_h - self.nominal_hand_poses[0]) 
                    eef_distance = np.linalg.norm(curr_p[:3] - self.nominal_eef_poses[0][:3])
                    
                ok_h = (self.nominal_hand_poses is None or hand_distance < 23)
                ok_p = (self.nominal_eef_poses is None or eef_distance < 0.1)
                
                nominal_met = (ok_h and ok_p)
                    
                if self.nominal_eef_poses is not None:
                    self.get_logger().info(
                        f"eef distance {eef_distance:.3f}"
                    )
                if self.nominal_hand_poses is not None:
                    self.get_logger().info(
                        f"hand distance {hand_distance}"
                    )
                    
                time.sleep(1)

            # decide start condition
            if 'replay' in mode:
                self.start = minimal and nominal_met
            else:
                self.start = full_keys and nominal_met
                self.get_logger().info(
                    f"Buffer Full: {full_keys}, Nominal Pose Met: {nominal_met}"
                )
       
    def pub_messages(self):
        """
        Publish the current target pose and hand joint commands.
        Supports both bimanual and single-arm modes, and RMP vs ROS topics.
        """
        target_pose = self.curr_target_pose
        target_hand = self.curr_target_hand_joint
        
        # get one timestamp for all messages
        now = self.get_clock().now().to_msg()

        def _publish_eef(publisher_rmp, publisher_ros, pose_arr):
            """
            Send end-effector commands via RMP or ROS topic.
            """
            if self.use_rmp:
                if publisher_rmp and pose_arr is not None:
                    publisher_rmp.send_message(pose_arr)
            else:
                if publisher_ros and pose_arr is not None:
                    msg = PoseStamped()
                    msg.header.stamp = now
                    msg.pose.position = Point(x=pose_arr[0], y=pose_arr[1], z=pose_arr[2])
                    msg.pose.orientation = Quaternion(
                        x=pose_arr[3], y=pose_arr[4],
                        z=pose_arr[5], w=pose_arr[6]
                    )
                    publisher_ros.publish(msg)

        def _publish_hand(publisher_ros, hand_arr):
            """
            Send hand joint commands via ROS topic.
            """
            if publisher_ros and hand_arr is not None:
                msg = JointState()
                msg.header.stamp = now
                msg.position = hand_arr.tolist()
                publisher_ros.publish(msg)

        # Bimanual mode: two arms
        if 'bimanual' in self.id:
            # EEF publishers: RMP or ROS
            _publish_eef(
                getattr(self, 'right_eef_publisher', None),
                getattr(self, 'right_pose_pub',     None),
                target_pose[0] if target_pose is not None else None
            )
            _publish_eef(
                getattr(self, 'left_eef_publisher', None),
                getattr(self, 'left_pose_pub',      None),
                target_pose[1] if target_pose is not None else None
            )
            # Hand publishers (always ROS)
            _publish_hand(
                getattr(self, 'right_hand_pub', None),
                target_hand[0] if target_hand is not None else None
            )
            _publish_hand(
                getattr(self, 'left_hand_pub',  None),
                target_hand[1] if target_hand is not None else None
            )
        else:
            # Single-arm mode: pick side
            side = 'left' if 'left' in self.id else 'right'
            # end-effector pubs
            _publish_eef(getattr(self, f"{side}_eef_publisher", None), 
                         getattr(self, f"{side}_pose_pub",      None),
                         target_pose[0] if target_pose is not None else None)
            # hand pub
            _publish_hand(getattr(self, f"{side}_hand_pub", None),
                          target_hand[0] if target_hand is not None else None)

    def zed_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.obs_buffer["zed"].append(cv_image)
    
    def left_thumb_cam_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.obs_buffer["left_thumb_cam"].append(cv_image)
    
    def right_thumb_cam_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.obs_buffer["right_thumb_cam"].append(cv_image)

    def left_pinky_cam_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.obs_buffer["left_pinky_cam"].append(cv_image)
    
    def right_pinky_cam_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.obs_buffer["right_pinky_cam"].append(cv_image)
        
    def left_hand_callback(self):
        # append previous hand position
        if "bimanual" in self.id:
            self.obs_buffer["left_hand"].append(self.prev_action[0][1])
        else:
            self.obs_buffer["left_hand"].append(self.prev_action[0][0])
    
    def right_hand_callback(self):
        # append previous hand position
        if "bimanual" in self.id:
            self.obs_buffer["right_hand"].append(self.prev_action[0][0])
        else:
            self.obs_buffer["right_hand"].append(self.prev_action[0][0])
            
    def curr_right_pose_callback(self, msg):
        pose = np.concatenate((np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]),
                                np.array([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])))
        
        self.curr_pose["right_arm_eef"] = pose7d_to_pose6d(pose)
        self.policy.update_pose(self.curr_pose)
        
    def curr_left_pose_callback(self, msg):
        pose = np.concatenate((np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]),
                                np.array([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])))
        
        self.curr_pose["left_arm_eef"] = pose7d_to_pose6d(pose)
        self.policy.update_pose(self.curr_pose)

    def left_pose_callback(self, msg):
        pose = np.concatenate((np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]), 
                                np.array([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])))
        
        
        self.obs_buffer["left_pose"].append(pose7d_to_pose6d(pose))
        self.left_hand_callback() # update the left hand buffer
        
        if self.id == 'bimanual':
            # intergripper pose
            intergripper_pose_hist = np.array(to_intergripper_pose(np.array([self.obs_buffer["right_pose"][-1]]), np.array([self.obs_buffer["left_pose"][-1]])))

            # publish to transforms
            intergripper_pose_7d = pose6d_to_pose7d(intergripper_pose_hist[-1])
            
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = "right_mobile"
            t.child_frame_id = "intergripper"

            # Set position
            t.transform.translation.x = intergripper_pose_7d[0]
            t.transform.translation.y = intergripper_pose_7d[1]
            t.transform.translation.z = intergripper_pose_7d[2]

            t.transform.rotation.x = intergripper_pose_7d[3]
            t.transform.rotation.y = intergripper_pose_7d[4]
            t.transform.rotation.z = intergripper_pose_7d[5]
            t.transform.rotation.w = intergripper_pose_7d[6]

            # Broadcast the transform
            self.tf_br.sendTransform(t)

    def right_pose_callback(self, msg):
        pose = np.concatenate((np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]), 
                                np.array([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])))


        self.obs_buffer["right_pose"].append(pose7d_to_pose6d(pose))
            
        self.right_hand_callback() #update the right hand buffer

    
    def destroy_node(self):
        # self.policy.save_action_history()
        super().destroy_node()

def main(args=None):
    rclpy.init()
    node = DeployPolicy()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    
if __name__ == '__main__':
    main()