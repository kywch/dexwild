

import argparse
import faulthandler
import glob
import os
import shutil

import cv2
import numpy as np
import dexwild_utils.ogl_viewer.tracking_viewer as gl
import pyzed.sl as sl
from dexwild_utils.aruco_utils import CubeTracker, get_mask
from dexwild_utils.pose_utils import (save_one_path3d)
from dexwild_utils.data_processing import smooth_path
from termcolor import colored
from dexwild_utils.data_processing import left_face_corners_3d, right_face_corners_3d, save_pickle

faulthandler.enable()
#
def get_videos_only(svo_file_path, video_path, timestamps_dict):
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Create a ZED camera object
    zed = sl.Camera()

    # Initialize the parameters for opening the SVO file
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(svo_file_path)
    init_params.svo_real_time_mode = False  # Disable real-time mode for better processing
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE #sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP # Set a coordinate system for tracking
    init_params.depth_stabilization = 0  # 1-100 level of stabilization
    
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Failed to open SVO file: {status}")
        return 
    
    left_image = sl.Mat()
    image_res = sl.Resolution(1280, 720)
    
    camera_info = zed.get_camera_information()
    
    fps = camera_info.camera_configuration.fps
    
    # check that timestamps_dict_keys are in order
    timestamps_dict_keys = list(timestamps_dict.keys())
    for i in range(len(timestamps_dict_keys) - 1):
        if timestamps_dict[timestamps_dict_keys[i]][-1] > timestamps_dict[timestamps_dict_keys[i + 1]][0]:
            raise ValueError(f"Timestamps for {timestamps_dict_keys[i]} and {timestamps_dict_keys[i + 1]} are not in order")
            return
    
    current_ep_idx = 0
    current_episode = list(timestamps_dict.keys())[current_ep_idx]
    start_timestamp = timestamps_dict[current_episode][0]
    end_timestamp = timestamps_dict[current_episode][-1]
    
    video_writers = {}
    for ep in timestamps_dict.keys():
        video_writers[ep] = None
        
    try:
        while True:
            if zed.grab() == sl.ERROR_CODE.SUCCESS:
                # Get the pose of the camera relative to the world frame
                timestamp = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
                
                zed.retrieve_image(left_image, view = sl.VIEW.LEFT, resolution = image_res)
                
                img = left_image.get_data()
                
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                
                h, w, c = img.shape
                
                if video_writers[current_episode] is None:
                    video_writers[current_episode] = cv2.VideoWriter(os.path.join(video_path, f"tracker_cam_{current_episode}.mp4"), fourcc, fps, (w, h))
                
                if timestamp >= start_timestamp and timestamp <= end_timestamp:
                    video_writers[current_episode].write(img)
                elif timestamp > end_timestamp:
                    print(f"Finished processing episode {current_episode}")
                    video_writers[current_episode].release()
                    current_ep_idx += 1
                    if current_ep_idx >= len(timestamps_dict):
                        break
                    current_episode = list(timestamps_dict.keys())[current_ep_idx]
                    start_timestamp = timestamps_dict[current_episode][0]
                    end_timestamp = timestamps_dict[current_episode][-1]
            else:
                print("Failed to grab frame")
                break
    finally:
        # Make sure to close everything properly even if there's an error
        for ep, writer in video_writers.items():
            if writer is not None:
                writer.release()
        zed.close()
        print("Camera and video writers closed")
        
# Process the SVO2 File to extract SLAM data and wrist tracker poses
def process_svo_gen2(svo_file_path, all_data, use_viewer = False, show_image = False, skip_slam = False, use_imu = True, use_roi = False, trackers = None, make_videos = False):
    
    if all_data is None:
        return
    
    if skip_slam:
        print("Skipping SLAM!!")
        
    if make_videos:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Create a ZED camera object
    zed = sl.Camera()

    # Initialize the parameters for opening the SVO file
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(svo_file_path)
    init_params.svo_real_time_mode = False  # Disable real-time mode for better processing
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS
    init_params.depth_minimum_distance = 1.00 # IN METERS
    init_params.depth_maximum_distance = 5.00 # IN METERS
    init_params.depth_stabilization = 0  # 1-100 level of stabilization

    # Open the SVO file
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Failed to open SVO file: {status}")
        return 
    
    sensor_data = sl.SensorsData()
    status = zed.get_sensors_data(sensor_data, sl.TIME_REFERENCE.IMAGE)
    if status == sl.ERROR_CODE.SUCCESS and sensor_data.get_imu_data().is_available:
        print("IMU data available in the SVO file.")
        use_imu = True
    else:
        print("No IMU data in the SVO file.!!")
        use_imu = False

    camera_info = zed.get_camera_information()
    calibration_params = camera_info.camera_configuration.calibration_parameters.left_cam
    
    fps = camera_info.camera_configuration.fps

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
    
    camera_matrix = np.array([[fx, 0, cx], 
                            [0, fy, cy], 
                            [0, 0, 1]], dtype=np.float32)
    
    dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)
    
    # initialize the trackers and set intrinsics
    hands = trackers.keys()
    if trackers is not None:
        for tracker in trackers.values():
            tracker.set_intrinsics(camera_matrix, dist_coeffs)

    # Enable positional tracking
    tracking_params = sl.PositionalTrackingParameters()
    tracking_params.set_gravity_as_origin = False
    tracking_params.enable_area_memory = True
    tracking_params.enable_pose_smoothing = False
    tracking_params.enable_imu_fusion = use_imu
    tracking_params.mode = sl.POSITIONAL_TRACKING_MODE.GEN_2
    # tracking_params.area_file_path = "test.area"
    
    if skip_slam:
        print("CAMERA IS STATIC")
        tracking_params.set_as_static = True
    
    status = zed.enable_positional_tracking(tracking_params)
    
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Failed to enable positional tracking: {status}")
        zed.close()
        return
    
    all_episodes = sorted(all_data.keys())
    
    for ep in all_episodes:
        all_data[ep]["zed_pose"] = []
        all_data[ep]["SLAM_failed"] = False
        if trackers is not None:
            for key in trackers.keys():
                all_data[ep][f"{key}_tracker"] = []
        
        save_camera_intrincs_path = os.path.join(all_data[ep]["save_zed_pose_path"], "camera_intrinsics.pkl")
        camera_intrinsics = {"cammera_matrix": camera_matrix, "distortion": disto}
        try:
            save_pickle(camera_intrinsics, save_camera_intrincs_path)
        except Exception as e:
            print("Failed to save camera intrinsics")
            print(e)

    curr_idx = 0
    curr_ep = all_episodes[curr_idx] # get the first episode in the list
    
    if use_viewer:
        camera_info = zed.get_camera_information()
        viewer = gl.GLViewer()
        viewer.init(camera_info.camera_model)
        
    left_image = sl.Mat()
    depth_map = sl.Mat()
    image_res = sl.Resolution(1280, 720)
    
    fail_count = 0
    exception_count = 0
    
    pose_diff_threshold = 0.15  # meters (if instantaneous pose change is greater than this, we just take the previous pose)
    last_img = None
    last_poses = {}
    for key in trackers.keys():
        last_poses[key] = None

    # Loop through frames in the SVO file
    while not use_viewer or viewer.is_available():
        try:
            if not skip_slam:
                zed_pose = sl.Pose()
            
            if zed.grab() == sl.ERROR_CODE.SUCCESS:
                # Get the pose of the camera relative to the world frame
                timestamp = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
                
                zed.retrieve_image(left_image, view = sl.VIEW.LEFT, resolution = image_res)
                raw_img = left_image.get_data()
                img = cv2.cvtColor(raw_img, cv2.COLOR_BGRA2BGR)
                
                zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH, resolution = image_res)
                depth_img = depth_map.get_data()
                
                # img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                
                overall_mask = np.ones((720, 1280, 3), dtype=np.uint8)
                
                poses = {}
                for key in trackers.keys():
                    poses[key] = None
                    
                frame_jumped = False
                if trackers is not None:
                    for key, tracker in trackers.items():
                        pose = tracker.get_pose(img, depth_img)
                        
                        if pose is None:
                            pose = np.array([0., 0., 0., 0., 0., 0., 1.]) 

                        # if last_poses[key] is not None and (timestamp >= all_data[curr_ep]["ep_start_ts"] and timestamp <= all_data[curr_ep]["ep_end_ts"]):
                        #     print(colored(f'Delta pose for {key} at timestamp {timestamp}: {np.linalg.norm(pose[0:3] - last_poses[key][0:3])}', "yellow"))
                        
                        if (last_poses[key] is not None and np.linalg.norm(pose[0:3] - last_poses[key][0:3]) > pose_diff_threshold) and (timestamp >= all_data[curr_ep]["ep_start_ts"] and timestamp <= all_data[curr_ep]["ep_end_ts"]):
                            frame_jumped = True
                            print(colored(f"Frame Jump Detected for {key}, skipping this frame. Distance: {np.linalg.norm(pose[0:3] - last_poses[key][0:3])}", "red"))
                            # skip this frame
                            break
                            
                        last_poses[key] = pose.copy()
                        last_img = img.copy()
                        
                        poses[key] = pose
                        
                        curr_mask_points = tracker.get_mask_points(pose)
                        curr_mask = get_mask(curr_mask_points, 1280, 720)

                        overall_mask = np.logical_and(overall_mask, curr_mask).astype(np.uint8)
                
                if frame_jumped:
                    # take last pose and last image
                    img = last_img
                    for key in poses.keys():
                        poses[key] = last_poses[key]
                
                for key, pose in poses.items():
                    assert pose is not None, f"Pose for {key} is None"
                    all_data[curr_ep][f"{key}_tracker"].append([timestamp] + pose.tolist())
                
                if make_videos:                            
                    overlay = tracker.overlay_cube_pose(
                        img,
                        poses
                    )
                    # Initialize the appropriate VideoWriter once we know frame shape
                    if all_data[curr_ep]["vid_writer"]is None:
                        h, w, c = overlay.shape
                        all_data[curr_ep]["vid_writer"] = cv2.VideoWriter(all_data[curr_ep]["tracker_video_path"] , fourcc, fps, (w, h))
                        
                    if timestamp >= all_data[curr_ep]["ep_start_ts"] and timestamp <= all_data[curr_ep]["ep_end_ts"]:
                        all_data[curr_ep]["vid_writer"].write(overlay)
                
                if use_roi:
                    print("using ROI")
                    overall_mask *= 255
                    
                    height, width = overall_mask.shape[:2]
                    mat_type = sl.MAT_TYPE.U8_C3
                    
                    roi_mat = sl.Mat(width, height, mat_type)
                    roi_mat_array = roi_mat.get_data()
        
                    np.copyto(roi_mat_array, overall_mask)
                    if zed.set_region_of_interest(roi_mat) != sl.ERROR_CODE.SUCCESS:
                        print("Failed to set ROI")
                
                if not skip_slam:  
                    state = zed.get_position(zed_pose, sl.REFERENCE_FRAME.WORLD)
                    tracking_status = zed.get_positional_tracking_status()
            
                img = cv2.resize(img, (320, 240))
            
                # show_image = True
                
                if show_image:
                    show_img = img.copy()
                    # visualize the curr_mask
                    if use_roi:
                        mask = cv2.resize(overall_mask, (320, 240))
                        # invert_mask = cv2.bitwise_not(mask).astype(np.uint8)
                        # apply mask
                        show_img[mask == 0] = 0
                        cv2.imshow("Mask", mask)
                        
                    cv2.imshow("Image", show_img)
                    key = cv2.waitKey(1)
                
                if not skip_slam and state == sl.POSITIONAL_TRACKING_STATE.OK:
                    py_translation = sl.Translation()
                    tx = zed_pose.get_translation(py_translation).get()[0]
                    ty = zed_pose.get_translation(py_translation).get()[1]
                    tz = zed_pose.get_translation(py_translation).get()[2]
                    
                    #Display orientation quaternion
                    py_orientation = sl.Orientation()
                    ox = zed_pose.get_orientation(py_orientation).get()[0]
                    oy = zed_pose.get_orientation(py_orientation).get()[1]
                    oz = zed_pose.get_orientation(py_orientation).get()[2]
                    ow = zed_pose.get_orientation(py_orientation).get()[3]
                    
                    if use_viewer:
                        rotation = zed_pose.get_rotation_vector()
                        translation = zed_pose.get_translation(py_translation)
                        text_rotation = str((round(rotation[0], 2), round(rotation[1], 2), round(rotation[2], 2)))
                        text_translation = str((round(translation.get()[0], 2), round(translation.get()[1], 2), round(translation.get()[2], 2)))

                        viewer.updateData(zed_pose.pose_data(sl.Transform()), text_translation, text_rotation, tracking_status)
                    # print(f"Translation: Tx: {tx}, Ty: {ty}, Tz: {tz}")
                else:
                    tx, ty, tz = 0, 0, 0
                    ox, oy, oz, ow = 0, 0, 0, 1
                    
                    if not skip_slam:
                        fail_count += 1
                        print(f"State: {state}")
                        print(f"Tracking Failed at {curr_ep}")
                    
                    if fail_count > 10:
                        all_data[curr_ep]["SLAM_failed"] = True
                        
                # check for nans 
                if np.isnan(tx) or np.isnan(ty) or np.isnan(tz) or np.isnan(ox) or np.isnan(oy) or np.isnan(oz) or np.isnan(ow):
                    print(colored("Nans detected", "red"))
                    print(f"tx: {tx}, ty: {ty}, tz: {tz}, ox: {ox}, oy: {oy}, oz: {oz}, ow: {ow}")
                    # set to last non nan value
                    for i in range(len(all_data[curr_ep]["zed_pose"]) - 1, 0, -1):
                        if not np.isnan(all_data[curr_ep]["zed_pose"][i, 1:4]).any() and not np.isnan(all_data[curr_ep]["zed_pose"][i, 4:]).any():
                            tx = all_data[curr_ep]["zed_pose"][i, 1]
                            ty = all_data[curr_ep]["zed_pose"][i, 2]
                            tz = all_data[curr_ep]["zed_pose"][i, 3]
                            ox = all_data[curr_ep]["zed_pose"][i, 4]
                            oy = all_data[curr_ep]["zed_pose"][i, 5]
                            oz = all_data[curr_ep]["zed_pose"][i, 6]
                            ow = all_data[curr_ep]["zed_pose"][i, 7]
                            break
                
                all_data[curr_ep]["zed_pose"].append([timestamp, tx, ty, tz, ox, oy, oz, ow])
                
                if all_data[curr_ep]["save_img_path"] is not None:
                    cv2.imwrite(os.path.join(all_data[curr_ep]["save_img_path"], f"{timestamp}.jpg"),  img)
                
                if timestamp > all_data[curr_ep]["ep_end_ts"]:
                    print(f"Finished processing episode {curr_ep}")
                    
                    fail_count = 0 # reset fail count
                    all_data[curr_ep]["zed_pose"] = np.array(all_data[curr_ep]["zed_pose"])
                    
                    all_data[curr_ep] = match_timestamps(all_data[curr_ep]) # moved inside
                        
                    retimestamp_images(all_data[curr_ep])

                    # match the timestamps of the tracker poses
                    for key, tracker in trackers.items():
                        if key == "left_hand":
                            new_key = "left_hand_tracker"
                        else:
                            new_key = "right_hand_tracker"
                
                        all_data[curr_ep][new_key] = np.array(all_data[curr_ep][new_key])
                        all_data[curr_ep][new_key] = retimestamp_tracker(all_data[curr_ep][new_key], all_data[curr_ep]["zed_pose"][:, 0])
                    
                    if make_videos:
                        if all_data[curr_ep]["vid_writer"] is not None:
                            all_data[curr_ep]["vid_writer"].release()
                    
                    # SLAM SUCCESSFUL
                    if not all_data[curr_ep]["SLAM_failed"] or skip_slam:
                        all_data[curr_ep]["zed_pose"] = smooth_path(all_data[curr_ep]["zed_pose"], sigma=2)
                            
                        # save output to a pickle file
                        smooth_zed_pose = all_data[curr_ep]["zed_pose"]
                        try:
                            smooth_zed_pose_path = os.path.join(all_data[curr_ep]["save_zed_pose_path"], 'zed_pose.pkl')
                            save_pickle(smooth_zed_pose, smooth_zed_pose_path)
                            print("Saved to ", smooth_zed_pose_path)
                        except Exception as e:
                            print("Failed to save zed pose")
                            print(e)
                        
                        save_one_path3d(smooth_zed_pose[:, 1:4], os.path.join(all_data[curr_ep]["save_zed_pose_path"], 'zed_pose.png'))
                        
                        for key in trackers.keys():
                            if key == "left_hand":
                                save_path = os.path.join(all_data[curr_ep]["left_save_zed_tracker_path"], 'zed_left_tracker.pkl')
                            elif key == "right_hand":
                                save_path = os.path.join(all_data[curr_ep]["right_save_zed_tracker_path"], 'zed_right_tracker.pkl')
                            else:
                                print(f"Unknown tracker key: {key}")
                            
                            try:
                                save_pickle(all_data[curr_ep][f"{key}_tracker"], save_path)
                                print("Saved to ", save_path)
                            except Exception as e:
                                print(f"Failed to save {key} tracker")
                                
                    curr_idx += 1
                    
                    # reset 
                    last_img = None
                    last_poses = {}
                    for key in trackers.keys():
                        last_poses[key] = None

                    if curr_idx >= len(all_episodes):
                        break
                    
                    curr_ep = all_episodes[curr_idx]
            else:
                print("Failed to grab frame")
                break
            
        except Exception as e:
            print("An unexpected error occurred during SLAM processing.")
            print(e)
            breakpoint()
            exception_count += 1
            
    # Close the camera and release resources
    
    print("Fail Count", exception_count)
    if use_viewer:
        viewer.exit()
    
    if not skip_slam:
        zed.disable_positional_tracking()
        
    zed.close()
    cv2.destroyAllWindows()

    return all_data

def retimestamp_tracker(tracker, zed_pose_ts):
    matched_tracker = []
    for i in range(zed_pose_ts.shape[0]):
        zed_ts = zed_pose_ts[i]
        closest_idx = np.argmin(np.abs(tracker[:, 0] - zed_ts))
        matched_tracker.append(tracker[closest_idx])
    
    return np.array(matched_tracker)

def match_timestamps(data):
    matched_zed_poses = data["zed_pose"]

    zed_pose_ts = data["zed_pose"][:, 0]
    
    tk_timestamps = data["timesteps"]
    
    matched_timestamps = []
    
    for i in range(tk_timestamps.shape[0]):
        closest_idx = np.argmin(np.abs(zed_pose_ts - tk_timestamps[i]))
        closest_zed_pose_ts = zed_pose_ts[closest_idx]
        matched_timestamps.append([tk_timestamps[i], closest_zed_pose_ts])
        zed_pose_ts = np.delete(zed_pose_ts, closest_idx)
    
    matched_timestamps = np.array(matched_timestamps)
    
    updated = np.zeros(matched_zed_poses.shape[0])
    for i in range(matched_timestamps.shape[0]):
        tk_timestamp = matched_timestamps[i, 0]
        zed_timestamp = matched_timestamps[i, 1]
        all_zed_timestamps = matched_zed_poses[:, 0]
        
        zed_index = np.where(all_zed_timestamps == zed_timestamp)[0][0]

        matched_zed_poses[zed_index, 0] = tk_timestamp
        updated[zed_index] = 1
        
    matched_zed_poses = matched_zed_poses[updated == 1]
    
    data["zed_pose"] = matched_zed_poses
    data["matched_timestamps"] = matched_timestamps
        
    return data

def retimestamp_images(data):
    image_dir = data["save_img_path"]
    matched_timestamps = data["matched_timestamps"]
    images = glob.glob(os.path.join(image_dir, "*.jpg")) 
    all_zed_matches = matched_timestamps[:, 1]
    all_tk_matches = matched_timestamps[:, 0]
    for i, name in enumerate(images):
        if "jpg" in name:
            zed_timestamp = int(os.path.basename(name).split(".")[0])
            if zed_timestamp in all_zed_matches:
                tk_timestamp = all_tk_matches[np.where(all_zed_matches == zed_timestamp)[0][0]]
                new_name = f"{tk_timestamp.astype(int)}.jpg"
                os.rename(name, os.path.join(image_dir, new_name))
            else:
                os.remove(name)

class ZedProcessor:
    def __init__(self, svo_path, use_viewer = False, show_image = False, skip_slam = False, use_imu = True, use_roi = False, face_corners_3d = {"left_hand": None},
                    cube_size = 8, marker_size = 6.4, transformation = [0, 0, 0.10, 0, 0, 0], marker_ids = {"left_hand": [None, 46, 44, 45, 47, 43]}):
        self.svo_file_path = svo_path
        self.use_viewer = use_viewer
        self.show_image = show_image
        self.skip_slam = skip_slam
        self.use_imu = use_imu
        self.use_roi = use_roi
        self.data_path = os.path.dirname(os.path.dirname(svo_path))
        self.no_arm = True
        self.make_videos = True
        
        self.trackers = {}
        
        self.hands = marker_ids.keys()
        
        # case for multiple cubes
        for key in marker_ids.keys():
            tracker = CubeTracker(None, None, cube_size, marker_size, marker_ids[key], face_corners_3d[key], transformation)
            self.trackers[key] = tracker
        
        if self.skip_slam:
            print("Skipping SLAM")
        if not self.use_imu:
            print("WARNING: Not using IMU!!")
        
        data_path = os.path.dirname(os.path.dirname(self.svo_file_path))
        
        try:
            start_episode = os.path.basename(self.svo_file_path).split("_")[-2]
            end_episode = os.path.basename(self.svo_file_path).split("_")[-1].split(".")[0]
            episodes = range(int(start_episode), int(end_episode) + 1)
        except:
            print("SLAM Failed to get episodes")
            self.all_data = None
            return
        
        self.all_data = {}
        
        for episode in episodes:
            try:
                episode_path = os.path.join(data_path, f"ep_{episode}")
                self.all_data[episode] = {}
                self.all_data[episode]["ep_path"] = episode_path
                if not os.path.exists(episode_path):
                    if os.path.exists(os.path.join(data_path, f"failed_ep_{episode}")):
                        os.rename(os.path.join(data_path, f"failed_ep_{episode}"), episode_path)
                        print(f"Resetting failed episode {episode}")
                self.all_data[episode]["timesteps"]= np.loadtxt(os.path.join(episode_path, 'timesteps', 'timesteps.txt'))
                self.all_data[episode]["ep_start_ts"] = self.all_data[episode]["timesteps"][0] #timestep is the first camera timestamp
                self.all_data[episode]["ep_end_ts"] = self.all_data[episode]["timesteps"][-1] #timestep is the last camera timestamp
                self.all_data[episode]["save_img_path"] = os.path.join(episode_path, 'zed_obs')
                self.all_data[episode]["save_zed_pose_path"] = os.path.join(episode_path, 'zed')
                if self.make_videos:
                    # Initialize video writers for each tracker (if needed)
                    self.all_data[episode]["vid_writer"] = None    
                    if "left_hand" in self.hands or "right_hand" in self.hands:
                        self.all_data[episode]["tracker_video_path"] = os.path.join(episode_path, 'tracker_debug.mp4') 
            except Exception as e:
                print(f"Failed to load episode {episode}")
                del self.all_data[episode]
                pass
        
        if "left_hand" in self.hands:
            for episode in self.all_data.keys():
                self.all_data[episode]["left_save_zed_tracker_path"] = os.path.join(self.all_data[episode]['ep_path'], 'left_tracker')
                os.makedirs(self.all_data[episode]["left_save_zed_tracker_path"], exist_ok=True)
                
                if os.path.exists(self.all_data[episode]["save_img_path"]):
                    print(f"Deleting {self.all_data[episode]['save_img_path']}")
                    shutil.rmtree(self.all_data[episode]["save_img_path"])
                    
                os.makedirs(self.all_data[episode]["save_img_path"])
                
        if "right_hand" in self.hands:
            for episode in self.all_data.keys():
                self.all_data[episode]["right_save_zed_tracker_path"] = os.path.join(self.all_data[episode]['ep_path'], 'right_tracker')
                os.makedirs(self.all_data[episode]["right_save_zed_tracker_path"], exist_ok=True)
                
                if not "left_hand" in self.hands: # delete and make directory only if left hand is not being processed as well
                    if os.path.exists(self.all_data[episode]["save_img_path"]):
                        print(f"Deleting {self.all_data[episode]['save_img_path']}")
                        shutil.rmtree(self.all_data[episode]["save_img_path"])
                        
                    os.makedirs(self.all_data[episode]["save_img_path"])

    def process_slam_data(self):
        all_data = process_svo_gen2(self.svo_file_path, self.all_data, self.use_viewer, self.show_image, self.skip_slam, self.use_imu, self.use_roi, self.trackers, self.make_videos)
        
        if all_data is None:
            print("Failed to process SLAM data")
            return
        
        failed_episodes = np.array([key for key, data in all_data.items() if data["SLAM_failed"]])
        
        # free up memory
        del all_data
        del self.all_data

        print("Failed Episodes", failed_episodes)
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Process ZED SVO files")
    parser.add_argument(
        '--recording_path', 
        '-p',
        type=str, 
        required=True, 
        help="Path to the SVO file to process."
    )
    
    parser.add_argument("--no_slam",
                        "-n",
                        action="store_true",
                        help="No SLAM")
    
    parser.add_argument(
        '--use_viewer',
        '-v', 
        action='store_true', 
        help="Enable 3D visualization with the OpenGL viewer."
    )
    parser.add_argument(
        '--show_image', 
        '-im',
        action='store_true', 
        help="Show image frames during processing."
    )
    
    parser.add_argument(
        "--use_imu",
        "-imu",
        action='store_true',
        help="Use IMU data for tracking"
    )
    
    use_roi = True
    isleft = False
    
    args = parser.parse_args()
    
    svo_file_path = args.recording_path
    show_image = args.show_image
    use_viewer = args.use_viewer
    skip_slam = args.no_slam
    use_imu = args.use_imu
    
    cube = 8
    marker = 6.4
    
    half_cube = cube / 2
    half_marker = marker / 2
    
    if isleft:
        print("Processing left hand zed tracker")
        face_corners = {"left_hand": left_face_corners_3d}
        marker_ids = {"left_hand": [None, 46, 44, 45, 47, 43]}
    else:
        print("Processing right hand zed tracker")
        face_corners = {"right_hand": right_face_corners_3d}
        marker_ids = {"right_hand" :[None, 3, 0, 4, 5, 2]}
    
    zed_processor = ZedProcessor(svo_file_path, use_viewer, show_image, skip_slam, use_imu, use_roi, face_corners, cube_size = 8, marker_size=6.4, transformation = [0, 0, 0.10, 0, 0, 0], marker_ids = marker_ids)
    zed_processor.process_slam_data()
    
    print("DONE PROCESSING")
    