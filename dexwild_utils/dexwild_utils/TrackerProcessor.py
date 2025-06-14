from dexwild_utils.pose_utils import pose7d_to_mat, mat_to_pose7d, plot_one_path3d, save_one_path3d, pose6d_to_mat, plot_paths3d, save_paths3d
from dexwild_utils.data_processing import load_pickle, auto_match, smooth_path, slerp
import os
import pickle
import numpy as np
import argparse


class TrackerProcessor:
    def __init__(self, episode_path = None, isleft = True):
        self.episode_path = episode_path
        self.isleft = isleft
        # transformation from the zed camera frame to head camera frame
        self.zed_to_head_cam_mat = pose6d_to_mat(np.array([0.023, -0.036, 0.0, 0.0, 0.0, 0.0 ]))
        
    def interpolate_smooth(self, tracker_in_world, tracking_lost, timestamps):
        # Find consecutive “lost” blocks
        lost_blocks = []
        start_idx = None
        for i in range(len(tracking_lost)):
            if tracking_lost[i] == 1 and start_idx is None:
                start_idx = i
            elif tracking_lost[i] == 0 and start_idx is not None:
                # We just ended a lost block
                lost_blocks.append((start_idx, i - 1))
                start_idx = None
        # if the last frames are lost
        if start_idx is not None:
            lost_blocks.append((start_idx, len(tracking_lost) - 1))
        
        # Interpolate each lost block
        # Pose is [x,y,z, qx,qy,qz,qw]
        for (s, e) in lost_blocks:
            block_len = e - s + 1
            # previous valid index is s-1 if s>0
            prev_idx = s - 1 if s > 0 else s
            # next valid index is e+1 if e < N-1
            next_idx = e + 1 if e < (len(tracking_lost) - 1) else e

            prev_pose = tracker_in_world[prev_idx].copy()  # [x,y,z, qx,qy,qz,qw]
            next_pose = tracker_in_world[next_idx].copy()  # [x,y,z, qx,qy,qz,qw]

            # If the lost block is at the very start, we “extend” next_pose
            if s == 0:
                for i in range(s, e+1):
                    tracker_in_world[i] = next_pose
                continue
            
            # If the lost block is at the very end, we “extend” prev_pose
            if e == len(tracking_lost) - 1:
                for i in range(s, e+1):
                    tracker_in_world[i] = prev_pose
                continue

            # Otherwise, linearly interpolate positions, slerp orientations
            # For frames from s..e, we param in (1..block_len)
            # param t in [0..1] from s->e
            for idx in range(s, e+1):
                alpha = (idx - s + 1) / (block_len + 1)  # fraction along the block

                # Positions
                p0 = prev_pose[0:3]  # x,y,z
                p1 = next_pose[0:3]
                pos_interp = p0 * (1 - alpha) + p1 * alpha

                # Orientations
                q0 = prev_pose[3:7]  # [qx,qy,qz,qw]
                q1 = next_pose[3:7]
                quat_interp = slerp(q0, q1, alpha)  # shape (4,)

                tracker_in_world[idx, 0:3] = pos_interp
                tracker_in_world[idx, 3:7] = quat_interp
        
        # Smooth path
        tracker_in_world = smooth_path(tracker_in_world)
        
        tracker_in_world = np.hstack((timestamps.reshape(-1, 1), tracker_in_world))
        
        return tracker_in_world

    def single_tracker(self, zed_tracker_data):
        print("wrist tracker shape:", zed_tracker_data.shape)
        aligned_data = np.zeros_like(zed_tracker_data)

        first_zed_tracker_pose = None
        
        temp_zed = []
        
        tracking_lost = np.zeros(zed_tracker_data.shape[0])
        
        for i in range(zed_tracker_data.shape[0]):
            zed_tracker_pose = pose7d_to_mat(zed_tracker_data[i, 1:])
            
            zed_tracker_lost = np.allclose(zed_tracker_data[i, 1:],  np.array([0., 0., 0., 0., 0., 0., 1.]), atol=1e-5)
            
            # if trackers are lost, just skip
            if zed_tracker_lost:
                aligned_data[i, 1:] = np.array([0., 0., 0., 0., 0., 0., 1.])
                tracking_lost[i] = 1
                continue
            
            if first_zed_tracker_pose is None and not zed_tracker_lost:
                first_zed_tracker_pose = zed_tracker_pose
                
            if first_zed_tracker_pose is None:
                zed_tracker_pose_in_zed_frame = np.eye(4)
            else:
                zed_tracker_pose_in_zed_frame = np.linalg.inv(first_zed_tracker_pose) @ zed_tracker_pose
            
            aligned_data[i, 1:] = mat_to_pose7d(zed_tracker_pose_in_zed_frame)
            
            temp_zed.append(mat_to_pose7d(zed_tracker_pose_in_zed_frame))
        
        aligned_data[:, 0] = zed_tracker_data[:, 0] # timestamps
        return aligned_data, tracking_lost

    
    def process_tracker(self):
        episode_path = self.episode_path
        isleft = self.isleft
        
        if isleft:
            zed_tracker_path = os.path.join(episode_path, 'left_tracker', 'zed_left_tracker.pkl')
        else:
            zed_tracker_path = os.path.join(episode_path, 'right_tracker', 'zed_right_tracker.pkl')
        
        
        zed_path = os.path.join(episode_path, 'zed', 'zed_pose.pkl')
        
        try:
            zed_tracker_data = load_pickle(zed_tracker_path)
                
            zed_data = load_pickle(zed_path)
            
        except Exception as e:
            print(f"Error loading data in episode: {episode_path}", e)
            if isleft:
                fname = "left_tracking_lost.txt"
            else:
                fname = "right_tracking_lost.txt"
            with open(os.path.join(episode_path, fname), 'w') as f:
                f.write("Error Loading Tracker Data\n")
            return 100.0
        
        tracker_raw_data = zed_tracker_data.copy()
        tracker_data, tracking_lost = self.single_tracker(zed_tracker_data)
        
        # print("Tracker Shape:", tracker_data.shape)
        # print("Zed Shape:", zed_data.shape)
        
        timestamps = tracker_data[:, 0]
        
        tracker_data = tracker_data[:, 1:]
        tracker_raw_data = tracker_raw_data[:, 1:]
        zed_data = zed_data[:, 1:]
        
        tracker_in_world = tracker_data.copy()

        # ratio of tracking lost
        tracking_lost_ratio = np.sum(tracking_lost) / tracker_in_world.shape[0]
        print("tracking lost ratio:", tracking_lost_ratio)
        
        if tracking_lost_ratio > 0.25:
            # save a txt file to indicate that the tracking is lost
            if isleft:
                print("WARNING!! Left Tracking lost ratio is too high. Skipping...")
                fname = "left_tracking_lost.txt"
            else:
                print("WARNING!! Right Tracking lost ratio is too high. Skipping...")
                fname = "right_tracking_lost.txt"
            with open(os.path.join(episode_path, fname), 'w') as f:
                f.write("Tracking is lost\n")
                f.write("Total Frames: {}\n".format(tracker_in_world.shape[0]))
                f.write("Tracking lost frames: {}\n".format(np.sum(tracking_lost)))
                f.write(f"Tracking lost ratio: {tracking_lost_ratio}")
                
            return tracking_lost_ratio
        
        # for the frames where tracking is lost, interpolate between the previous and next known position
        tracker_in_world = self.interpolate_smooth(tracker_in_world, tracking_lost, timestamps) # smoothed out tracker data in world frame
        tracker_raw_interpolated_data = self.interpolate_smooth(tracker_raw_data, tracking_lost, timestamps) # smoothed out raw tracker data (in camera frame)
        
        if isleft:
            save_one_path3d(tracker_in_world[:, 1:4], os.path.join(episode_path, 'left_tracker', 'left_tracker_world_abs.png'))
            save_one_path3d(tracker_raw_interpolated_data[:, 1:4], os.path.join(episode_path, 'left_tracker', 'left_tracker_cam_frame_abs.png'))
            with open(os.path.join(episode_path, 'left_tracker', 'left_tracker_world_abs.pkl'), 'wb') as f:
                pickle.dump(tracker_in_world, f)
            with open(os.path.join(episode_path, 'left_tracker', 'left_tracker_cam_frame_abs.pkl'), 'wb') as f:
                pickle.dump(tracker_raw_interpolated_data, f)
        else:
            save_one_path3d(tracker_in_world[:, 1:4], os.path.join(episode_path, 'right_tracker', 'right_tracker_world_abs.png'))
            save_one_path3d(tracker_raw_interpolated_data[:, 1:4], os.path.join(episode_path, 'right_tracker', 'right_tracker_cam_frame_abs.png'))
            with open(os.path.join(episode_path, 'right_tracker', 'right_tracker_world_abs.pkl'), 'wb') as f:
                pickle.dump(tracker_in_world, f)
            with open(os.path.join(episode_path, 'right_tracker', 'right_tracker_cam_frame_abs.pkl'), 'wb') as f:
                pickle.dump(tracker_raw_interpolated_data, f)
                
        return tracking_lost_ratio

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Tracker files")
    parser.add_argument(
        '--episiode_path', 
        '-p',
        type=str, 
        required=False, 
        help="Path to the episode folder to process."
    )
    parser.add_argument("--isleft",
                        "-l", 
                        action="store_true", 
                        help="Is the tracker on the left hand?")
    
    args = parser.parse_args()
    
    episode_path = args.episiode_path
    isleft = args.isleft
    
    tracker_processor = TrackerProcessor(episode_path, isleft)
    tracker_processor.process_tracker() 
    