import pickle
import os
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from dexwild_utils.pose_utils import plot_one_path3d, poses7d_to_mats, mats_to_poses7d, plot_paths3d
from dexwild_utils.data_processing import load_pickle
import argparse
import ipdb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "file_path",
        type=str,
        default =""
    )
    args = parser.parse_args()
    file_path = args.file_path
    
    data = load_pickle(file_path)
    
    # path = data[:, 1:4]
    
    # plot_one_path3d(path)
    
    ipdb.set_trace()
    

def get_image(encoded):
    encoded_image_np = np.frombuffer(encoded, dtype=np.uint8)
    bgr_image = cv2.imdecode(encoded_image_np, cv2.IMREAD_COLOR)
    assert bgr_image.shape[:2] == (256, 256), f"height and width don't match! {bgr_image.shape[:2]}"
    rgb_image = bgr_image[:,:,::-1]
    return bgr_image

file_path = "/home/tony/umi-hand-data/dynamic_picking/robot/data_buffers/leapv2_dynamic_picking_robot_only_mar31/mixed_no_eef/buffer.pkl"
np.set_printoptions(suppress=True)

data = load_pickle(file_path)
# data2 = load_pickle(file_path2)

# traj = data[:, 1:]
# traj2 = data2[:, 1:]

# plot_one_path3d(traj)
# plot_paths3d([traj, traj2])
print(data)
breakpoint()

first_data = data[0]

for i in range(0, len(first_data)):
    inputs = first_data[i][0]
    
    image = get_image(inputs["enc_cam_1"])
    
    # print(image.shape)
    
    cv2.imshow("img", image)
    
    key = cv2.waitKey(30)  # Wait for 30 ms (adjust for speed)

cv2.destroyAllWindows()
   
# cv2.destroyAllWindows()

breakpoint()

for traj_idx in range(0, len(data)):
    for time_idx in range(0, len(data[traj_idx])):
        state = data[traj_idx][time_idx][0]
        action = data[traj_idx][time_idx][1]
        
        for key in state.keys():
            if np.isnan(state[key]).any():
                print("State nan found at", traj_idx, time_idx, key)
                breakpoint()
        
        if np.isnan(action).any():
            print("nan found at", traj_idx, time_idx, "action")
            breakpoint()

print("no nans found")
# check for nans
breakpoint()


# print(data)
# print(data[0])
# print(data)

# check for nans
# nan_found = Falsebbbbb
# for i in range(0, len(data)):
#     if np.isnan(data[i][1:]).any():
#         print("nan found at", i)
#         nan_found = True
# print("nan found", nan_found)
quit()

# undo the first pose

mats = poses7d_to_mats(data[:, 1:])
first_mat = mats[0]
first_mat_inv = np.linalg.inv(first_mat)

for i in range(0, len(mats)):
    mats[i] = first_mat_inv @ mats[i]
    
data[:, 1:] = mats_to_poses7d(mats)

print("first_pose", data[0, 1:])
print("last_pose", data[-1, 1:])

plot_one_path3d(data[:, 1:4])

# print("first eef", data[0][1:4])
# print("first euler", R.from_quat(data[0][4:]).as_euler('xyz'))
# print(data.shape)
breakpoint()

# img = data[0][0]

# cv2.imshow("img", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

