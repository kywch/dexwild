from robobuf.buffers import ObsWrapper, Transition, ReplayBuffer
import pickle as pkl
import numpy as np
from tqdm import tqdm
import ipdb
from termcolor import colored


# Validate the contents of a robobuf buffer for NaN values and missing images
def validate_buffer(buffer_path):
    
    cam_indexes = [0, 1, 2, 3]
    
    # load the buffer
    with open(buffer_path, "rb") as f:
        buf = ReplayBuffer.load_traj_list(pkl.load(f))
        
    buf_len = len(buf)
    
    index_list = list(range(buf_len))
    
    for i in tqdm(index_list):
        stp = buf[i]
        action = stp.action
        obs = stp.obs
        state = obs.state
        
        # check for nans in action
        for key in action.keys():
            if np.isnan(action[key]).any():
                print(f"Action nan found at {i} {key}")
                ipdb.set_trace()
        
        # check for nans in state
        for key in state.keys():
            if np.isnan(state[key]).any():
                print(f"State nan found at {i} {key}")
                ipdb.set_trace()
        
        for cam_idx in cam_indexes:
            image = obs.image(cam_idx)
            if image is None:
                print(f"Image None found at {i} cam {cam_idx}")
                ipdb.set_trace()
    
    print(colored("No nans found in buffer", "green"))
                

if __name__ == "__main__":
    buffer_path = "/mnt/drive2/umi-hand-data/clothes_folding/data_buffers/cloth_bimanual_robot_not_normalized_apr23/buffer/buffer.pkl"
    validate_buffer(buffer_path)