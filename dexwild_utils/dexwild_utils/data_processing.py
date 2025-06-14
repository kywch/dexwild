import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import dexwild_utils.leap_hand_utils as lhu
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy.ndimage import gaussian_filter1d

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol = pickle.HIGHEST_PROTOCOL)
    assert(os.path.exists(path)), "File Did Not Save Correctly"
        
def left_face_corners_3d(orientation, c, m):
        """
        orientation: a string like '+Z', '-Z', '+Y', etc.
        Returns the 4 corner points in 3D for a marker of physical size marker_size_cm
        that is centered on the face of a cube whose half-side is half_side.
        """
        
        '''
        Coord Order:
        
        Top-left corner.
        Top-right corner.
        Bottom-right corner.
        Bottom-left corner.
        '''
        
        if orientation == '+Z':
            # Marker plane at Z = +c, corners ±m in X and Y
            return None
        elif orientation == '-Z':
            # Marker plane at Z = -c
            return np.array([
                [ m,  m, -c],
                [ -m,  m, -c],
                [ -m, -m, -c],
                [m, -m, -c]
            ], dtype=np.float32)
        elif orientation == '+Y':
            # Marker plane at Y = +c, corners ±m in X and Z
            return np.array([
                [m,  c,  m],
                [-m,  c,  m],
                [ -m,  c,  -m],
                [ m,  c, -m]
            ], dtype=np.float32)
        elif orientation == '-Y':
            # Marker plane at Y = -c
            return np.array([
                [-m, -c,  -m],
                [ -m, -c,  m],
                [m, -c,  m],
                [m, -c, -m]
            ], dtype=np.float32)
        elif orientation == '+X':
            # Marker plane at X = +c, corners ±m in Y and Z
            return np.array([
                [ c,  -m,  -m],
                [ c,  -m,  m],
                [ c, m,  m],
                [ c, m,  -m]
            ], dtype=np.float32)
        elif orientation == '-X':
            # Marker plane at X = -c
            return np.array([
                [-c,  m, -m],
                [-c,  m,  m],
                [-c,  -m,  m],
                [-c,  -m, -m]
            ], dtype=np.float32)
        else:
            raise ValueError("Unknown orientation string.")   
        
def right_face_corners_3d(orientation, c, m):
    """
    orientation: a string like '+Z', '-Z', '+Y', etc.
    Returns the 4 corner points in 3D for a marker of physical size marker_size_cm
    that is centered on the face of a cube whose half-side is half_side.
    """
    
    '''
    Coord Order:
    
    Top-left corner.
    Top-right corner.
    Bottom-right corner.
    Bottom-left corner.
    '''
    
    if orientation == '+Z':
        # Marker plane at Z = +c, corners ±m in X and Y
        return None
    elif orientation == '-Z':
        # Marker plane at Z = -c
        return np.array([
            [ -m,  m, -c],
            [ -m,  -m, -c],
            [ m, -m, -c],
            [m, m, -c]
        ], dtype=np.float32)
    elif orientation == '+Y':
        # Marker plane at Y = +c, corners ±m in X and Z
        return np.array([
            [m,  c,  -m],
            [m,  c,  m],
            [ -m,  c,  m],
            [ -m,  c, -m]
        ], dtype=np.float32)
    elif orientation == '-Y':
        # Marker plane at Y = -c
        return np.array([
            [-m, -c,  -m],
            [ -m, -c,  m],
            [m, -c,  m],
            [m, -c, -m]
        ], dtype=np.float32)
    elif orientation == '+X':
        # Marker plane at X = +c, corners ±m in Y and Z
        return np.array([
            [ c,  m,  -m],
            [ c,  -m,  -m],
            [ c, -m,  m],
            [ c, m,  m]
        ], dtype=np.float32)
    elif orientation == '-X':
        # Marker plane at X = -c
        return np.array([
            [-c,  m, -m],
            [-c,  m,  m],
            [-c,  -m,  m],
            [-c,  -m, -m]
        ], dtype=np.float32)
    else:
        raise ValueError("Unknown orientation string.")
        
def index_episodes(data_dir):
    all_episodes = os.listdir(data_dir)
    all_episodes = [episode for episode in all_episodes if episode.startswith('ep') and os.path.isdir(os.path.join(data_dir, episode))]
    all_episodes = sorted(all_episodes, key=lambda x: int(x.split('_')[-1]))
    return all_episodes

def get_clip_thresh(data_dir, all_actions, percentile=99.0):
    trans = all_actions[:, :3]
    quats = all_actions[:, 3:]
    
    trans_norms = np.linalg.norm(trans, axis=1)  # shape (N,)
    
    # 2) Convert quaternion to a rotation angle: angle = 2 * arccos(qw)
    qw = quats[:, 3]
    qw_clamped = np.clip(qw, -1.0, 1.0)  # avoid invalid arccos
    angles = 2.0 * np.arccos(qw_clamped)  # shape (N,) in radians
    
    trans_thresh = np.percentile(trans_norms, percentile)
    rot_thresh = np.percentile(angles, percentile)
    
    print(f"Translational Threshold: {trans_thresh}")
    print(f"Rotational Threshold: {rot_thresh}")
    
    # plot the distributuon of norms
    plt.hist(trans_norms, bins=50, alpha=0.5, label='Translation Norms')
    plt.legend(loc='upper right')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Translation Norms')
    save_path = os.path.join(data_dir, "stats", "translation_norms_hist.png")
    plt.savefig(save_path)
    plt.close('all')
    
    plt.hist(angles, bins=50, alpha=0.5, label='Rotation Angles')
    plt.legend(loc='upper right')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Rotation Angles')
    save_path = os.path.join(data_dir, "stats", "rotation_angles_hist.png")
    plt.savefig(save_path)
    plt.close('all')
    
    return trans_thresh, rot_thresh

def clip_one_episode(actions, trans_thresh, rot_thresh):
    """
    Given an (N, 7) array of actions, clip both translation and rotation
    to not exceed trans_thresh or rot_thresh.
    
    actions: (N, 7) -> each row is [x, y, z, qx, qy, qz, qw]
    clip_thresh: tuple (trans_thresh, rot_thresh)
    """
    
    # keep track of which actions are clipped
    clipped = np.zeros(actions.shape[0], dtype=bool)
    
    # Split into translation and quaternion parts
    translations = actions[:, :3]      # (N, 3)
    quaternions  = actions[:, 3:]      # (N, 4)

    # 1) Clip translation norm
    trans_norms = np.linalg.norm(translations, axis=1, keepdims=True)  # shape (N, 1)
    too_large = trans_norms > trans_thresh
    
    clipped[too_large.reshape(-1)] = True
    
    # scale factor is 1 if below threshold, or ratio to bring norm down to trans_thresh
    scale_factor = np.ones_like(trans_norms)
    scale_factor[too_large] = trans_thresh / (trans_norms[too_large] + 1e-8)
    clipped_translations = translations * scale_factor
    
    # 2) Clip rotation
    #    angle = 2 * arccos(qw). If angle > rot_thresh, we clamp.
    clipped_quats = []
    for i, q in enumerate(quaternions):
        qx, qy, qz, qw = q
        current_angle = 2.0 * np.arccos(np.clip(qw, -1.0, 1.0))
        
        if current_angle <= rot_thresh:
            # No clipping needed
            clipped_quats.append(q)
        else:
            # Clamp the angle to rot_thresh
            clamped_angle = rot_thresh
            
            # Rotation axis
            axis = np.array([qx, qy, qz], dtype=float)
            axis_norm = np.linalg.norm(axis)
            
            # If degenerate quaternion (axis ~ zero), just use identity rotation
            if axis_norm < 1e-8:
                clipped_quats.append([0.0, 0.0, 0.0, 1.0])
                continue
            
            axis_unit = axis / axis_norm
            
            half_angle = clamped_angle / 2.0
            new_qw = np.cos(half_angle)
            sin_half = np.sin(half_angle)
            new_qx = axis_unit[0] * sin_half
            new_qy = axis_unit[1] * sin_half
            new_qz = axis_unit[2] * sin_half
            
            clipped_quats.append([new_qx, new_qy, new_qz, new_qw])
            
            clipped[i] = True
    
    clipped_quats = np.array(clipped_quats)
    
    # 3) Re-combine clipped translation + clipped rotation
    clipped_actions = np.hstack([clipped_translations, clipped_quats])
        
    return clipped_actions, np.sum(clipped)

def auto_match(array1, array2, array1_ts, array2_ts):

    def get_closest_indices(ref_timestamps, query_timestamps):
    # For each element in query_timestamps, find which index in ref_timestamps is closest
        return np.array([np.abs(ref_timestamps - t).argmin() for t in query_timestamps])

    if array1.shape[0] < array2.shape[0]:
        idxs = get_closest_indices(array2_ts, array1_ts)
        # Now pick out the rows from head_data that best match each zed timestamp
        array2 = array2[idxs]
    
    else:
        idxs = get_closest_indices(array1_ts, array2_ts)
        
        array1 = array1[idxs]
        
    # check that the lengths are the same
    assert array1.shape[0] == array2.shape[0], f"Timestamps do not match after auto-matching. Shapes: {array1.shape}, {array2.shape}"
    
    print(f"Auto-matched timestamps. New shapes: {array1.shape}, {array2.shape}")
    
    return array1, array2

def smooth_path(data, sigma = 2):
    data= gaussian_filter1d(data, sigma, axis=0)
    return data

def slerp(quat0, quat1, t):
    """
    Spherical Linear intERPolation for quaternions.
    quat0, quat1 shape: (4,) in order (qx, qy, qz, qw)
    t: float in [0,1]
    """
    # Use scipy Rotation for convenience
    r0 = R.from_quat(quat0)
    r1 = R.from_quat(quat1)
    # Define the keyframes for SLERP
    key_times = [0, 1]  # Start and end times
    key_rots = R.concatenate([r0, r1])  # Concatenate rotations

    # Create the Slerp object
    slerp = Slerp(key_times, key_rots)

    # Interpolate at time `t`
    interpolated_rot = slerp([t])[0]
    return interpolated_rot.as_quat()  # returns [qx, qy, qz, qw]

def leapv2_to_leapv1(leapv2_data, ema_amount = 0.2):
    v2_motors_side =    [0,3,6,9,12]
    v2_motors_forward = [1,4,7,10,13]
    v2_motors_curl =    [2,5,8,11,14]
    v2_motors_palm =    [15,16]  # 15 is for the thumb, 16 is between the 4 fingers, 
    v2_all_motors = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    
    timestamps = leapv2_data[:, 0]
    leapv2_data = leapv2_data[:, 1:]  # remove timestamps
    
    leapv1_data = np.zeros((leapv2_data.shape[0], 16))  # initialize leapv1 data
    
    prev_pos = None
    
    for i in range(leapv2_data.shape[0]):
        leapv2_joints = leapv2_data[i, :]
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
        
        # nominal values for leapv1 joints
        nominal_open = [-3.42902317e-01, -4.13555283e-01, -1.62902839e-01,
                            4.44754828e-01, -5.98419838e-03, -4.23198776e-01, -1.73425600e-01,
                            4.45409983e-01,  3.16566506e-01, -3.91376854e-01, -2.06551253e-01,
                            5.48279969e-01,  2.05204916e-02, -1.18676704e-01,  3.90739471e-01,
                            -9.40697476e-03
                            ]
        
        nominal_closed =  [4.98404110e-02, 1.76172334e+00, 5.71870863e-01,
                            1.47162994e+00, 4.52223428e-01, 2.00516025e+00, 4.97504339e-01,
                            1.51511221e+00, 2.83616293e-01, 2.06586776e+00, 4.33664997e-01,
                            1.52440990e+00, 8.41671372e-01, 7.92767966e-01, 2.95032727e-01,
                            1.26972666e+00]
        
        #LEAPV1: The joint numbering goes from Index (0-3), Middle(4-7), Ring(8-11) to Thumb(12-15) and from MCP Side, MCP Forward, PIP, DIP for each finger.
        leapv1_joints[0:3] = leapv2_joints_normed[0:3] * nominal_closed[0:3] + (1 - leapv2_joints_normed[0:3]) * nominal_open[0:3]
        leapv1_joints[1] += leapv2_joints_normed[16] * nominal_closed[1] + (1 - leapv2_joints_normed[16]) * nominal_open[1] # account for palm forward
        leapv1_joints[3] = leapv2_joints_normed[2] * nominal_closed[3] + (1 - leapv2_joints_normed[2]) * nominal_open[3] + 1
        
        leapv1_joints[4:7] = leapv2_joints_normed[3:6] * nominal_closed[4:7] + (1 - leapv2_joints_normed[3:6]) * nominal_open[4:7]
        leapv1_joints[5] += leapv2_joints_normed[16] * nominal_closed[5] + (1 - leapv2_joints_normed[16]) * nominal_open[5] # account for palm forward
        leapv1_joints[7] = leapv2_joints_normed[5] * nominal_closed[7] + (1 - leapv2_joints_normed[5]) * nominal_open[7] + 1
        
        leapv1_joints[8:11] = leapv2_joints_normed[6:9] * nominal_closed[8:11] + (1 - leapv2_joints_normed[6:9]) * nominal_open[8:11]
        leapv1_joints[9] += leapv2_joints_normed[16] * nominal_closed[9] + (1 - leapv2_joints_normed[16]) * nominal_open[9] # account for palm forward
        leapv1_joints[11] = leapv2_joints_normed[8] * nominal_closed[11] + (1 - leapv2_joints_normed[8]) * nominal_open[11] + 1
        
        leapv1_joints[12:15] = leapv2_joints_normed[12:15] * nominal_closed[12:15] + (1 - leapv2_joints_normed[12:15]) * nominal_open[12:15]
        leapv1_joints[12] += leapv2_joints_normed[15] * nominal_closed[12] + (1 - leapv2_joints_normed[15]) * nominal_open[12] # account for thumb palm forward
        leapv1_joints[14] = leapv2_joints_normed[15] * nominal_closed[14] + (1 - leapv2_joints_normed[15]) * nominal_open[14]

        leapv1_joints[15] = leapv2_joints_normed[11] * nominal_closed[15] + (1 - leapv2_joints_normed[11]) * nominal_open[15]
        
        # use ema
        
        pose = lhu.allegro_to_LEAPhand(leapv1_joints)
        curr_pos = np.array(pose)
        # use ema
        if prev_pos is None:
            prev_pos = curr_pos
            
        curr_pos = prev_pos * (1 - ema_amount) + curr_pos * ema_amount
        
        leapv1_data[i, :] = curr_pos
        # curr_pos[12] = 3 * np.pi / 2
        prev_pos = curr_pos
    
    leapv1_data = np.hstack((timestamps.reshape(-1, 1), leapv1_data))
    return leapv1_data