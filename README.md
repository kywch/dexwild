<h1 align="center"> DexWild: Dexterous Human Interactions for In-the-Wild Robot Policies </h1>


<div align="center">

#### [Tony Tao](https://tony-tao.com/)<sup>\*</sup>, [Mohan Kumar Srirama](https://www.mohansrirama.com/)<sup>\*</sup>, [Jason Jingzhou Liu](https://jasonjzliu.com/), [Kenneth Shaw](https://kennyshaw.net/), [Deepak Pathak](https://www.cs.cmu.edu/~dpathak/)

#### Robotics: Science and Systems (RSS) 2025

<p align="center">
    <img src="website_assets/imgs/teaser.gif" width="25%"> 
</p>

[[Project page]](https://dexwild.github.io/) [[Video]](https://youtu.be/oMaamSkcl5E) [[ArXiv]](https://arxiv.org/abs/2505.07813) 


[[Hardware Guide]](https://resisted-salad-9e6.notion.site/DexWild-Hardware-Setup-Guide-20eee3f68d27801b8eb8dfde3a5bb7c4?source=copy_link) [[Data Collection Guide]](https://resisted-salad-9e6.notion.site/DexWild-Data-Collection-Guide-20fee3f68d27803b9a88ef9847e292d4?source=copy_link) [[Policy Training]](https://github.com/dexwild/dexwild-training)


[![Linux platform](https://img.shields.io/badge/Platform-linux--64-orange.svg)](https://ubuntu.com/blog/tag/22-04-lts) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()


</div>

---

# Overview of Repository

Folder Structure:
```Bash
├── _hardware           # 3D files
├── _MANUS_SDK          # Manus Glove SDK
├── data_preprocessing  # Used for data preprocessing
├── dexwild_ros2        # ROS2 workspace
├── dexwild_utils       # Miscellaneous Utilities
├── misc_scripts        # Miscellaneous Scripts
├── model_checkpoints   # Model Checkpoints
├── shell_scripts       # Shell Scripts
├── training            # Training Code
└── website_assets      # Assets for Website
```

# Data Collection

## 🛠️ Hardware Guide

Setup hardware components following this [Hardware Guide](https://resisted-salad-9e6.notion.site/DexWild-Hardware-Setup-Guide-20eee3f68d27801b8eb8dfde3a5bb7c4?source=copy_link)

## ⚙️ Environment Setup (Main Computer)
Only tested on Ubuntu 22.04.

Core Dependencies:
1. Install [ROS2 Humble](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html) on your machine.
2. Install [ZED SDK 4.2](https://www.stereolabs.com/docs/installation/linux) on your machine. Make sure it is the correct version (4.2).

Since ROS2 has common compatibility issues with conda, we recommend installing everything into the **base system python**.

3. Clone this repository and cd into the directory:
    ```bash
    git clone git@github.com:dexwild/dexwild.git
    cd dexwild
    ```
4. Install Dependencies
    ```bash
    pip install -r requirements.txt
    ```

5. Install DexWild Utils
    ```bash
    cd dexwild_utils
    pip install -e .
    ```

5. Build ROS Packages
    ```bash
    cd dexwild_ros2
    source /opt/ros/humble/setup.bash

    colcon build --symlink-install
    ```

    This should create `build`, `install`, and `log`, folders.

For robot data collection, install:
1. Install [GELLO](https://github.com/wuphilipp/gello_software) following this guide. 
2. Install [XARM Python SDK](https://github.com/xArm-Developer/xArm-Python-SDK) following this guide.

## 🖥️ Environment Setup (Mini-PC)
Only tested on Nvidia [Jetpack 5.1.1](https://developer.nvidia.com/embedded/jetpack-sdk-511), but later versions should still work.

Core Dependencies:
1. Install [ROS2 Foxy](https://docs.ros.org/en/foxy/Installation.html) on your machine. If using later version of Jetpack, [ROS2 Humble](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html) will also work.
2. Install [ZED SDK 4.2](https://www.stereolabs.com/docs/installation/linux) on your machine. Make sure it is the correct version (4.2).

3. Docker Image For Manus SDK
    Pull the docker image from docker hub
    ```bash
    docker pull boardd/manussdk:v0
    ```

    Setup for Docker:
    ```bash
    sudo apt update
    sudo apt install -y qemu qemu-user-static

    sudo update-binfmts --install qemu-x86_64 /usr/bin/qemu-x86_64-static
    sudo update-binfmts --enable qemu-x86_64
    sudo update-binfmts --display qemu-x86_64

    # Check that QEMU configuration is registered
    cat /proc/sys/fs/binfmt_misc/qemu-x86_64

    sudo docker run --rm --platform linux/amd64 debian uname -m
    # expected output: x86_64

    ```

4. Clone this repository and cd into the directory:
    ```bash
    git clone git@github.com:dexwild/dexwild.git
    cd ~/dexwild
    ```
5. Install Dependencies
    ```bash
    pip install -r requirements.txt
    ```

6. Install DexWild Utils
    ```bash
    cd dexwild_utils
    pip install -e .
    ```

7. Build ROS Packages
    ```bash
    cd dexwild_ros2
    source /opt/ros/foxy/setup.bash
    # alternatively
    source /opt/ros/humble/setup.bash

    colcon build --symlink-install
    ```

    This should create `build`, `install`, and `log`, folders.

8. Misc Installs
    ```bash
    sudo apt update
    sudo apt install gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly
    ```

8. Update Desktop File Paths and put on desktop.
    ```bash
    cd ~/dexwild
    cd shell_scripts/desktop_apps
    ```
    First go to `launch_collect.desktop`. This is a one click app that launches all of the data collection code. 

    Edit the two lines such that the paths match your username.
    ```bash
    Exec=/home/$USERNAME/dexwild/shell_scripts/kill_tmux.sh
    Icon=/home/$USERNAME/dexwild/shell_scripts/imgs/stop.png
    ```

    Then go to `kill_collect.desktop`. This is one click app that shuts down all data collection cleanly.
    
    Similarly, update the paths such that the paths have your username.
    
    Last, copy the desktop files to your applications folder

    ```bash
    mkdir -p ~/.local/share/applications
    cp launch_collect.desktop kill_collect.desktop ~/.local/share/applications/

    # make executable
    chmod +x ~/.local/share/applications/launch_collect.desktop
    chmod +x ~/.local/share/applications/kill_collect.desktop

    update-desktop-database ~/.local/share/applications/
    ```

    Lastly, it helps to favorite the apps so that they appear in the sidebar for easy access.

## 📈 Data Collection Instructions

Follow [Data Collection Instruction](https://resisted-salad-9e6.notion.site/DexWild-Data-Collection-Guide-20fee3f68d27803b9a88ef9847e292d4?source=copy_link) to collect data.

Data is saved in the following structure:
```bash
data
├── ep_0
│   ├── intergripper
│   │   └── intergripper.pkl
│   ├── left_leapv2
│   │   └── left_leapv2.pkl
│   ├── left_manus
│   │   └── left_manus_full.pkl, left_manus.pkl
│   ├── left_pinky_cam
│   │   └── timestamp1.jpg, timestamp2.jpg, ...
│   ├── left_thumb_cam
│   │   └── timestamp1.jpg, timestamp2.jpg, ...
│   ├── left_tracker
│   │   └── left_tracker_cam_frame_abs.pkl, ...
│   ├── right_leapv2
│   │   └── right_leapv2.pkl
│   ├── right_manus
│   │   └── right_manus_full.pkl, right_manus.pkl
│   ├── right_pinky_cam
│   │   └── timestamp1.jpg, timestamp2.jpg, ...
│   ├── right_thumb_cam
│   │   └── timestamp1.jpg, timestamp2.jpg, ...
│   ├── right_tracker
│   │   └── right_tracker_cam_frame_abs.pkl, ...
│   ├── timesteps
│   │   └── timesteps.txt
│   ├── zed
│   │   └── zed_ts.pkl  
│   └── zed_obs
│       └── timestamp1.jpg, timestamp2.jpg, ...
├── ep_1
├── ep_2
├── ...
└── zed_recordings
    ├── output_0_4.svo2
    ├── output_5_9.svo2
    └── ...
```

# Data Processing

There are two main scripts for data processing. One preprocesses the data and the other turns the processed data into a robobuf buffer

## 🧮 Preprocessing

In `data_preprocessing/process_data.sh` there are a few parameters that must be changed.

```bash
# Basic settings
IS_ROBOT=false        # Set True if this is robot data
PARALLELIZE=true      # Set True if you want to process in parallel (usually yes)
LEFT_HAND=true        # Set True to process left hand
RIGHT_HAND=true       # Set True to process right hand

# Data Directory
DATA_DIR="/path/to/data_folder"  # SET to your input dataset directory

# Processing options
PROCESS_SVO=true       # Process zed data
SKIP_SLAM=false        # Skip SLAM (set True if camera is static)
PROCESS_HAND=false     # Retarget hand to robot joint angles
PROCESS_TRACKER=false  # Clean up tracker wrist trajectories
CLIP_ACTIONS=false     # Clip action outliers
INTERGRIPPER=false     # Process intergripper poses (for bimanual tasks)

# Optional arguments
GENERATE_VIDEOS=false  # Generate videos of processed results
HAND_TYPE="v2"         # Choose the hand type: "v1" or "v2"
LANG=false             # Add a natural language description
```

After checking that the parameters are what you want, run the script. It may take up to 30 minutes depending on how much data you have and how powerful your computer is.

```bash
chmod +x process_data.sh
./process_data.sh
```

## 💾 Buffer Creation

First make sure you have the `robobuf` package
```bash
pip install git+https://github.com/AGI-Labs/robobuf.git
```

Now, we must convert the data from our format to one that is consumable by the training. 

In `data_preprocessing/dataset_to_robobuf.sh` there are a also few parameters that must be changed.

```bash
RETAIN_PCT=1.0           # Percentage of original data to retain (1.0 = keep all data), always removes from the end of trajectories

UPSAMPLE_PCTS=("0.60" "1.0")    # List of upsample intervals (e.g., upsample between 60% and 100% of each trajectory)
UPSAMPLE_MULTIPLIER=0           # 0 = no upsampling; n = replicate samples to increase dataset size by n×
SUBSAMPLE_RATE=1.0              # 1.0 = no subsampling; <1.0 = keep only a fraction of the data.

MODE="abs"                      # Action mode: "abs" = absolute actions, "rel" = relative actions
ROT_REPR="rot6d"                # Rotation representation: choose from "quat", "euler", "rot6d"

HUMAN=True                      # True = process human demonstration data?
LEFT_HAND=True                  # True = include left hand data
RIGHT_HAND=True                 # True = include right hand data
LANGUAGE=False                  # True = include language annotations

DATA="/path/to/data"            # Root directory containing data folders
BUF_NAME="buffer name"          # Name/tag of the buffer
```

Usually, all parameters can just be kept constant, except a few:

1. Change the path to the task data folder containing all the data for the particular task
    ```bash
    data_buffers/
    task_data/
    ├── data_1/
    │   ├── ep_0/
    │   ├── ep_1/
    │   └── ...
    ├── data_2/
    │   ├── ep_0/
    │   └── ...
    └── ...
    ```
2. Choose a name for the buffer. This will be saved in `data_buffers` folder as `BUF_NAME`
3. Choose if the data is human demonstration or robot
4. Choose which hands are present in dataset

When you're ready
```bash
chmod +x dataset_to_robobuf.sh
./dataset_to_robobuf.sh
```

# Training

To train the policy, all data must be in robobuf format.

First clone the repository used for training.

```bash
git clone https://github.com/dexwild/dexwild-training
cd ~/dexwild-training
```

Follow the install and training instructions in the [training repository](https://github.com/dexwild/dexwild-training).

# Deployment

There are two launch files for deployment, one for single arm and single hand, and the other for bimanual. Within each launch file, the main parameters to change are as listed below:

```bash
"checkpoint_path":  # Path to the trained policy checkpoint
"replay_path":  # Path to the input Robobuf replay data
"replay_id":  # Index of the episode in the replay buffer to play
"id":  # Identifier for setup type (e.g., "bimanual", "left_mobile")
"observation_keys":  # List of sensor keys used as inputs to the policy

"openloop_length":  # Number of steps to run before running inference again
"skip_first_actions":  # Number of initial actions to skip (to account for delay)

"freq":  # Control loop frequency in Hz
"rot_repr":  # Rotation representation used for actions ("euler", "quat", "rot6d")

"buffer_size":  # Number of past steps to keep for input to model. Must be at least max length of history for policy.

"ema_amount":  # Exponential moving average weight for action smoothing
"use_rmp":  # If True, uses Riemannian Motion Policy (RMP) controller

"start_poses":  # Initial arm poses at start of episode (flattened list)
"start_hand_poses":  # Initial hand poses at start of episode (flattened list)

"pred_horizon":  # How many future steps the model predicts for ensembling
"exp_weight":  # Weight for blending old vs new predictions (0 = only latest) for ensembling

"mode":  # Action interpretation mode ("rel", "abs", "hybrid", etc.)
```

## Single Arm Deployment
```bash
cd ~/dexwild
cd dexwild_ros2
ros2 launch launch/deploy_policy.launch.py
```

## Bimanual Deployment
```bash
cd ~/dexwild
cd dexwild_ros2
ros2 launch launch/bimanual_deploy_policy.launch.py
```

# Citation
If you find this project useful, please cite our work:
```
@article{tao2025dexwild,
      title={DexWild: Dexterous Human Interactions for In-the-Wild Robot Policies},
      author={Tao, Tony and Srirama, Mohan Kumar and Liu, Jason Jingzhou and Shaw, Kenneth and Pathak, Deepak},
      journal={Robotics: Science and Systems (RSS)},
      year={2025}}

```

