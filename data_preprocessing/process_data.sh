#!/bin/bash

# Bash script to run the mass data processing script

source /opt/ros/humble/setup.bash
source /home/$USER/dexwild/dexwild_ros2/install/setup.bash

# Path to your Python script
SCRIPT_PATH="mass_process_data.py"

# Basic settings
IS_ROBOT=false        # Set True if this is robot data
PARALLELIZE=true      # Set True if you want to process in parallel (usually yes)
LEFT_HAND=true        # Set True to process left hand
RIGHT_HAND=true       # Set True to process right hand

# Data Directory
DATA_DIR="/path/to/data_folder"  # SET to your input dataset directory

# Processing options
PROCESS_SVO=false      # Process zed data
SKIP_SLAM=false        # Skip SLAM (set True if camera is static)
PROCESS_HAND=true      # Retarget hand to robot joint angles
PROCESS_TRACKER=false  # Clean up tracker  wrist trajectories
CLIP_ACTIONS=false     # Clip action outliers
INTERGRIPPER=false     # Process intergripper poses (for bimanual tasks)

# Optional arguments
GENERATE_VIDEOS=false  # Generate videos of processed results
HAND_TYPE="v2"         # Choose the hand type: "v1" or "v2"
LANG=false             # Add a natural language description

# Construct command
CMD="python3 $SCRIPT_PATH --data_dir $DATA_DIR"

[ "$LEFT_HAND" = true ] && CMD+=" --left_hand"
[ "$RIGHT_HAND" = true ] && CMD+=" --right_hand"
[ "$PROCESS_SVO" = true ] && CMD+=" --process_svo"
[ "$SKIP_SLAM" = true ] && CMD+=" --skip_slam"
[ "$PARALLELIZE" = true ] && CMD+=" --parallelize"
[ "$PROCESS_HAND" = true ] && CMD+=" --process_hand"
[ "$PROCESS_TRACKER" = true ] && CMD+=" --process_tracker"
[ "$CLIP_ACTIONS" = true ] && CMD+=" --clip_actions"
[ "$INTERGRIPPER" = true ] && CMD+=" --intergripper"
[ "$GENERATE_VIDEOS" = true ] && CMD+=" --generate_videos"
[ "$HAND_TYPE" != "" ] && CMD+=" --hand_type $HAND_TYPE"
[ "$IS_ROBOT" = true ] && CMD+=" --robot"
[ "$LANG" = true ] && CMD+=" --lang"

# Run the command
echo "Running command:"
echo "$CMD"
eval $CMD
