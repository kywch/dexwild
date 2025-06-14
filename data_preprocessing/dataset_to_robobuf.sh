#!/bin/bash

# Default values — change these as needed
RETAIN_PCT=1.0           # Percentage of original data to retain (1.0 = keep all data), always removes from the end of trajectories

UPSAMPLE_PCTS=("0.60" "1.0")    # List of upsample intervals (e.g., upsample between 60% and 100% of each trajectory)
UPSAMPLE_MULTIPLIER=0           # 0 = no upsampling; n = replicate samples to increase dataset size by n×
SUBSAMPLE_RATE=1.0              # 1.0 = no subsampling; <1.0 = keep only a fraction of the data.

MODE="abs"                      # Action mode: "abs" = absolute actions, "rel" = relative actions
ROT_REPR="rot6d"                # Rotation representation: choose from "quat", "euler", "rot6d"

HUMAN=True                      # True = process human demonstration data
LEFT_HAND=True                  # True = include left hand data
RIGHT_HAND=True                 # True = include right hand data
LANGUAGE=False                  # True = include language annotations

DATA="/path/to/data"            # Root directory containing data folders
BUF_NAME="testing_for_release"  # Name/tag of the buffer or processed output group

# Construct the command
CMD="python3 robobuf/dataset_to_robobuf.py \
    --retain_pct $RETAIN_PCT \
    --upsample_multiplier $UPSAMPLE_MULTIPLIER \
    --mode $MODE \
    --data $DATA \
    --buf_name $BUF_NAME\
    --rot_repr $ROT_REPR \
    --subsample_rate $SUBSAMPLE_RATE \
    "

if [ "$NORMALIZE" = "True" ]; then
    CMD="$CMD --normalize"
fi

if [ "$HUMAN" = "True" ]; then
    CMD="$CMD --human"
fi

if [ "$LEFT_HAND" = "True" ]; then
    CMD="$CMD --left_hand"
fi

if [ "$RIGHT_HAND" = "True" ]; then
    CMD="$CMD --right_hand"
fi

if [ "$LANGUAGE" = "True" ]; then
    CMD="$CMD --language"
fi

# Add list arguments
CMD="$CMD --upsample_pcts ${UPSAMPLE_PCTS[*]}"

# Echo and run the command
echo "Running: $CMD"
eval $CMD
