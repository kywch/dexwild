import cloudpickle as pkl

# Define buffer paths using a dictionary for better organization
buffer_paths = {
    "spray_human": "/home/tony/umi-hand-data/spray_on_cloth_wild2/data_buffers/spray_wild_jan22_human_only_full/buffer.pkl",
    "spray_robot": "/home/tony/umi-hand-data/spray_on_cloth_wild2/data_buffers/robot_all_jan20/mixed_no_eef/buffer.pkl",
    "toy_cleanup_human": "/home/tony/umi-hand-data/toy_cleanup_wild/data_buffers/toy_human_all_jan23/buffer.pkl",
    "toy_cleanup_robot": "/home/tony/umi-hand-data/toy_cleanup_wild/data_buffers/toy_robot_all_jan23/mixed_no_eef/buffer.pkl",
    "pour_human": "/home/tony/umi-hand-data/pour_wild/data_buffers/pour_wild_human_full/buffer.pkl",
    "pour_robot": "/home/tony/umi-hand-data/pour_wild/data_buffers/pour_robot_only_lab_jan27/mixed_no_eef/buffer.pkl",
}

for name, path in buffer_paths.items():
    try:
        with open(path, "rb") as f:
            traj_buffer = pkl.load(f)
            num_trajectories = len(traj_buffer)
            print(f"{name}: {num_trajectories} trajectories")  # More concise output
    except FileNotFoundError:
        print(f"Error: File not found at {path}")  # Handle potential file errors
    except Exception as e:  # Catch other potential errors during loading
        print(f"Error loading {path}: {e}")