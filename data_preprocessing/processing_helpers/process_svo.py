import os
import time

import ray
from tqdm import tqdm

def process_single_svo(svo_path, face_corners_3d, cube_size, marker_size, transformation, marker_ids, skip_slam=False):
    """
    Ray remote function to process one SVO file.
    """
    start = time.time()

    # Import everything needed INSIDE the function for Ray workers
    from dexwild_utils.ZedProcessor import ZedProcessor  # or wherever ZedProcessor is defined
    
    if skip_slam: 
        use_roi = False 
    else:
        use_roi = True

    zed_processor = ZedProcessor(
        svo_path,
        use_viewer=False,
        show_image=False,
        skip_slam=skip_slam,
        use_imu=True,
        use_roi=use_roi,
        face_corners_3d = face_corners_3d,
        cube_size=cube_size,
        marker_size=marker_size,
        transformation=transformation,
        marker_ids=marker_ids,
    )

    try:
        zed_processor.process_slam_data()
    except Exception as e:
        print(f"Error Processing {svo_path}")
        print(e)
        
    # free up memory
    del zed_processor
    
    elapsed = time.time() - start
    
    return svo_path, elapsed

def process_svo_wrapper(all_svos, slam_dir, corner_faces, cube_size, marker_size, transformation, marker_ids, skip_slam, parallelize):
    if parallelize:
        print("Processing SLAM... (Parallelized with Ray)")
        
        batch_size = min(len(all_svos), 20)
        
        batches = [all_svos[i:i + batch_size] for i in range(0, len(all_svos), batch_size)]
        
        print(f"Number of batches: {len(batches)}")
        
        # Initialize Ray
        ray.init(num_gpus = 1)  # Or ray.init(address="auto") if you're using a Ray cluster
        print(ray.cluster_resources()) 
        
        @ray.remote(num_gpus=1/15, max_retries=0) # WRAP in RAY
        def process_single_svo_ray(svo_path, face_corners_3d, cube_size, marker_size, transformation, marker_ids, skip_slam):
            return process_single_svo(svo_path, face_corners_3d, cube_size, marker_size, transformation, marker_ids, skip_slam)
        
        for batch in batches:
            # Create Ray tasks
            futures = []
            
            for slam_svo in batch:
                if "output" in slam_svo:
                    start = time.time()
                    svo_path = os.path.join(slam_dir, slam_svo)
                    svo_path = os.path.join(slam_dir, slam_svo)
                    futures.append(process_single_svo_ray.remote(svo_path, corner_faces, cube_size, marker_size, transformation, marker_ids, skip_slam))
            
            results = []
            for fut in tqdm(futures, desc="Processing SVOs", total=len(futures)):
                # Ray get one by one for nice progress
                try:
                    res = ray.get(fut)  
                    results.append(res)
                except ray.exceptions.WorkerCrashedError:
                    print("Worker failed, skipping this task.")
                except Exception as e:
                    print(f"Task failed: {e}")

            # Print final summary
            print("All SVOs processed!")
            for svo_path, elapsed in results:
                print(f"SVO {os.path.basename(svo_path)} took {elapsed:.2f}s")
            
        # Shutdown Ray
        ray.shutdown()
        
    else:
        print("Processing SLAM... (Sequentially)")
        
        for slam_svo in all_svos:
            if "output" in slam_svo:
                start = time.time()
                svo_path = os.path.join(slam_dir, slam_svo)
                print(f"Processing {svo_path}")
                
                svo_path, elapsed = process_single_svo(svo_path, corner_faces, cube_size, marker_size, transformation, marker_ids, skip_slam)
    
                
                print(f"{svo_path} Processed in {elapsed} seconds")