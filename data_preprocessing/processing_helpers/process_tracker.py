import os
import time
import ray
from tqdm import tqdm
import matplotlib.pyplot as plt

from dexwild_utils.TrackerProcessor import TrackerProcessor

def process_single_tracker(episode_path, left_hand, right_hand):
    start = time.time()
    
    if left_hand:
        tracker_processor = TrackerProcessor(episode_path, isleft=True)
        tracking_lost_ratio = tracker_processor.process_tracker()
    if right_hand:
        tracker_processor = TrackerProcessor(episode_path, isleft=False)
        tracking_lost_ratio = tracker_processor.process_tracker()
        
    elapsed = time.time() - start
    return episode_path, tracking_lost_ratio, elapsed

@ray.remote
def ray_process_single_tracker(episode_path, left_hand, right_hand):
    return process_single_tracker(episode_path, left_hand, right_hand)
    
def process_tracker_wrapper(all_episodes, data_dir, left_hand, right_hand, parallelize):
    
    # make a tracking lost dict
    tracking_lost_dict = {}
    if parallelize:
        print(f"Processing {len(all_episodes)} episodes for tracker (Parallelized with Ray)")
        ray.init(num_gpus = 1)  # Or ray.init(address="auto") if you're using a Ray cluster
        print(ray.cluster_resources())
        
        batch_size = min(len(all_episodes), 200)
    
        batches = [all_episodes[i:i + batch_size] for i in range(0, len(all_episodes), batch_size)]
        
        for batch in batches:
            # Create Ray tasks
            futures = []
            for episode in batch:
                episode_path = os.path.join(data_dir, episode)
                futures.append(ray_process_single_tracker.remote(episode_path, left_hand, right_hand))
                    
            results = []
            for fut in tqdm(futures, desc="Processing Trackers", total=len(futures)):
                # Ray get one by one for nice progress
                res = ray.get(fut)  
                results.append(res)

            # Print final summary
            print("All Trackers processed!")
            for episode_path, tracking_lost_ratio, elapsed in results:
                print(f"Tracker {os.path.basename(episode_path)} took {elapsed:.2f}s")
                tracking_lost_dict[os.path.basename(episode_path)] = tracking_lost_ratio
        # Shutdown Ray
        ray.shutdown()
    else:
        print(f"Processing {len(all_episodes)} episodes for tracker (Sequentially)")
            
        for episode in tqdm(all_episodes):
            episode_path = os.path.join(data_dir, episode)
            episode_path, tracking_lost_ratio, elapsed = process_single_tracker(episode_path, left_hand, right_hand)
            tracking_lost_dict[os.path.basename(episode_path)] = tracking_lost_ratio
            
            print(f"{episode} Trackers Processed in {elapsed} seconds")
    
    # show a histogram of tracking losts
    plt.hist(tracking_lost_dict.values(), bins=20)
    plt.xlabel("Tracking Lost Ratio")
    plt.ylabel("Frequency")
    plt.title("Tracking Lost Ratio Histogram")
    save_path = os.path.join(data_dir, "stats", "tracking_lost_hist.png")
    plt.savefig(save_path)
    plt.close('all')

    #show a bar chart of tracking losts
    plt.bar(tracking_lost_dict.keys(), tracking_lost_dict.values())
    plt.xlabel("Episode")
    # rotate x labels
    plt.xticks(rotation=90)
    # make labels smaller
    plt.tick_params(axis='x', which='major', labelsize=5)
    plt.ylabel("Tracking Lost Ratio")
    plt.title("Tracking Lost Ratios")
    save_path = os.path.join(data_dir, "stats", "tracking_lost_ratios.png")
    plt.savefig(save_path)
    plt.close('all')
    
    # save the tracking lost ratio dictionary as a text file
    with open(os.path.join(data_dir, "stats", "tracking_lost_ratios.txt"), 'w') as f:
        for key in tracking_lost_dict.keys():
            f.write(f"{key}: {tracking_lost_dict[key]}\n") 