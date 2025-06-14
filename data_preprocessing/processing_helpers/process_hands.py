import os
import time
import ray
from tqdm import tqdm

from dexwild_utils.data_processing import load_pickle, save_pickle
from dexwild_utils.data_processing import leapv2_to_leapv1
from retargeting.leap_v2_ik import ray_process_glove_data, process_glove_data
from retargeting.leap_ik import ray_process_glove_data_v1, process_glove_data_v1

def process_leapv2_to_v1_wrapper(all_episodes, data_dir, left_hand, right_hand):
    print(f"Converting {len(all_episodes)} episodes from leap v2 to leap v1")
    hands = []
    if left_hand:
        hands.append("left_hand")
    if right_hand:
        hands.append("right_hand")
        
    for episode in tqdm(all_episodes):   
        episode_path = os.path.join(data_dir, episode)
        for hand in hands:
            if hand == "left_hand":
                source_hand_path = os.path.join(episode_path, 'left_leapv2', 'left_leapv2.pkl')
                target_hand_path= os.path.join(episode_path, 'left_leapv1', 'left_leapv1.pkl')
            elif hand == "right_hand":
                source_hand_path = os.path.join(episode_path, 'right_leapv2', 'right_leapv2.pkl')
                target_hand_path= os.path.join(episode_path, 'right_leapv1', 'right_leapv1.pkl')
            os.makedirs(os.path.dirname(target_hand_path), exist_ok=True)
            
            if os.path.exists(source_hand_path):
                leapv2_data = load_pickle(source_hand_path)
                leapv1_data = leapv2_to_leapv1(leapv2_data)
                save_pickle(leapv1_data, target_hand_path)
            else:
                print(f"Source hand path {source_hand_path} does not exist, skipping...")
                continue

def process_hand_wrapper(all_episodes, data_dir, left_hand, right_hand, parallelize, hand_type):
    if parallelize:
        print(f"Processing {len(all_episodes)} episodes for hands (Parallelized with Ray)")
        hands = []
        if left_hand:
            hands.append("left_hand")
        if right_hand:
            hands.append("right_hand")
            
        ray.init(num_gpus = 1)  # Or ray.init(address="auto") if you're using a Ray cluster
        print(ray.cluster_resources())
        
        batch_size =  min(len(all_episodes), 24)
    
        batches = [all_episodes[i:i + batch_size] for i in range(0, len(all_episodes), batch_size)]
        
        for batch in batches:
            # Create Ray tasks
            futures = []
            
            for episode in batch:
                episode_path = os.path.join(data_dir, episode)
                for hand in hands:
                    if hand_type == "v2":
                        if hand == "left_hand":
                            glove_path = os.path.join(episode_path, 'left_manus', 'left_manus.pkl')
                            hand_path= os.path.join(episode_path, 'left_leapv2', 'left_leapv2.pkl')
                            isleft = True
                        elif hand == "right_hand":
                            glove_path = os.path.join(episode_path, 'right_manus', 'right_manus.pkl')
                            hand_path= os.path.join(episode_path, 'right_leapv2', 'right_leapv2.pkl')
                            isleft = False
                    elif hand_type == "v1":
                        if hand == "left_hand":
                            glove_path = os.path.join(episode_path, 'left_manus', 'left_manus.pkl')
                            hand_path= os.path.join(episode_path, 'left_leapv1', 'left_leapv1.pkl')
                            isleft = True
                        elif hand == "right_hand":
                            glove_path = os.path.join(episode_path, 'right_manus', 'right_manus.pkl')
                            hand_path= os.path.join(episode_path, 'right_leapv1', 'right_leapv1.pkl')
                            isleft = False
                            
                    os.makedirs(os.path.dirname(hand_path), exist_ok=True)
                    
                    if hand_type == "v2":
                        futures.append(ray_process_glove_data.remote(glove_path, hand_path, isleft, False))
                    elif hand_type == "v1":
                        futures.append(ray_process_glove_data_v1.remote(glove_path, hand_path, isleft, False))
                        
            results = []
            for fut in tqdm(futures, desc="Processing Glove", total=len(futures)):
                # Ray get one by one for nice progress
                res = ray.get(fut)  
                results.append(res)

            # Print final summary
            print("All Glove processed!")
            for elapsed in results:
                print(f"Glove took {elapsed:.2f}s")
            
        # Shutdown Ray
        ray.shutdown()
    
    else:
        print(f"Processing {len(all_episodes)} episodes for hands (Sequentially)")
        hands = []
        if left_hand:
            hands.append("left_hand")
        if right_hand:
            hands.append("right_hand")
            
        for episode in tqdm(all_episodes):   
            episode_path = os.path.join(data_dir, episode)
            for hand in hands:
                if hand_type == "v2":
                    if hand == "left_hand":
                        glove_path = os.path.join(episode_path, 'left_manus', 'left_manus.pkl')
                        hand_path= os.path.join(episode_path, 'left_leapv2', 'left_leapv2.pkl')
                        isleft = True
                    elif hand == "right_hand":
                        glove_path = os.path.join(episode_path, 'right_manus', 'right_manus.pkl')
                        hand_path= os.path.join(episode_path, 'right_leapv2', 'right_leapv2.pkl')
                        isleft = False
                    os.makedirs(os.path.dirname(hand_path), exist_ok=True)                
                elif hand_type == "v1":
                    if hand == "left_hand":
                        glove_path = os.path.join(episode_path, 'left_manus', 'left_manus.pkl')
                        hand_path= os.path.join(episode_path, 'left_leapv1', 'left_leapv1.pkl')
                        isleft = True
                    elif hand == "right_hand":
                        glove_path = os.path.join(episode_path, 'right_manus', 'right_manus.pkl')
                        hand_path= os.path.join(episode_path, 'right_leapv1', 'right_leapv1.pkl')
                        isleft = False
                        
                    os.makedirs(os.path.dirname(hand_path), exist_ok=True)
                
                if hand_type == "v2":
                    process_glove_data(glove_path, hand_path, isleft, False)
                elif hand_type == "v1":
                    process_glove_data_v1(glove_path, hand_path, isleft, False)