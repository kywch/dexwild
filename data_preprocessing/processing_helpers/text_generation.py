import tqdm as tqdm
import os

def generate_text_labels(data_dir, all_episodes, dummy_label = ""):
    """
    Generate text labels for each episode. 
    If dummy_label is provided, it will be used as the label for all episodes.
    """
    print("GENERATING TEXT LABELS...")
    
    for episode in tqdm(all_episodes):
        episode_path = os.path.join(data_dir, episode)
        save_path = os.path.join(episode_path, "language")
        os.makedirs(save_path, exist_ok=True)
        text_labels_path = os.path.join(save_path, "language.txt")
        
        if not os.path.exists(text_labels_path):
            with open(text_labels_path, 'w') as f:
                if dummy_label:
                    f.write(dummy_label)