from datasets import load_dataset
import pandas as pd
import os

def download_go_emotions():
    # Create folder if it doesn't exist
    os.makedirs("data/raw", exist_ok=True)
    
    print("Connecting to Hugging Face to download GoEmotions...")
    # Load the simplified version (one label per text)
    dataset = load_dataset("go_emotions", "simplified")
    
    # Convert the 'train' split to a DataFrame
    df = pd.DataFrame(dataset['train'])
    
    # Save it to the exact path main.py is looking for
    df.to_csv("data/raw/emotion_data.csv", index=False)
    print("✅ Success! 'data/raw/emotion_data.csv' has been created.")

if __name__ == "__main__":
    download_go_emotions()