
import os
from pathlib import Path
from preprocessing import preprocess_image

def main():
    # Define the specific files to process
    target_files = ["fall001.jpg", "fall002.jpg", "fall003.jpg"]
    
    # We will look for them in dataset/images/train as a default location
    # verify if they exist there
    base_src_dir = Path("dataset/images/train")
    base_dst_dir = Path("dataset/images_preprocessed_selected")

    print(f"Processing selected files from {base_src_dir} to {base_dst_dir}...")
    
    for filename in target_files:
        src_path = base_src_dir / filename
        if src_path.exists():
            dst_path = base_dst_dir / filename
            print(f"Processing {src_path}...")
            preprocess_image(str(src_path), str(dst_path))
        else:
            print(f"Warning: {src_path} does not exist.")

if __name__ == "__main__":
    main()
