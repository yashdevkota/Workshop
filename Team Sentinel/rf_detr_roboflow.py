from inference import get_model
import supervision as sv
import cv2
import numpy as np
import os

# Load a pre-trained yolov8n model
model = get_model(model_id="fall-detection-djoqw/10")

# Initialize annotators
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

def process_frame(frame: np.ndarray, index: int) -> np.ndarray:
    """
    Callback function to process each video frame:
    1. Runs inference
    2. Annotates the frame
    """
    # Runs inference on the frame
    results = model.infer(frame)[0]

    # Load results into supervision Detections API
    detections = sv.Detections.from_inference(results)

    # Annotate the frame
    annotated_image = bounding_box_annotator.annotate(
        scene=frame, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections)
    
    return annotated_image

def process_single_video(source_path, target_path):
    if not os.path.exists(source_path):
        print(f"Error: Source file does not exist: {source_path}")
        return

    print(f"Processing {source_path} -> {target_path}...")
    
    try:
        sv.process_video(
            source_path=source_path,
            target_path=target_path,
            callback=process_frame
        )
        print(f"Successfully saved to {target_path}")
    except Exception as e:
        print(f"Failed to process video {source_path}: {e}")

def main():
    # Define the output directory
    output_dir = "out"
    os.makedirs(output_dir, exist_ok=True)

    # Define the tasks with correct paths
    tasks = [
        {
            "source": os.path.join("fall video", "Fall", "Raw_Video", "S_N_564.mp4"),
            "output": os.path.join(output_dir, "fall_result.mp4")
        },
        {
            "source": os.path.join("fall video", "No_Fall", "Raw_Video", "B_D_0112.mp4"),
            "output": os.path.join(output_dir, "no_fall_result.mp4")
        }
    ]

    for task in tasks:
        process_single_video(task["source"], task["output"])
 
if __name__ == "__main__":
    main()