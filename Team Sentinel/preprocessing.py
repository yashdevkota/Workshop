
import os
import cv2
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path
import random

def add_noise(image, percentage=0.0168):
    """
    Adds salt-and-pepper like noise to up to 'percentage' of pixels.
    """
    row, col = image.shape
    num_pixels = row * col
    # Determine number of pixels to alter (random up to percentage)
    actual_percent = random.uniform(0, percentage)
    number_of_pixels = int(actual_percent * num_pixels)
    
    output = np.copy(image)
    
    # Randomly pick coordinates
    for _ in range(number_of_pixels):
        y_coord = random.randint(0, row - 1)
        x_coord = random.randint(0, col - 1)
        # Randomly choose white or black for salt/pepper
        output[y_coord, x_coord] = 255 if random.random() > 0.5 else 0
        
    return output

def apply_blur(image, max_ksize=2.5):
    """
    Applies Gaussian Blur. 
    '2.5px' roughly translates to a kernel size or sigma. 
    We'll interpret 'Up to 2.5px' as a max sigma or varying kernel size.
    """
    # Randomly decide whether to blur or not, or just always apply random amount?
    # Usually augmentation has a probability. We'll applying a random amount up to max.
    
    # Random sigma from 0 to 2.5
    sigma = random.uniform(0, max_ksize)
    if sigma > 0.1:
        # Kernel size roughly 6*sigma, must be odd
        k = int(6 * sigma) 
        if k % 2 == 0: k += 1
        if k < 3: k = 3
        return cv2.GaussianBlur(image, (k, k), sigma)
    return image

def augment_image(image):
    h, w = image.shape[:2]
    
    # 1. Flip: Horizontal
    if random.random() < 0.5:
        image = cv2.flip(image, 1) # 1 is horizontal
        
    # 2. 90° Rotate: Clockwise, Counter-Clockwise, Upside Down
    # We interpret this as a random choice among 0, 90, 180, 270
    rotate_code = random.choice([None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180])
    if rotate_code is not None:
        image = cv2.rotate(image, rotate_code)
        
    # 3. Affine Transforms: Rotation, Shear, Zoom
    # Rotation: -15 to +15
    angle = random.uniform(-15, 15)
    
    # Shear: +/- 15 degrees. 
    # Shear in OpenCV is usually done via affine matrix construction
    shear_x_deg = random.uniform(-15, 15)
    shear_y_deg = random.uniform(-15, 15)
    
    # Zoom: 0% to 30%. Implies scaling up effectively (viewing less of the image)
    # Scale factor: 1.0 to ~1.42 (approx 1/(1-0.3)). 
    # Let's say zoom amount z corresponds to crop factor.
    zoom_pct = random.uniform(0.0, 0.30)
    scale = 1.0 + zoom_pct # Simple zoom interpretation: scale > 1
    
    # Construct combined affine matrix
    # Center of rotation/scale
    center = (w // 2, h // 2)
    
    # Get rotation matrix (handles rotation + scale)
    M_rot = cv2.getRotationMatrix2D(center, angle, scale)
    
    # Add shear. 
    # Shear matrix: [1, tan(sx), 0; tan(sy), 1, 0]
    # We need to compose this with M_rot.
    tangent_x = np.tan(np.deg2rad(shear_x_deg))
    tangent_y = np.tan(np.deg2rad(shear_y_deg))
    
    M_shear = np.float32([
        [1, tangent_x, 0],
        [tangent_y, 1, 0]
    ])
    
    # To combine, we need 3x3 matrices. 
    M_rot_3x3 = np.vstack([M_rot, [0, 0, 1]])
    M_shear_3x3 = np.vstack([M_shear, [0, 0, 1]])
    
    # Adjust shear to happen around center? Standard shear is from origin.
    # To shear from center, T_inv * Shear * T.
    # For simplicity, we'll just apply shear then rotate/scale or vice versa.
    # Let's just multiply them.
    
    M_combined = np.dot(M_rot_3x3, M_shear_3x3)
    
    # Extract top 2 rows
    M_final = M_combined[:2, :]
    
    image = cv2.warpAffine(image, M_final, (w, h), borderMode=cv2.BORDER_REFLECT101)
    
    # 4. Blur
    image = apply_blur(image, max_ksize=2.5)
    
    # 5. Noise
    image = add_noise(image, percentage=0.0168)
    
    return image

def preprocess_image(image_path, output_path_base):
    try:
        # 1. Load image and 2. Auto-Orient
        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img)

        # 3. Resize: Stretch to 512x512
        img = img.resize((512, 512))

        # Convert to numpy array
        img_np = np.array(img)

        # 4. Grayscale
        if len(img_np.shape) == 3:
            if img_np.shape[2] == 4:
                img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGBA2GRAY)
            else:
                img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img_np

        # 5. Auto-Adjust Contrast: CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_processed = clahe.apply(img_gray)

        # Ensure output directory exists [moved out to be safe]
        os.makedirs(os.path.dirname(output_path_base), exist_ok=True)

        # 6. Generate 3 Augmented Versions
        # We split the filename to add suffixes
        path_obj = Path(output_path_base)
        stem = path_obj.stem
        suffix = path_obj.suffix
        parent = path_obj.parent
        
        for i in range(3):
            # Apply augmentation
            aug_img = augment_image(img_processed)
            
            # Save
            out_name = f"{stem}_aug_{i+1}{suffix}"
            final_path = parent / out_name
            cv2.imwrite(str(final_path), aug_img)

    except Exception as e:
        print(f"Failed to process {image_path}: {e}")

def main():
    base_src_dir = Path("dataset/images")
    base_dst_dir = Path("dataset/images_preprocessed")

    print(f"Starting preprocessing and augmentation from {base_src_dir} to {base_dst_dir}...")
    
    count = 0
    for root, dirs, files in os.walk(base_src_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                src_path = os.path.join(root, file)
                
                # Determine relative path
                rel_path = os.path.relpath(src_path, base_src_dir)
                dst_path = os.path.join(base_dst_dir, rel_path)

                preprocess_image(src_path, dst_path)
                count += 1
                if count % 50 == 0:
                    print(f"Processed {count} input images -> {count * 3} output images...")

    print(f"Finished. Processed {count} images -> {count * 3} augmented images saved to {base_dst_dir}")

if __name__ == "__main__":
    main()
