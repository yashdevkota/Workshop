# Google reCAPTCHA Image Classifer: From ML to Deep Learning

This document provides a report on the development of an AI model to classify Google reCAPTCHA v2 image tiles (e.g., traffic lights, cars, crosswalks). It details the technical evolution from machine learning approaches to a deep learning solution, achieving 98.78% accuracy through a Data-Centric AI methodology.

---

## Part 1. Exploratory Data Analysis & Preprocessing

The foundation of the project involved understanding the data distribution and establishing a preprocessing pipeline to prepare images for feature extraction and model training.

### 1-1. Dataset Analysis
Initial analysis revealed variation in raw data.

![Dataset Size Analysis](./data/visualization/preprocessing/analysis_v2/specific_sizes.png)
*Image size distribution*

- **Issue**: The dataset contained images of both 100x100 and 120x120 pixels.
- **Solution**: A uniform resolution of 112x112 was established.
- **Rationale**: 112px is practical for the dataset average (107.4px) and allows HOG (Histogram of Oriented Gradients) feature extraction with an integer split of 14 cells (8x8 pixels each).

![Resize Comparison](./data/visualization/resize/resize_comparison.png)
*Resizing comparison - unifying dimensions*

### 1-2. Preprocessing Experiments
Various techniques were evaluated to enhance image quality and feature saliency.

#### A. Noise Reduction
We compared Gaussian Blur and Bilateral Filtering to remove sensor noise while preserving edges.

![Noise Reduction Comparison](./data/visualization/preprocessing/noise_reduction_v2/noise_reduction/noise_reduction_comparison.png)

- **Result**: Bilateral Filtering is theoretically advantageous for edge preservation but more computationally expensive. Given the low resolution (112px), the difference was minimal, leading to the use of Gaussian Blur.

#### B. Contrast Enhancement (CLAHE)
Contrast Limited Adaptive Histogram Equalization (CLAHE) was used to recover details from low-contrast images.

![CLAHE Comparison](./data/visualization/clahe_v2/clahe/clahe_comparison.png)
*Left: Original, Right: CLAHE Applied - Outlines improved*

#### C. Color Space Conversion
We explored alternative color spaces to identify informative features.

![Color Space](./data/visualization/color_space_v2/color_space_conversions.png)
*Evaluation of YCrCb and HSV color spaces*

---

## Part 2. Machine Learning Approach

Before using deep learning, we attempted classification using features and machine learning models.

### 2-1. Feature Engineering
We extracted a combination of HOG (Shape), Color Histograms (Color), and LBP (Texture) features.

**Feature Distribution and Correlation:**
![Feature Distribution](./data/visualization/feature_extraction/combined/train_feature_distribution.png)
![Feature Correlation Heatmap](./data/visualization/feature_extraction/combined/train_feature_heatmap.png)

- Correlation analysis showed dependencies between features, suggesting non-linear boundaries.

### 2-2. Dimensionality Reduction (PCA Analysis)
Principal Component Analysis was used to reduce the extracted features.

![PCA Scree Plot](./data/visualization/feature_extraction/combined/pca_scree_plot.png)
![PCA 2D Projection](./data/visualization/feature_extraction/combined/train_pca_analysis.png)

- **Finding**: The first principal component (PC1) explained 6.9% of the total variance.
- **Interpretation**: This indicated that hand-crafted features could not capture the dataset complexity in a low-dimensional space.

### 2-3. Random Forest Performance
![Random Forest CV](./data/visualization/training/random_forest_combined_pca/cv_results.png)
- **Bottleneck**: Cross-validation accuracy reached the mid-70% range. This confirmed the limitations of human-defined features for this task.

---

## Part 3. Deep Learning (CNN)

We transitioned to a Convolutional Neural Network (EfficientNet). Several iterations were used to reach the final performance.

### 3-1. Phase 1: Preprocessing Impact
Our initial CNN attempt used preprocessed images (Denoised + CLAHE).

**Confusion Matrix (Initial):**
![V2 Confusion Matrix](./data/visualization/cnn/v2/confusion_matrix.png)
- The model had difficulty distinguishing trucks from buses and bridges from roads.
- **Root Cause**: Over-smoothing. Excessive preprocessing removed fine textures used as discriminative cues.

**Learning Curve (Initial):**
![V2 Learning Curve](./data/visualization/cnn/v2/learning_curves.png)
- Loss fluctuations and stagnation in validation accuracy were observed.

### 3-2. Phase 2: Data-Centric AI Methodology
We focused on data quality to improve performance.

1. **Cleaning**: Removed duplicate images to prevent data leakage.
2. **Label Merging**: Consolidated labels (e.g., 'TLight' into 'Traffic Light').
3. **Class Imbalance (Weighted Sampling)**:
   - **Problem**: Disparity between classes (e.g., 1200 'Bus' vs 44 'Motorcycle').
   - **Solution**: Implemented `WeightedRandomSampler` to oversample minority classes during training.

### 3-3. Phase 3: Augmentation
We used augmentation to increase model robustness.

![V3 Augmentation](./data/visualization/cnn/v3/aug_samples.png)
*Samples of Cutout and Perspective Distortion used*

---

## Part 4. Final Results

The V3 model, using a data-centric approach, achieved the following results.

### 4-1. Training Metrics
![V3 Learning Curve](./data/visualization/cnn/v3/learning_curves.png)
- **Final Accuracy**: 98.78%
- **Loss**: Convergence reached

### 4-2. Final Confusion Matrix
![V3 Confusion Matrix](./data/visualization/cnn/v3/confusion_matrix.png)
- Accuracy improved across categories, with previous errors reduced.

### 4-3. Model Interpretability (Grad-CAM)
We verified the visual features the model uses for decisions.

**Case 1: Car**
![GradCAM Car](./data/visualization/cnn/v3/grad_cam_3_Car.png)
*Model focuses on the wheels and silhouette*

**Case 2: Crosswalk**
![GradCAM Crosswalk](./data/visualization/cnn/v3/grad_cam_0_Crosswalk.png)
*Model identifies repeating diagonal stripe patterns*

**Case 3: Palm Tree**
![GradCAM Palm](./data/visualization/cnn/v3/grad_cam_1_Palm.png)
*Model prioritizes the texture of the leaves*

---

## Conclusion

Key insights include:

1. **Preprocessing Paradox**: Techniques for traditional ML can hinder CNNs.
2. **Data Quality**: Data cleaning and handling imbalance improved performance.
3. **Explainability**: Grad-CAM provides visual verification for model decisions.
