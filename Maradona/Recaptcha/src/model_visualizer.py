import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

def save_learning_curves(history, save_path):
    """
    학습 과정(Loss, Accuracy)을 시각화하여 저장합니다 (판단 증거).
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    plt.plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    os.makedirs(Path(save_path).parent, exist_ok=True)
    plt.savefig(save_path)
    print(f"✓ Learning curves saved to {save_path}")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    """
    Confusion Matrix를 생성하고 저장합니다 (판단 증거).
    """
    from sklearn.metrics import classification_report
    
    # 모든 클래스가 포함되도록 labels 지정
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    plt.figure(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title('Confusion Matrix')
    
    os.makedirs(Path(save_path).parent, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to {save_path}")
    plt.close()
    
    # Classification Report 저장
    report_path = Path(save_path).parent / "classification_report.txt"
    report = classification_report(y_true, y_pred, target_names=classes, labels=range(len(classes)), zero_division=0)
    with open(report_path, "w") as f:
        f.write(report)
    print(f"✓ Classification report saved to {report_path}")

def generate_grad_cam(model, image_tensor, original_image, save_path):
    """
    Grad-CAM 시각화를 생성하여 저장합니다 (판단 증거: 모델이 무엇을 보고 판단했는지).
    """
    model.eval()
    
    # Hook 등록 및 예측
    target_layer = model.get_grad_cam_target_layer()
    if target_layer is None:
        print("⚠️ Warning: Could not find target layer for Grad-CAM. Skipping visualization.")
        return

    def hook_fn(module, input, output):
        output.register_hook(model.activations_hook)

    handle = target_layer.register_forward_hook(hook_fn)
    
    pred = model(image_tensor.unsqueeze(0))
    pred_class = pred.argmax(dim=1).item()
    
    # 하위 단계를 위해 Backprop
    pred[:, pred_class].backward()
    
    # 그래디언트와 활성화 맵 가져오기
    gradients = model.get_activations_gradient()
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    activations = model.get_activations(image_tensor.unsqueeze(0)).detach()
    
    # 활성화 맵 가중 합산
    for i in range(activations.size(1)):
        activations[:, i, :, :] *= pooled_gradients[i]
        
    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = np.maximum(heatmap.cpu().numpy(), 0)
    heatmap /= np.max(heatmap) + 1e-8
    
    # 히트맵 상향 샘플링 및 이미지 중첩
    heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    superimposed_img = heatmap_colored * 0.4 + original_image * 0.6
    superimposed_img = np.uint8(superimposed_img)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(superimposed_img)
    plt.title(f"Grad-CAM (Predicted: {pred_class})")
    plt.axis('off')
    
    plt.tight_layout()
    os.makedirs(Path(save_path).parent, exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    
    handle.remove()
    print(f"✓ Grad-CAM saved to {save_path}")
