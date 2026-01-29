import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import argparse
import sys
import yaml
from tqdm import tqdm
import os
import numpy as np

# 경로 추가
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.preprocess.utils import load_config
from src.train.dataset import RecaptchaDataset, get_transforms, visualize_augmentations
from src.models.cnn_model import get_model
from src.visualization_cnn import save_learning_curves, plot_confusion_matrix, generate_grad_cam

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, all_preds, all_labels

def main():
    parser = argparse.ArgumentParser(description='Train CNN for reCAPTCHA')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config.yaml')
    parser.add_argument('--dry-run', action='store_true', help='Run 1 epoch with small data for testing')
    parser.add_argument('--version', type=str, default='v3', help='Version name for this run (e.g., v1, v2, v3)')
    parser.add_argument('--evaluate-only', action='store_true', help='Skip training and only evaluate the best model')
    args = parser.parse_args()
    
    # 설정 로드
    project_root = Path(__file__).parent.parent.parent
    config = load_config(project_root / args.config)
    
    # 장치 설정
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        import os
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    else:
        device = torch.device("cpu")
    print(f"🚀 Using device: {device} (Version: {args.version})")
    
    # 데이터셋 로드 (v3부터는 config에서 지정한 processed_dir 내부를 직접 사용)
    processed_dir = project_root / config['data']['processed_dir']
    
    if not processed_dir.exists():
        # 하위 resized 폴더가 있는지 한 번 더 확인 (이전 버전 호환성)
        alt_path = processed_dir / "resized"
        if alt_path.exists():
            processed_dir = alt_path
        else:
            processed_dir = project_root / config['data']['raw_dir']
            print(f"⚠️ Processed directory not found. Using raw directory: {processed_dir}")
        
    print(f"📂 Loading data from: {processed_dir}")
    train_transform, val_transform = get_transforms(config)
    
    # v2: 'Other' 클래스 제외
    full_dataset = RecaptchaDataset(processed_dir, transform=train_transform, exclude_classes=['Other'])
    num_classes = len(full_dataset.classes)
    print(f"Loaded dataset with {len(full_dataset)} images and {num_classes} classes: {full_dataset.classes}")
    
    # 클래스 가중치 계산 (데이터 불균형 해소)
    labels = np.array(full_dataset.labels)
    class_sample_count = np.array([len(np.where(labels == t)[0]) for t in range(num_classes)])
    # 0으로 나누기 방지
    class_sample_count = np.where(class_sample_count == 0, 1, class_sample_count)
    weight = 1. / class_sample_count
    samples_weight = torch.from_numpy(weight).float().to(device)
    # 가중치 정규화
    samples_weight = samples_weight / samples_weight.sum() * num_classes
    print(f"⚖️ Applied class weights: {samples_weight.cpu().numpy()}")

    # 데이터 증강 시각화 (사용자 요청 증거)
    vis_dir = project_root / config['cnn']['visualization']['save_dir'] / args.version
    visualize_augmentations(full_dataset, vis_dir / "aug_samples.png")
    
    # 데이터 분할
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
    # Validation용은 transform 변경
    val_ds.dataset.transform = val_transform
    
    batch_size = config['cnn']['training']['batch_size']
    if args.dry_run:
        train_ds = torch.utils.data.Subset(train_ds, range(min(128, len(train_ds))))
        val_ds = torch.utils.data.Subset(val_ds, range(min(64, len(val_ds))))
        batch_size = 8
        config['cnn']['training']['epochs'] = 1
        print("💡 Dry-run mode: using small subset of data.")
        
    # v3: 메모리 부족 해결 (Batch 32 + Workers 2)
    # Mac 통합 메모리 부하를 줄이기 위해 worker와 batch를 하향 조정
    num_workers = 2 if device.type == 'mps' else 0
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=False, # MPS에서 pin_memory 미지원
        prefetch_factor=2 if num_workers > 0 else None
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    # 모델 정의
    model = get_model(config, num_classes).to(device)
    
    # 손실 함수 (클래스 가중치 + 라벨 스무딩) 및 옵티마이저 (가중치 감쇠)
    criterion = nn.CrossEntropyLoss(weight=samples_weight, label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=config['cnn']['training']['learning_rate'], weight_decay=1e-4)
    
    # 더 정교한 스케줄러: 성능 정체 시 LR 감소
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )
    
    # 학습 루프
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0
    patience_counter = 0
    early_stop_patience = 15  # v2: 15 에포크 동안 개선 없으면 종료 (기존 10)
    
    num_epochs = config['cnn']['training'].get('epochs', 100)
    if args.dry_run: num_epochs = 1

    # 학습 모드
    if not args.evaluate_only:
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device)
            
            # Scheduler Step (Validation Accuracy 기준)
            scheduler.step(val_acc)
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # 모델 저장 및 중간 결과 시각화
            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
                model_path = project_root / "models" / args.version / "cnn_best_model.pth"
                os.makedirs(model_path.parent, exist_ok=True)
                torch.save(model.state_dict(), model_path)
                print(f"⭐ New best model saved with Acc: {best_acc:.4f} to {model_path}")
                
                # Confusion Matrix & Classification Report 저장
                plot_confusion_matrix(val_labels, val_preds, full_dataset.classes, vis_dir / "confusion_matrix.png")
            else:
                patience_counter += 1
                
            # 학습 곡선 업데이트
            save_learning_curves(history, vis_dir / "learning_curves.png")
            
            # Early Stopping 체크
            if patience_counter >= early_stop_patience:
                print(f"🛑 Early stopping triggered after {epoch+1} epochs.")
                break
    else:
        print("\n🔍 Evaluation Mode: Skipping training...")
        
    # 최종 결과 분석 (Evaluation만 할 때도 실행됨)
    print("\nGenerating Grad-CAM samples (Focused on Bridge & Car)...")
    best_model_path = project_root / "models" / args.version / "cnn_best_model.pth"
    
    if not best_model_path.exists():
        print(f"❌ Error: Model file not found at {best_model_path}")
        return

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print(f"✅ Loaded best model from {best_model_path}")
    
    # 평가 모드일 경우 Confusion Matrix 다시 생성
    if args.evaluate_only:
        loss, acc, preds, labels = validate(model, val_loader, criterion, device)
        print(f"📊 Validation Set Evaluation - Loss: {loss:.4f}, Acc: {acc:.4f}")
        plot_confusion_matrix(labels, preds, full_dataset.classes, vis_dir / "confusion_matrix.png")
    
    # 특정 클래스 인덱스 찾기
    target_cls_indices = [i for i, cls in enumerate(full_dataset.classes) if cls in ["Bridge", "Car"]]
    
    num_cam_samples = config['cnn']['visualization']['num_samples']
    samples_found = 0
    
    # 섞어서 샘플링
    indices = np.random.permutation(len(val_ds))
    for idx in indices:
        if samples_found >= num_cam_samples: break
        
        image_tensor, label = val_ds[idx]
        
        # Bridge나 Car인 경우 우선적으로 시각화
        if label in target_cls_indices or (samples_found < 2): # 최소 2개까지는 일반 샘플도 허용
            image_tensor = image_tensor.to(device)
            orig_img_np = image_tensor.cpu().permute(1, 2, 0).numpy()
            orig_img_np = orig_img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            orig_img_np = np.uint8(np.clip(orig_img_np, 0, 1) * 255)
            
            cls_name = full_dataset.classes[label]
            generate_grad_cam(model, image_tensor, orig_img_np, vis_dir / f"grad_cam_{samples_found}_{cls_name}.png")
            samples_found += 1

        
    print("\n✅ Training and Visualization Complete!")
    print(f"All evidence images are saved in: {vis_dir}")

if __name__ == "__main__":
    main()
