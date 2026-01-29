import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os

class RecaptchaDataset(Dataset):
    """
    reCAPTCHA 이미지 분류를 위한 PyTorch Dataset 클래스.
    """
    def __init__(self, root_dir, transform=None, label_mapping=None, exclude_classes=None):
        """
        Args:
            root_dir (str or Path): 전처리된 이미지가 있는 디렉토리 경로 (data/processed/resized 등)
            transform (callable, optional): 이미지에 적용할 torchvision transforms
            label_mapping (dict, optional): 클래스 이름과 숫자의 매핑
            exclude_classes (list, optional): 제외할 클래스 이름 리스트
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.exclude_classes = exclude_classes if exclude_classes else []
        self.image_paths = []
        self.labels = []
        
        # 클래스 탐색 및 이름 정규화 매핑
        if label_mapping is None:
            image_dirs = set()
            for img_path in self.root_dir.rglob("*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    image_dirs.add(img_path.parent)
            
            # 클래스 이름 정규화 (유사한 레이블 통합)
            raw_classes = sorted([d.name for d in image_dirs])
            self.label_mapping = {}
            temp_classes = []
            
            for cls_name in raw_classes:
                # 제외 클래스 건너뛰기
                if cls_name in self.exclude_classes:
                    continue
                    
                # 'Traffic Light'와 'TLight'를 'Traffic Light'로 통합
                norm_name = "Traffic Light" if cls_name in ["TLight", "Traffic Light"] else cls_name
                
                if norm_name in self.exclude_classes:
                    continue

                if norm_name not in temp_classes:
                    temp_classes.append(norm_name)
                
                self.label_mapping[cls_name] = temp_classes.index(norm_name)
            
            self.classes = temp_classes
            
            # 이미지 경로와 레이블 매핑
            for img_dir in image_dirs:
                class_name = img_dir.name
                if class_name not in self.label_mapping: # 제외된 클래스인 경우 건너뜀
                    continue
                    
                mapped_label = self.label_mapping[class_name]
                for img_path in img_dir.glob("*"):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        self.image_paths.append(img_path)
                        self.labels.append(mapped_label)
        else:
            self.label_mapping = label_mapping
            self.classes = sorted(list(set(label_mapping.values()))) # 중복 제거된 실제 클래스 리스트
            
            # 전달받은 매핑을 사용하여 데이터 로드
            for img_path in self.root_dir.rglob("*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    dir_name = img_path.parent.name
                    if dir_name in self.label_mapping and dir_name not in self.exclude_classes:
                        self.image_paths.append(img_path)
                        self.labels.append(self.label_mapping[dir_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def get_transforms(config):
    """
    설정 파일에 따른 이미지 변환 정의
    """
    aug_config = config['cnn']['augmentation']
    resize_size = tuple(aug_config['resize'])
    
    train_transform = transforms.Compose([
        transforms.Resize(resize_size),
        # 1. 기하학적 변형 (회전, 이동, 스케일링, 전단)
        transforms.RandomAffine(
            degrees=aug_config['rotation'],
            translate=(0.1, 0.1),
            scale=(0.8, 1.2),
            shear=10,
            interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3), # 원근 변형 추가
        transforms.RandomHorizontalFlip(p=0.5 if aug_config['horizontal_flip'] else 0),
        
        # 2. 색상 및 질감 변형
        transforms.ColorJitter(
            brightness=aug_config['brightness'], 
            contrast=aug_config['contrast'],
            saturation=0.2,
            hue=0.1
        ),
        transforms.RandomGrayscale(p=0.1),  # 10% 확률로 흑백 전환
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0))
        ], p=0.3),
        
        # 3. 필수 변환 및 강력한 규제 (RandomErasing)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0) # Cutout 효과
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def visualize_augmentations(dataset, save_path, num_samples=5):
    """
    데이터 증강이 적용된 샘플을 시각화하여 저장합니다 (사용자 요청 증거).
    """
    plt.figure(figsize=(15, 3 * num_samples))
    
    for i in range(num_samples):
        # 동일한 인덱스를 여러 번 변형해서 보여줌
        idx = np.random.randint(len(dataset))
        img_path = dataset.image_paths[idx]
        orig_img = Image.open(img_path).convert('RGB')
        label_name = dataset.classes[dataset.labels[idx]]
        
        # 1. 원본
        plt.subplot(num_samples, 4, i * 4 + 1)
        plt.imshow(orig_img)
        if i == 0: plt.title("Original")
        plt.axis('off')
        plt.ylabel(label_name)
        
        # 2-4. 증강 적용 결과들
        for j in range(2, 5):
            augmented_img, _ = dataset[idx]
            # 역정규화 (시각화용)
            img_np = augmented_img.permute(1, 2, 0).numpy()
            img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_np = np.clip(img_np, 0, 1)
            
            plt.subplot(num_samples, 4, i * 4 + j)
            plt.imshow(img_np)
            if i == 0: plt.title(f"Augmented {j-1}")
            plt.axis('off')
            
    plt.tight_layout()
    os.makedirs(Path(save_path).parent, exist_ok=True)
    plt.savefig(save_path)
    print(f"✓ Augmentation samples saved to {save_path}")
    plt.close()
