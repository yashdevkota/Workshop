import torch
import torch.nn as nn
from torchvision import models

class RecaptchaCNN(nn.Module):
    """
    reCAPTCHA 이미지 분류를 위한 CNN 모델.
    ResNet, MobileNet, EfficientNet 등을 지원하며 Grad-CAM을 고려한 설계.
    """
    def __init__(self, num_classes, architecture="resnet18", pretrained=True, dropout=0.5):
        super(RecaptchaCNN, self).__init__()
        
        self.architecture = architecture
        
        if architecture == "resnet18":
            if hasattr(models, 'ResNet18_Weights'):
                weights = models.ResNet18_Weights.DEFAULT if pretrained else None
                self.model = models.resnet18(weights=weights)
            else:
                self.model = models.resnet18(pretrained=pretrained)
                
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(num_ftrs, num_classes)
            )
            
        elif architecture == "mobilenet_v2":
            if hasattr(models, 'MobileNet_V2_Weights'):
                weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
                self.model = models.mobilenet_v2(weights=weights)
            else:
                self.model = models.mobilenet_v2(pretrained=pretrained)
                
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(num_ftrs, num_classes)
            )

        elif architecture.startswith("efficientnet_b"):
            variant = architecture.split("_")[1] # b0, b1, etc.
            model_fn = getattr(models, architecture)
            
            if hasattr(models, f'EfficientNet_{variant.upper()}_Weights'):
                weights = getattr(models, f'EfficientNet_{variant.upper()}_Weights').DEFAULT if pretrained else None
                self.model = model_fn(weights=weights)
            else:
                self.model = model_fn(pretrained=pretrained)
                
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(num_ftrs, num_classes)
            )
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        return self.model(x)

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        """Grad-CAM을 위해 마지막 컨볼루션 레이어 특징맵 추출"""
        if self.architecture == "resnet18":
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)
            return x
        elif self.architecture.startswith("efficientnet_b") or self.architecture == "mobilenet_v2":
            return self.model.features(x)
        return None

    def get_grad_cam_target_layer(self):
        """Grad-CAM을 위한 후크(hook) 대상 레이어 반환"""
        if self.architecture == "resnet18":
            return self.model.layer4
        elif self.architecture.startswith("efficientnet_b"):
            return self.model.features  # 또는 구체적으로 self.model.features[-1]
        elif self.architecture == "mobilenet_v2":
            return self.model.features
        return None

def get_model(config, num_classes):
    model_config = config['cnn']['model']
    return RecaptchaCNN(
        num_classes=num_classes,
        architecture=model_config['architecture'],
        pretrained=model_config['pretrained'],
        dropout=model_config['dropout']
    )
