
from torchvision import models
import torch.nn as nn
import torch
class DepressionClassifier(nn.Module):
    """우울증 분류를 위한 ResNet50 기반 모델"""
    
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super(DepressionClassifier, self).__init__()
        
        # 사전 학습된 ResNet50 백본
        self.backbone = models.resnet50(pretrained=True)
        
        # 특징 추출기 (분류기 제외)
        self.features = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # 전역 평균 풀링
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 분류 헤드
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
        # Grad-CAM을 위한 훅 등록
        self.gradients = None
        self.activations = None
        
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x):
        # 특징 추출
        features = self.features(x)
        
        # Grad-CAM을 위한 활성화 저장
        if features.requires_grad:
            h = features.register_hook(self.activations_hook)
        self.activations = features
        
        # 전역 평균 풀링 및 분류
        pooled = self.global_avg_pool(features)
        flattened = torch.flatten(pooled, 1)
        output = self.classifier(flattened)
        
        return output
    
    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self):
        return self.activations
    

import torch

def load_image_model():
    model = DepressionClassifier()
    # 모델 가중치 로드
    
    checkpoint = torch.load("C:\\PythonProject\\aug-08month_project5\\jin_sup\\model\\hwa_in_model_epoch_001_f1_0.7009.pt", map_location=torch.device('cpu'))
    # state_dict만 로드 (전체 checkpoint인 경우)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    return model


import torchvision.transforms as T
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
def transform_image(image):
    """이미지 전처리 함수"""
    transform = T.Compose([
        T.Grayscale(3),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform(image).unsqueeze(0) # 배치 차원 추가
