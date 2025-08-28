
from torchvision import models
import torch.nn as nn
import torch
# class DepressionClassifier(nn.Module):
#     """우울증 분류를 위한 ResNet50 기반 모델"""
    
#     def __init__(self, num_classes=2, dropout_rate=0.5):
#         super(DepressionClassifier, self).__init__()
        
#         # 사전 학습된 ResNet50 백본
#         self.backbone = models.resnet50(pretrained=True)
        
#         # 특징 추출기 (분류기 제외)
#         self.features = nn.Sequential(*list(self.backbone.children())[:-2])
        
#         # 전역 평균 풀링
#         self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
#         # 분류 헤드
#         self.classifier = nn.Sequential(
#             nn.Dropout(dropout_rate),
#             nn.Linear(2048, 512),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout_rate),
#             nn.Linear(512, num_classes)
#         )
        
#         # Grad-CAM을 위한 훅 등록
#         self.gradients = None
#         self.activations = None
        
#     def activations_hook(self, grad):
#         self.gradients = grad
        
#     def forward(self, x):
#         # 특징 추출
#         features = self.features(x)
        
#         # Grad-CAM을 위한 활성화 저장
#         if features.requires_grad:
#             h = features.register_hook(self.activations_hook)
#         self.activations = features
        
#         # 전역 평균 풀링 및 분류
#         pooled = self.global_avg_pool(features)
#         flattened = torch.flatten(pooled, 1)
#         output = self.classifier(flattened)
        
#         return output
    
#     def get_activations_gradient(self):
#         return self.gradients
    
#     def get_activations(self):
#         return self.activations
    

# import torch

# def load_image_model():
#     model = DepressionClassifier()
#     # 모델 가중치 로드
    
#     checkpoint = torch.load("C:\\PythonProject\\aug-08month_project5\\jin_sup\\model\\hwa_in_convnext_base_best_f1_0.7556.pth", map_location=torch.device('cpu'))
#     # state_dict만 로드 (전체 checkpoint인 경우)
#     if 'model_state_dict' in checkpoint:
#         model.load_state_dict(checkpoint['model_state_dict'])
#     else:
#         model.load_state_dict(checkpoint)
#     return model


# import torchvision.transforms as T
# IMAGENET_MEAN = [0.485, 0.456, 0.406]
# IMAGENET_STD = [0.229, 0.224, 0.225]
# def transform_image(image):
#     """이미지 전처리 함수"""
#     transform = T.Compose([
#         T.Grayscale(3),
#         T.Resize((224, 224)),
#         T.ToTensor(),
#         T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
#     ])
#     return transform(image).unsqueeze(0) # 배치 차원 추가


import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import convnext_base, ConvNeXt_Base_Weights # ✨ [수정] ConvNeXt
from PIL import Image

# config 파일에서 설정값을 가져옵니다.
from image.config import settings

class DepressionClassifier(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        # ✨✨✨ [수정] Backbone을 ConvNeXt-Base로 교체 ✨✨✨
        weights = ConvNeXt_Base_Weights.IMAGENET1K_V1
        self.backbone = convnext_base(weights=weights)
        
        # ConvNeXt의 분류기는 'classifier'라는 이름의 Sequential 안에 있습니다.
        # 마지막 Linear 레이어의 in_features를 가져옵니다.
        in_features = self.backbone.classifier[-1].in_features
        # 원래 분류기의 마지막 레이어를 제거합니다.
        self.backbone.classifier[-1] = nn.Identity()

        # 2. Classifier 정의: '우울/비우울'만 판단할 "신입 전문가"를 새로 고용합니다.
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 2)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

    def freeze_backbone(self, freeze=True):
        # ConvNeXt의 파라미터를 순회하며 고정/해제
        for param in self.backbone.parameters():
            param.requires_grad = not freeze
        
        # 분류기 부분은 항상 학습 가능하도록 설정
        for param in self.classifier.parameters():
            param.requires_grad = True
                
def transform_image() -> transforms.Compose:
    """추론에 사용할 이미지 전처리기를 정의합니다."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def load_image_model(device) -> DepressionClassifier:
    """모델 구조를 가져와 학습된 가중치를 로드합니다."""
    try:
        model = DepressionClassifier()
        model.load_state_dict(torch.load(settings.MODEL_PATH, map_location=device))
        model.to(device)
        model.eval() # 평가 모드로 설정
        print(f"🚀 Model loaded successfully from {settings.MODEL_PATH}")
        return model
    except FileNotFoundError:
        raise RuntimeError(f"❌ Error: Model file not found at {settings.MODEL_PATH}")
    except Exception as e:
        raise RuntimeError(f"❌ Error loading model: {e}")

