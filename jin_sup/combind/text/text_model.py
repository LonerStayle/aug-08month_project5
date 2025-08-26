import torch
import torch.nn as nn


class TextModel(nn.Module):
    """
    텍스트 임베딩을 위한 간단한 신경망 분류기.
    """
    def __init__(self, hidden_dim=256, num_classes=2):
        super(TextModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1024, hidden_dim), 
            nn.ReLU(), 
            nn.Dropout(0.3), 
            nn.Linear(hidden_dim, num_classes) 
        )

    def forward(self, x):
        return self.fc(x)

import torch

def load_text_model():
    model = TextModel()
    # 모델 가중치 로드
    
    checkpoint = torch.load("C:\\PythonProject\\aug-08month_project5\\jin_sup\\model\\nan_hye_model_depression_classifier.pt", map_location=torch.device('cpu'))
    # state_dict만 로드 (전체 checkpoint인 경우)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    return model

