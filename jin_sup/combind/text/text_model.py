import torch
import torch.nn as nn

import torch.nn as nn

# class TextModel(nn.Module):
#     """
#     텍스트 임베딩 기반 로지스틱 회귀 모델
#     """
#     def __init__(self, num_classes=2):
#         super(TextModel, self).__init__()
#         self.linear = nn.Linear(768, num_classes)

#     def forward(self, x):
#         return self.linear(x)
    
class TextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 2)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

import torch

def load_text_model():
    model = TextModel()
    # 모델 가중치 로드
    
    checkpoint = torch.load("C:\\PythonProject\\aug-08month_project5\\jin_sup\\model\\nan_hye_depression_mlp_earlystop.pth", map_location=torch.device('cpu'))
    # state_dict만 로드 (전체 checkpoint인 경우)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    return model

