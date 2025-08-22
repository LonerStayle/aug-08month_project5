
from ModelType import ModelType
import torch.nn as nn
import torchvision.models as M


class BuildModel(nn.Module):
    def __init__(self, model_type:ModelType, output_feature:int = 2, weights = None):
        super().__init__()
        if weights is None:
            raise ValueError("weights 값이 None 입니다!")
        
        if model_type == ModelType.RESNET18:
            
            self.model = M.resnet18(weights)
            self.model.fc = nn.Linear(self.model.fc.in_features, output_feature, bias=True)

        elif model_type == ModelType.CONVNEXT_TINY:
            self.model = M.convnext_tiny(weights)
            in_features = self.model.classifier[2].in_features
            self.model.classifier[2] = nn.Linear(in_features, output_feature, bias=True)

        elif model_type == ModelType.CONVNEXT_SMALL:
            self.model = M.convnext_small(weights)
            in_features = self.model.classifier[2].in_features
            self.model.classifier[2] = nn.Linear(in_features, output_feature, bias=True)

        elif model_type == ModelType.CONVNEXT_BASE:
            self.model = M.convnext_small(weights)
            in_features = self.model.classifier[2].in_features
            self.model.classifier[2] = nn.Linear(in_features, output_feature, bias=True)

        elif model_type == ModelType.CONVNEXT_LARGE:
            self.model = M.convnext_large(weights)
            in_features = self.model.classifier[2].in_features
            self.model.classifier[2] = nn.Linear(in_features, output_feature, bias=True)
        

        elif model_type == ModelType.VIT_B_16:
            self.model = M.vit_b_16(weights)
            in_features = self.model.heads.head.in_features
            self.model.heads.head = nn.Linear(in_features, output_feature, bias=True)

        else : pass

        self.weights = weights
        self.preprocess = weights.transforms()

        


