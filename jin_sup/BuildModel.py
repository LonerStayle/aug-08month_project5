
from ModelType import ModelType
import torch.nn as nn
import torchvision.models as M


class BuildModel(nn.Module):
    def __init__(self, model_type:ModelType):
        super().__init__()
        output_feature = 2
    
        if model_type == ModelType.RESNET18:
            self.model = M.resnet18(M.ResNet18_Weights.IMAGENET1K_V1)
            self.model.fc = nn.Linear(self.model.fc.in_features, output_feature, bias=True)

        elif model_type == ModelType.CONVNEXT_SMALL:
            self.model = M.convnext_small(M.ConvNeXt_Small_Weights.IMAGENET1K_V1)
            in_features = self.model.classifier[2].in_features
            self.model.classifier[2] = nn.Linear(in_features, output_feature, bias=True)
        

        elif model_type == ModelType.VIT_B_16:
            self.model = M.vit_b_16(M.ViT_B_16_Weights.IMAGENET1K_V1)
            in_features = self.model.heads.head.in_features
            self.model.heads.head = nn.Linear(in_features, output_feature, bias=True)
        
        else : pass

        


