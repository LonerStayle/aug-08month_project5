
from data_model.ModelType import ModelType
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
            self.model = M.convnext_base(weights)
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

            
        elif model_type == ModelType.EFFICIENT_NET_B0:
            self.model = M.efficientnet_b0(weights)
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_features, 7)
            
        else : pass

        self.weights = weights
        self.preprocess = weights.transforms()

    
    @staticmethod
    def get_model(model_type:ModelType, weights, output_feature, checkpoint, device):

        if model_type == ModelType.RESNET18:   
            model = M.resnet18(weights)
            model.fc = nn.Linear(model.fc.in_features, output_feature, bias=True)

        elif model_type == ModelType.CONVNEXT_TINY:
            model = M.convnext_tiny(weights)
            in_features = model.classifier[2].in_features
            model.classifier[2] = nn.Linear(in_features, output_feature, bias=True)

        elif model_type == ModelType.CONVNEXT_SMALL:
            model = M.convnext_small(M.ConvNeXt_Small_Weights.IMAGENET1K_V1)
            in_features = model.classifier[2].in_features
            model.classifier[2] = nn.Linear(in_features, 7, bias=True)

        elif model_type == ModelType.CONVNEXT_BASE:
            model = M.convnext_base(weights)
            in_features = model.classifier[2].in_features
            model.classifier[2] = nn.Linear(in_features, output_feature, bias=True)

        elif model_type == ModelType.CONVNEXT_LARGE:
            model = M.convnext_large(weights)
            in_features = model.classifier[2].in_features
            model.classifier[2] = nn.Linear(in_features, output_feature, bias=True)

        elif model_type == ModelType.VIT_B_16:
            model = M.vit_b_16(weights)
            in_features = model.heads.head.in_features
            model.heads.head = nn.Linear(in_features, output_feature, bias=True)

        elif model_type == ModelType.EFFICIENT_NET_B0:
            model = M.efficientnet_b0(weights)
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, output_feature)

        else: pass 

        model.load_state_dict(checkpoint)
        model.to(device)
        return model 
        


