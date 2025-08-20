
from enum import Enum

class ModelType(str, Enum):
    RESNET18 = "RESNET18"
    CONVNEXT_SMALL = "CONVNEXT_SMALL"
    VIT_B_16 = "VIT_B_16"