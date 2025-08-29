
import torch
import os

class Settings:
    """애플리케이션의 모든 설정을 관리합니다."""

    if os.name == "posix":
        MODEL_PATH: str = "/home/wanted-1/PotenupWorkspace/aug-project5/hwa_in/model/best_model_epoch15_f1_0.7535.pth"
    else : 
        MODEL_PATH: str = "C:\\PythonProject\\aug-08month_project5\\jin_sup\\model\\hwa_in_convnext_base_best_f1_0.7556.pth"
    CLASS_NAMES: dict = {
        0: "Non-Depressed",
        1: "Depressed"
    }


settings = Settings()