import torch,os
from jin_sup.data_model.BuildModel import BuildModel
from jin_sup.data_model.ModelType import ModelType
import torchvision.models as M
from jin_sup.data_model.GlobalVariable import ImageVariable
from torchvision import transforms as T
from jin_sup.pre_process.img_preprocess import TrimBorder
import librosa
from jin_sup.data_model.GlobalVariable import SoundVariable
import numpy as np
import io
import matplotlib.pyplot as plt
import librosa, librosa.display
from datetime import datetime
from pydub import AudioSegment


def load_sound_model():

    if os.name == "posix":  # Linux, macOS
        path = "/home/wanted-1/PotenupWorkspace/aug-project5/jin_sup/model/model_ConvNeXt_Small_Weights.IMAGENET1K_V1___08-25_20-53-16.pth"
    elif os.name == "nt":  # Windows
        path = "C:\\PythonProject\\aug-08month_project5\\jin_sup\\model\\jin_sup/model/model_ConvNeXt_Small_Weights.IMAGENET1K_V1___08-25_20-53-16.pth"

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(path, map_location="cpu")
    model = BuildModel.get_model(ModelType.CONVNEXT_SMALL,
                             M.ConvNeXt_Small_Weights.IMAGENET1K_V1,
                               7 , checkpoint, device)
    
    return model

def sound_to_image(file):

    audio = AudioSegment.from_file(file, format="m4a")
    buf = io.BytesIO()
    audio.export(buf, format="wav")
    buf.seek(0)


    temp_y, sr =librosa.load(buf, sr=SoundVariable.HZ)
    y, _ = librosa.effects.trim(temp_y, top_db=SoundVariable.TOP_DB)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=SoundVariable.N_MELS, fmax=SoundVariable.F_MAX)
    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(SoundVariable.FIG_SIZE_W, SoundVariable.FIG_SIZE_H))
    librosa.display.specshow(S_dB, sr=sr, fmax=SoundVariable.F_MAX)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(
        'C:\\PythonProject\\aug-08month_project5\\jin_sup\\combind\\sound\\trans_img',
        f"{timestamp}_mel.png"
    )
    plt.savefig(save_path)
    # buf = io.BytesIO()
    # plt.savefig(buf, format="png")
    # buf.seek(0)
    plt.close()

    return save_path



def sound_image_tf():
    mean = ImageVariable.MEAN
    std = ImageVariable.STD
    norm = T.Normalize(mean=[mean, mean, mean], std=[std, std, std])       
        
    tf = T.Compose([
            TrimBorder(0),
            T.Grayscale(3),
            T.Resize(ImageVariable.IMAGE_SIZES, antialias=True),
            T.CenterCrop(ImageVariable.IMAGE_SIZES),
            T.ToTensor(),
            norm,
        ])
    return tf



