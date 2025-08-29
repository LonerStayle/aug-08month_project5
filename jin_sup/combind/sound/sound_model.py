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
import torchaudio
import torch.nn.functional as F


def load_sound_model():

    if os.name == "posix":  # Linux, macOS
        path = "/home/wanted-1/PotenupWorkspace/aug-project5/jin_sup/model/model_ConvNeXt_Small_Weights.IMAGENET1K_V1___08-29_02-24-17.pth"
    elif os.name == "nt":  # Windows
        path = "C:\\PythonProject\\aug-08month_project5\\jin_sup\\model\\model_ConvNeXt_Small_Weights.IMAGENET1K_V1___08-29_02-24-17.pth"

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(path, map_location="cpu")
    model = BuildModel.get_model(ModelType.CONVNEXT_SMALL,
                             M.ConvNeXt_Small_Weights.IMAGENET1K_V1,
                               7 , checkpoint, device)
    
    return model

def sound_to_image(file):

    # audio = AudioSegment.from_file(file, format="m4a")
    # buf = io.BytesIO()
    # audio.export(buf, format="wav")
    # buf.seek(0)

    temp_y, sr =librosa.load(file, sr=SoundVariable.HZ)
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


device = "cuda:0" if torch.cuda.is_available() else "cpu"
SR = 16000
N_FFT = 1024
HOP   = 256               
N_MELS = 64               
TOP_DB = 70               
FMIN_MEL, FMAX_MEL = 50.0, 8000.0
FMIN_F0,  FMAX_F0  = 50.0, 800.0     
TARGET = 224

mel_spec = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
    f_min=FMIN_MEL, f_max=FMAX_MEL, power=2.0, norm="slaney", mel_scale="htk"
).to(device)
to_db = torchaudio.transforms.AmplitudeToDB(top_db=TOP_DB).to(device)



def _avg_pool_2d(x: torch.Tensor, k_freq: int, k_time: int) -> torch.Tensor:
    if x.dim() == 2: x = x.unsqueeze(0)
    x = F.avg_pool2d(x, kernel_size=(k_freq, k_time), stride=(k_freq, k_time), ceil_mode=True)
    return x

def _resize_to(x: torch.Tensor, H: int, T: int) -> torch.Tensor:
    # x: [1, h, t]
    x = F.interpolate(x.unsqueeze(0), size=(H,T), mode="bilinear", align_corners=False).squeeze(0)
    return x  

def _f0_to_band(y_pos: np.ndarray, H: int, sigma: float = 1.5) -> np.ndarray:
    # y_pos: [T] (0..H-1 위치), 가우시안 띠
    yy = np.arange(H, dtype=np.float32)[:, None]
    dist2 = (yy - y_pos[None,:])**2
    band = np.exp(-dist2/(2*sigma**2))
    return (band - band.min()) / (band.max()-band.min() + 1e-8)



def wav_to_prosody_tensor(path: str) -> torch.Tensor:
    # 0) 로드
    y, sr = librosa.load(path, sr=SR, mono=True)
    y_t = torch.from_numpy(y).float().to(device)

    # 1) 멜 (저해상도 + 평활화)
    S = mel_spec(y_t)                            
    Sdb = to_db(S).clamp_(-TOP_DB, 0.0)
    Smel = (Sdb + TOP_DB) / TOP_DB               
    Smel = Smel.unsqueeze(0)                     

    #   음소 내용 희석: 주파수/시간 풀링으로 블러 + 다시 원래 크기로 리사이즈
    #   (k_freq, k_time)는 데이터에 맞춰 조정 가능. 값이 클수록 단어 정보 더 흐림.
    k_freq, k_time = 4, 4
    Smel_low = _avg_pool_2d(Smel, k_freq, k_time)          
    Smel_smooth = _resize_to(Smel_low, Smel.shape[1], Smel.shape[2])  
    ch_mel = Smel_smooth  

    # 2) 피치(F0) → 띠
    f0, vflag, vconf = librosa.pyin(y, fmin=FMIN_F0, fmax=FMAX_F0, sr=sr, frame_length=N_FFT, hop_length=HOP)
    T0 = len(f0)
    valid = ~np.isnan(f0)
    if valid.sum() < 3:
        ch_pitch = torch.zeros_like(ch_mel)
        vprob = torch.zeros(Smel.shape[-1], dtype=torch.float32, device=device)
    else:
        idx = np.arange(T0)
        f0i = np.interp(idx, idx[valid], f0[valid]).astype(np.float32)
        ylog = np.log(np.clip(f0i, FMIN_F0, FMAX_F0))
        y_pos = (ylog - np.log(FMIN_F0)) / (np.log(FMAX_F0)-np.log(FMIN_F0)) * (N_MELS-1)

        # 시간 정렬: 멜 T와 동일하게
        t_src = np.linspace(0,1,T0, dtype=np.float32)
        t_dst = np.linspace(0,1,Smel.shape[-1], dtype=np.float32)
        y_pos = np.interp(t_dst, t_src, y_pos)

        band = _f0_to_band(y_pos, N_MELS, sigma=1.5)       # [H,T] 0..1
        ch_pitch = torch.from_numpy(band).unsqueeze(0).float().to(device)
        vprob = np.nan_to_num(vconf, nan=0.0)
        vprob = np.interp(t_dst, t_src, vprob).astype(np.float32)

    # 3) RMS 에너지 → 밴드
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP)[0]  # [T_rms]
    
    # 멜 T와 정렬
    t_src2 = np.linspace(0,1,len(rms), dtype=np.float32)
    t_dst2 = np.linspace(0,1,Smel.shape[-1], dtype=np.float32)
    rms_t = np.interp(t_dst2, t_src2, rms)

    # 0..1 정규화 (강세/리듬만)
    rms_t = (rms_t - rms_t.min()) / (rms_t.max() - rms_t.min() + 1e-8)
    ch_rms = torch.from_numpy(np.tile(rms_t[None,:], (N_MELS,1))).unsqueeze(0).float().to(device)  # [1,H,T]

    # 4) 스택 & 리사이즈 → [3, TARGET, TARGET], 스케일 [-1,1]
    x = torch.cat([ch_mel, ch_pitch, ch_rms], dim=0)   # [3,H,T], 0..1
    x = F.interpolate(x.unsqueeze(0), size=(TARGET,TARGET), mode="bilinear", align_corners=False).squeeze(0)
    x = x*2 - 1  
    return x  

# def sound_to_image(file):

#     audio = AudioSegment.from_file(file, format="m4a")
#     buf = io.BytesIO()
#     audio.export(buf, format="wav")
#     buf.seek(0)


#     temp_y, sr =librosa.load(buf, sr=SoundVariable.HZ)
#     y, _ = librosa.effects.trim(temp_y, top_db=SoundVariable.TOP_DB)
#     S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=SoundVariable.N_MELS, fmax=SoundVariable.F_MAX)
#     S_dB = librosa.power_to_db(S, ref=np.max)

#     plt.figure(figsize=(SoundVariable.FIG_SIZE_W, SoundVariable.FIG_SIZE_H))
#     librosa.display.specshow(S_dB, sr=sr, fmax=SoundVariable.F_MAX)

#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     save_path = os.path.join(
#         'C:\\PythonProject\\aug-08month_project5\\jin_sup\\combind\\sound\\trans_img',
#         f"{timestamp}_mel.png"
#     )
#     plt.savefig(save_path)
#     # buf = io.BytesIO()
#     # plt.savefig(buf, format="png")
#     # buf.seek(0)
#     plt.close()

#     return save_path



# def sound_image_tf():
#     mean = ImageVariable.MEAN
#     std = ImageVariable.STD
#     norm = T.Normalize(mean=[mean, mean, mean], std=[std, std, std])       
        
#     tf = T.Compose([
#             TrimBorder(0),
#             T.Grayscale(3),
#             T.Resize(ImageVariable.IMAGE_SIZES, antialias=True),
#             T.CenterCrop(ImageVariable.IMAGE_SIZES),
#             T.ToTensor(),
#             norm,
#         ])
#     return tf



