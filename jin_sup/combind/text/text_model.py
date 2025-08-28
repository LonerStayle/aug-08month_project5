import torch
import re
from sentence_transformers import SentenceTransformer
import os
import speech_recognition as sr
import torch.nn as nn


class LogisticRegression(nn.Module):
    """
    텍스트 임베딩 기반 로지스틱 회귀 모델
    """
    def __init__(self, input_dim, num_classes=2):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if os.name == "posix":
    MODEL_PATH = r"C:\PythonProject\aug-08month_project5\jin_sup\model\nan_hye_depression_logistic_regression_jhgan_ko_sbert_multitask.pt"
else :
    MODEL_PATH = r"C:\PythonProject\aug-08month_project5\jin_sup\model\nan_hye_depression_logistic_regression_jhgan_ko_sbert_multitask.pt"

def clean_text(text):
    text = re.sub(r'\([가-힣\s]+\)', '', text)
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'[^가-힣a-zA-Z0-9.,?!"\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def load_text_model_embedder():
    embedder = SentenceTransformer("jhgan/ko-sbert-multitask", device=device)
    input_dim = embedder.get_sentence_embedding_dimension()
    model = LogisticRegression(input_dim).to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        raise FileNotFoundError(f"모델 파일이 없습니다: {MODEL_PATH}")
    return model, embedder

def predict(text, model, embedder, threshold=0.5):
    model.eval()
    with torch.no_grad():
        cleaned = clean_text(text)
        if not cleaned:
            return "입력 없음", None
        emb = embedder.encode([cleaned], device=device, convert_to_tensor=True)
        outputs = model(emb)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        label = "우울증(1)" if probs[1] >= threshold else "정상(0)"
        return label, probs

def wav_to_text(file_path):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(file_path) as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio, language="ko-KR")
        print(f"파일에서 변환된 텍스트: {text}")
        return text
    except sr.UnknownValueError:
        print("음성을 인식할 수 없습니다.")
        return None
    except sr.RequestError as e:
        print(f"음성 인식 서비스 오류: {e}")
        return None
