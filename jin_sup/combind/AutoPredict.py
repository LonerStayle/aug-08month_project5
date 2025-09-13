
from image.image_model import load_image_model, transform_image
from PIL import Image
import torch
from text.text_model import load_text_model_embedder,clean_text,wav_to_text
# from sound.sound_model import load_sound_model,sound_to_image, sound_image_tf
from sound.sound_model import load_sound_model, wav_to_prosody_tensor
import speech_recognition as sr
import numpy as np

class AutoPredict:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
                
        text_model, embedder = load_text_model_embedder()
        self.text_model = text_model
        self.text_embedder = embedder
        
        self.image_model = load_image_model(self.device)        
        self.sound_model = load_sound_model()

    # def image_predict(self,img_file):
    #     img = Image.open(img_file).convert("RGB")
    #     x = transform_image(img).to(self.device)
        
    #     result = {}
    #     self.image_model.eval()
    #     with torch.no_grad():
    #         logits = self.image_model(x)
    #         probs = torch.softmax(logits, dim=1 )
    #         conf, pred = torch.max(probs, dim=1 )         
    #         result = {"pred":pred.item(), "conf":conf.item()}    
    #     return result
    
    # def sound_predict(self, sound_file):
        
    #     image_path = sound_to_image(sound_file)
    #     image_tf = sound_image_tf()
    #     img = Image.open(image_path).convert("RGB")
    #     x = image_tf(img).unsqueeze(0).to(self.device)
        
    #     self.sound_model.eval()
    #     score = 0
    #     pred = 0
    #     result = {}
    #     with torch.no_grad():
    #         logits = self.sound_model(x)
    #         probs = torch.softmax(logits,dim = 1)
    #         print(probs)
    #         conf, pred = torch.max(probs,dim=1)        
    #         result = {"pred":pred.item(), "conf":conf.item()}    
    #         # print(conf.item())
    #         # print(pred.item())
    #         happiness, surprise, neutral, fear, disgust, anger, sadness = probs.tolist()[0]
                
    #         POS = (happiness + surprise) / 2
    #         NEG_w = (0.4 * sadness + 0.3 * neutral + 0.1 * fear + 0.1 * disgust + 0.1 * anger) / 5
    #         score = NEG_w / (NEG_w + POS + 1e-8)
    #         # POS = happiness + surprise
    #         # NEG_w = (0.4 * sadness) + (0.3 * neutral) + (0.1 * fear) + (0.1 * disgust) + (0.1 * anger
    #         # score = NEG_w / (NEG_w + POS + 1e-8)
    #         pred = int(score > 0.5)
    #         # result = {"pred":pred, "conf":score}    
    #     return result


    def text_predict(self, file_path):
        text = wav_to_text(file_path)
        self.text_model.eval()
        
        with torch.no_grad():
            try:
                cleaned = clean_text(text)
            except:
                return {"pred": 0, "conf": 0}
            
            emb = self.text_embedder.encode([cleaned], device=self.device, convert_to_tensor=True)
            outputs = self.text_model(emb)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    
            label = int(probs[1] >= 0.5)   # 0 or 1
            conf = float(np.max(probs))    # 최대 확률값
            
            return {"pred": label, "conf": conf}
    
    def image_predict(self, image_path) -> tuple[str, float]:
        image = Image.open(image_path)
        image = image.convert("RGB")
        tf = transform_image()
        input_tensor = tf(image)
        input_batch = input_tensor.unsqueeze(0).to(self.device)
        model = self.image_model
        result = {}
        with torch.no_grad():
            output = model(input_batch)

        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
        print(probabilities)
        return {"pred":predicted_idx.item(), "conf":confidence.item()}

    def sound_predict(self, sound_file):
        
        wav_tensor = wav_to_prosody_tensor(sound_file)
        x = wav_tensor.unsqueeze(0).to(self.device)
        
        self.sound_model.eval()
        score = 0
        pred = 0
        result = {}
        with torch.no_grad():
            logits = self.sound_model(x)
            probs = torch.softmax(logits,dim = 1)
            _, pred = torch.max(probs,dim=1)        
            # result = {"pred":pred.item(), "conf":conf.item()}    
            # print(conf.item())
            # print(pred.item())
            happiness, surprise, neutral, fear, disgust, anger, sadness = probs.tolist()[0]
                
            POS = (happiness + surprise) / 2
            NEG_w = (0.4 * sadness + 0.3 * neutral + 0.1 * fear + 0.1 * disgust + 0.1 * anger) / 5
            score = NEG_w / (NEG_w + POS + 1e-8)
            # POS = happiness + surprise
            # NEG_w = (0.4 * sadness) + (0.3 * neutral) + (0.1 * fear) + (0.1 * disgust) + (0.1 * anger
            # score = NEG_w / (NEG_w + POS + 1e-8)
            pred = int(score > 0.5)
            result = {"pred":pred, "conf":score}    
        return result
    
    def combind_predict(self, image_file, sound_file):
        image_result = self.image_predict(image_file)
        sound_result = self.sound_predict(sound_file)
        text_result = self.text_predict(sound_file)
        
        # 수학 수식 써서 어쩌고 저쩌고 결과 만들기(최종 우울증 확률) 아래 result 는 예시 입니다.
        result = image_result + sound_result + text_result
        return result