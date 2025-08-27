
from image.image_model import load_image_model, transform_image
from PIL import Image
import torch
from text.text_model import load_text_model
from sound.sound_model import load_sound_model,sound_to_image, sound_image_tf
from sentence_transformers import SentenceTransformer


class AutoPredict:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.image_model = load_image_model()

        self.text_model = load_text_model()
        self.text_embedder = SentenceTransformer("jhgan/ko-sbert-multitask", device=self.device)
        
        self.sound_model = load_sound_model()

    def text_predict(self, text):
        self.text_embedding = self.text_embedder.encode([text], device=self.device, convert_to_numpy=True)
        self.text_embedding_tensor = torch.tensor(self.text_embedding, dtype=torch.float32).to(self.device)
        result = {}
        self.text_model.eval()
        with torch.no_grad():
            logits = self.text_model(self.text_embedding_tensor)
            probs = torch.softmax(logits,dim = 1)
            conf, pred = torch.max(probs,dim=1)        
            result = {"pred":pred.item(), "conf":conf.item()}    
        return result

    def image_predict(self,img_file):
        img = Image.open(img_file).convert("RGB")
        x = transform_image(img).to(self.device)
        
        result = {}
        self.image_model.eval()
        with torch.no_grad():
            logits = self.image_model(x)
            probs = torch.softmax(logits, dim=1 )
            conf, pred = torch.max(probs, dim=1 )         
            result = {"pred":pred.item(), "conf":conf.item()}    
        return result
    

    def sound_predict(self, sound_file):
        
        image_path = sound_to_image(sound_file)
        image_tf = sound_image_tf()
        img = Image.open(image_path).convert("RGB")
        x = image_tf(img).unsqueeze(0).to(self.device)
        
        self.sound_model.eval()
        score = 0
        pred = 0
        result = {}
        with torch.no_grad():
            logits = self.sound_model(x)
            probs = torch.softmax(logits,dim = 1)
            print(probs)
            conf, pred = torch.max(probs,dim=1)        
            result = {"pred":pred.item(), "conf":conf.item()}    
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

            # result = {"pred":pred, "conf":score}    
        return result
    
    def combind_predict(self, image_file, sound_file, text):
        image_result = self.image_predict(image_file)
        sound_result = self.sound_predict(sound_file)
        text_result = self.text_predict(text)

        


