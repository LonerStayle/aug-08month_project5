from fastapi import FastAPI,  UploadFile
from jin_sup.combind.AutoPredict import AutoPredict

app = FastAPI()
auto_pred = AutoPredict()

@app.post('/predict')
def predict(image_file: UploadFile, sound_file:UploadFile):
    auto_pred.combind_predict(image_file,sound_file)

