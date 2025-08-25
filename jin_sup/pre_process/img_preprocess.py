import numpy as np
from PIL import Image
import random


class TrimBorder(object):
    """고정 마진 잘라내기 (필요시 값 조정)"""
    def __init__(self, margin=10):
        self.margin = margin
    def __call__(self, img):
        w, h = img.size
        m = self.margin
        return img.crop((m, m, w-m, h-m))
    

def compute_mean_std(image_paths):
    # Grayscale 후 0~1로 평균/표준편차 추정
    s, s2, n = 0.0, 0.0, 0
    for p in image_paths:
        x = np.array(Image.open(p).convert("L"), dtype=np.float32) / 255.0
        s  += x.mean()
        s2 += (x**2).mean()
        n  += 1
    mean = s / n
    std  = (s2 / n - mean**2) ** 0.5
    return float(mean), float(std)

"""가로 방향 띠(mask): 시간 축 마스킹"""
class TimeMask(object):
    def __init__(self, max_width=0.15, p=0.5, value=0.0):
        self.max_width = max_width
        self.p = p
        self.value = value
    def __call__(self, img):
        if random.random() > self.p: return img
        w, h = img.size
        band = int(w * random.uniform(0.03, self.max_width))
        x0 = random.randint(0, max(0, w - band))
        # 사각형 덮기
        mask = Image.new("RGB", (band, h), tuple(int(255*self.value) for _ in range(3)))
        img.paste(mask, (x0, 0))
        return img

"""세로 방향 띠(mask): 주파수 축 마스킹"""
class FreqMask(object):
    def __init__(self, max_height=0.2, p=0.5, value=0.0):
        self.max_height = max_height
        self.p = p
        self.value = value
    def __call__(self, img):
        if random.random() > self.p: return img
        w, h = img.size
        band = int(h * random.uniform(0.03, self.max_height))
        y0 = random.randint(0, max(0, h - band))
        mask = Image.new("RGB", (w, band), tuple(int(255*self.value) for _ in range(3)))
        img.paste(mask, (0, y0))
        return img