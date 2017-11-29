from PIL import Image
import numpy as np
import os

def average_hash(fname, size = 16):
        img = Image.open(fname)
        img = img.convert('L') # 그레이로 변환
        img = img.resize((size, size), Image.ANTIALIAS) # resize
        pixel_data = img.getdata() # 픽셀 데이터 가져오기
        pixels = np.array(pixel_data) # 1차원 배열로 변환
        pixels = pixels.reshape(size,size) # size*size 배열로 변환
        avg = pixels.mean() # 평균
        diff = 1 *(pixels > avg) # 평균보다 크면 1, 작으면 0
        return diff
def np2hash(ahash):
    bhash = []
    for nl in ahash.tolist():
        s1 = [str(i) for i in nl]
        s2 = "".join(s1)
        i = int(s2,2)
        bhash.append("%04x" % i)
    return "".join(bhash)

ahash = average_hash('data/46m.png')

# print(ahash)
print(np2hash(ahash))