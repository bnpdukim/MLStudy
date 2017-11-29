from PIL import Image
import numpy as np
import os, re
search_dir = os.path.join("data","101_ObjectCategories")
cache_dir = os.path.join("data","cache_avhash")

if not os.path.exists(cache_dir):
    os.mkdir(cache_dir)

def average_hash(fname, size = 16):
    fname2 = fname[len(search_dir):]
    cache_file = os.path.join(cache_dir ,fname2.replace('\\','_') + ".csv")
    if not os.path.exists(cache_file):
        img = Image.open(fname)
        img = img.convert('L') # 그레이로 변환
        img = img.resize((size, size), Image.ANTIALIAS) # resize
        pixel_data = img.getdata() # 픽셀 데이터 가져오기
        pixels = np.array(pixel_data) # 1차원 배열로 변환
        pixels = pixels.reshape(size,size) # size*size 배열로 변환
        avg = pixels.mean() # 평균
        px = 1 *(pixels > avg) # 평균보다 크면 1, 작으면 0
        np.savetxt(cache_file, px, fmt="%.0f", delimiter=",")
    else:
        px = np.loadtxt(cache_file, delimiter=",")
    return px;


def hamming_dist(a,b):
    aa = a.reshape(1, -1)
    ab = b.reshape(1, -1)
    dist = (aa != ab).sum()
    return dist

def enum_all_files(path):
    for root, dirs, files in os.walk(path):
        for f in files:
            fname = os.path.join(root,f)
            if re.search(r'\.(jpg|jpeg|png)$',fname):
                yield fname

def find_image(fname, rate):
    src = average_hash(fname)
    for fname in enum_all_files(search_dir):
        dst = average_hash(fname)
        diff_r = hamming_dist(src,dst) / 256
        print("[check] ", fname)
        if diff_r < rate:
            yield(diff_r, fname)

srcfile = os.path.join(search_dir, "chair", "image_0016.jpg")
html = ""
predictionImages = find_image(srcfile, 0.25)
sim = list(predictionImages)
sim = sorted(sim, key=lambda x:x[0])
for r, f in sim:
    print(r, ">", f)
    s = '<div style="float:left;"><h3>[ 차이 :' + str(r) + '-' + os.path.basename(f) + ' ]</h3>' + '<p><a href="' + f + '"><img src="' + f  + '" width=400>'+'</a></p></div>'
    html += s

html= """<html><head><meta charset="utf8"></head>
<body><h3>원본</h3><p>
<img src='{0}' width = 400></p>{1}
</body></html>""".format(srcfile, html)

with open(os.path.join("avhash-search-output.html"), "w", encoding="utf-8") as f:
    f.write(html)

print("ok")


