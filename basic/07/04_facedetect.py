import cv2
import sys, os, re

if len(sys.argv) <= 1:
  print("no imput file")
  quit()

image_file = os.path.join("data", "user", sys.argv[1])
cascade_file = os.path.join("haarcascade_frontalface_alt.xml")
output_file = re.sub(r'\.jpg|jpeg|PNG$', '-mosaic.jpg', image_file)
mosaic_rate = 30

image = cv2.imread(image_file)
image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cascade = cv2.CascadeClassifier(os.path.expanduser(cascade_file))
face_list = cascade.detectMultiScale(image_gs,scaleFactor=1.1,minNeighbors=1,minSize=(100,100))

if len(face_list) == 0:
  print("no face")
  quit()

for(x,y,w,h) in face_list:
  face_img = image[y:y+h, x:x+w]
  face_img = cv2.resize(face_img, (w//mosaic_rate, h//mosaic_rate))
  face_img = cv2.resize(face_img, (w,h), interpolation=cv2.INTER_AREA);
  image[y:y+h, x:x+w] = face_img

print(output_file)
cv2.imwrite(output_file, image)