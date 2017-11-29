import cv2
import sys, os

image_file = os.path.join("data", "user", sys.argv[1])

cascade_file = os.path.join("haarcascade_frontalface_alt.xml")

image = cv2.imread(image_file)
image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cascade = cv2.CascadeClassifier(os.path.expanduser(cascade_file))
face_list = cascade.detectMultiScale(image_gs,scaleFactor=1.1,minNeighbors=1,minSize=(100,100))

if len(face_list) > 0:
  print(face_list)
  color = (0,0,255)
  for face in face_list:
    x,y,w,h = face
    cv2.rectangle(image, (x,y), (x+w, y+h), color, thickness=8)
    cv2.imwrite("facedetect-output.PNG", image)
else:
  print("no face")