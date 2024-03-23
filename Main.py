import cv2
import numpy as np
import face_recognition
import os

ImageSourcePath = "Images"
ImgName = []
CuredImage = []
ImageList = os.listdir(ImageSourcePath)
print(ImageList)

for img in ImageList:
    curedimg = cv2.imread(f'{ImageSourcePath}/{img}')
    CuredImage.append(curedimg)
    ImgName.append(os.path.splitext(img)[0])

print(ImgName)


def Encoder(images):
    encodeList = []
    for img in CuredImage:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeList= Encoder(ImageList)
print(len(encodeList))



stopTheCap = cv2.VideoCapture(0)

while True:
    sc,imgs = stopTheCap.read()
    ImsRS = cv2.resize(imgs,(0,0),None,0.25,0.25)
    ImsRS = cv2.cvtColor(ImsRS, cv2.COLOR_BGR2RGB)

    faceCurrentFrame = face_recognition.face_locations(ImsRS)
    encodeCureentFrame = face_recognition.face_encodings(ImsRS,faceCurrentFrame)

    for encode,loc in zip(encodeCureentFrame,faceCurrentFrame):
        match = face_recognition.compare_faces(encodeList,encode)
        Dis = face_recognition.face_distance(encodeList,encode)
        print(Dis)


