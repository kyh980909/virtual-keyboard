import cv2
import mediapipe as mp
from pynput.keyboard import Controller
from time import sleep
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model("keyboard")

camIndex=0
cap=cv2.VideoCapture(camIndex)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

mpHands=mp.solutions.hands
Hands=mpHands.Hands()
mpDraw=mp.solutions.drawing_utils

keyboard = Controller()

class Store():
    def __init__(self,pos,size,text):
        self.pos=pos
        self.size=size
        self.text=text
    
def draw(img, storedVar):

    for button in storedVar:
        x, y = button.pos
        w, h = button.size
        cv2.rectangle(img, button.pos, (x + w, y + h), (64, 64, 64), thickness=2)
        cv2.putText(img, button.text, (x+10, y+43),cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)
    return img

StoredVar = []
# for i in range(len(keys)):
#     for j, key in enumerate(keys[i]):
#         StoredVar.append(Store([60 * j + 10, 60 * i + 10], [50,50],key))

StoredVar.append(Store([100, 100],[250, 150],"2"))
StoredVar.append(Store([930, 100],[250, 150],"3"))
StoredVar.append(Store([480, 280],[320, 120],"4"))
StoredVar.append(Store([270, 360],[150, 250],"5"))
StoredVar.append(Store([850, 360],[150, 250],"6"))
StoredVar.append(Store([560, 420],[160, 120],"7"))
StoredVar.append(Store([480, 570],[320, 120],"8"))
StoredVar.append(Store([100, 710],[250, 150],"9"))
StoredVar.append(Store([930, 710],[250, 150],"10"))



while (cap.isOpened()):
    success_,img=cap.read()
    img = cv2.flip(img, 1)
    cvtImg=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=Hands.process(cvtImg)
    lmList=[]

    if results.multi_hand_landmarks:
        for img_in_frame in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, img_in_frame, mpHands.HAND_CONNECTIONS)
        for id,lm in enumerate(results.multi_hand_landmarks[0].landmark):
            h,w,c=img.shape
            cx,cy=int(lm.x*w),int(lm.y*h)
            lmList.append([cx,cy])

    if lmList:
        for button in StoredVar:
            x, y = button.pos
            w, h = button.size

            if x < lmList[8][0] < x + w and y < lmList[8][1] < y + h: # 버튼 영역 내에 손가락이 들어온 것을 탐지
                cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 0, 255), thickness=2)
                x1,y1=lmList[8][0],lmList[8][1]
                x2,y2=lmList[4][0],lmList[4][1]
                l=math.hypot(x2-x1-30,y2-y1-30)
                ## when clicked
                if not l > 30:
                    try:
                        # keyboard.press(button.text)
                        print(np.argmax(model.predict([[x1, y1]])))
                        cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 255, 0), thickness=2)
                        sleep(0.1)
                    except Exception as e:
                        print(e)


    img = draw(img, StoredVar)

    cv2.imshow("Hand Tracking",img)

    if cv2.waitKey(1)==113: #Q=113
        break