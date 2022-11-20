import math
from time import sleep
import winsound as sd
import cv2
import mediapipe as mp
from pynput.keyboard import Controller
from time import sleep
import math
import numpy as np
import tensorflow as tf
import pyautogui
from tensorflow.keras.models import load_model
import asyncio

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

model = load_model("demo_keyboard_model")

key_map = {"1":"POWER ON", "2":"START", "3":"UP", "4":"LEFT", "5":"RIGHT", "6":"OK", "7":"DOWN", "8":"POWER OFF", "9":"STOP"}

camIndex=0
cap=cv2.VideoCapture(camIndex)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

mpHands=mp.solutions.hands  # type: ignore
Hands=mpHands.Hands()
mpDraw=mp.solutions.drawing_utils  # type: ignore

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
        cv2.rectangle(img, (x - w, y - h), (x + w, y + h), (255, 255, 255), thickness=2)
        cv2.putText(img, button.text, (x-15, y+15), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 255), 2)
    return img

def draw_legend(img):
    overlay = img.copy()
    output = img.copy()
    cv2.rectangle(overlay, (10, 600), (280, 950), (0, 0, 0), -1)
    i = 0
    for k, v in key_map.items():
        cv2.putText(overlay, f'{k} : {v}', (20, 650+(i*35)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        i+=1

    cv2.addWeighted(overlay, 0.7, output, 0.3, 0, output)
    return output

def draw_input(img, text):
    overlay = img.copy()
    output = img.copy()

    cv2.rectangle(overlay, (990, 900), (1270, 950), (0, 0, 0), -1)
    cv2.putText(overlay, text, (1010, 935), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    cv2.addWeighted(overlay, 0.7, output, 0.3, 0, output)
    return output

StoredVar = []

StoredVar.append(Store([390, 280],[45, 45],"1")) # 2
StoredVar.append(Store([890, 280],[45, 45],"2")) # 3
StoredVar.append(Store([640, 320],[110, 40],"3")) # 4
StoredVar.append(Store([480, 460],[45, 80],"4")) # 5
StoredVar.append(Store([800, 460],[45, 80],"5")) # 6
StoredVar.append(Store([640, 460],[50, 50],"6")) # 7
StoredVar.append(Store([640, 600],[110, 40],"7")) # 8
StoredVar.append(Store([390, 630],[45, 45],"8")) # 9
StoredVar.append(Store([890, 630],[45, 45],"9")) # 10

flag = 0
text = ''

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
        x1,y1 = lmList[12][0],lmList[12][1]
        x2,y2 = lmList[8][0],lmList[8][1]
        l = math.hypot(x2-x1,y2-y1)
        # when clicked
        if not l > 110:
            for button in StoredVar:
                x, y = button.pos
                w, h = button.size
                if x - w< lmList[12][0] < x + w and y - h < lmList[12][1] < y + h: # 버튼 영역 내에 손가락이 들어온 것을 탐지
                    try:
                        if flag == 0: # 클릭 한번 하면 손가락 뗄 뗴 까지 클릭 비활성화
                            for s in key_map[button.text]:
                                keyboard.press(s)

                            pyautogui.press('enter')
                            print("Correct result: ", button.text)
                            print("Predict result:", np.argmax(model.predict([[x1/1279, y1/959]]))-1)  # type: ignore
                            text = f'{key_map[button.text]} ({button.text})'
                            cv2.rectangle(img, (x - w - 5, y - h - 5), (x + w + 5, y + h + 5), (0, 255, 0), thickness=2)
                            sd.Beep(2000, 100)
                            # sleep(0.1)
                            flag = 1
                    except Exception as e:
                        print(e)
        else: # 손가락 때면 클릭 활성화
            for button in StoredVar:
                temp_x, temp_y = button.pos
                temp_w, temp_h = button.size
                if temp_x - temp_w < lmList[12][0] < temp_x + temp_w and temp_y - temp_h < lmList[12][1] < temp_y + temp_h: # 버튼 영역 내에 손가락이 들어온 것을 탐지
                    cv2.rectangle(img, (temp_x - temp_w - 5, temp_y - temp_h - 5), (temp_x + temp_w + 5, temp_y + temp_h + 5), (0, 0, 255), thickness=2)
            flag = 0


    img = draw(img, StoredVar)
    img = draw_legend(img)
    img = draw_input(img, text)
    cv2.imshow("Hand Tracking",img)

    if cv2.waitKey(1)==113: #Q=113
        break