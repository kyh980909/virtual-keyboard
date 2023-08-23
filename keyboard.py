import math
import cv2
import mediapipe as mp
from pynput.keyboard import Controller
from time import sleep
import math
import numpy as np
import pandas as pd
import tensorflow as tf
import pyautogui
from tensorflow.keras.models import load_model

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

model = load_model("demo_keyboard_model")

print("가상 인터페이스 생성중...")
test_data = pd.read_hdf('1280x720.h5', 'df')
test_input = tf.convert_to_tensor(test_data, dtype=tf.float64)
pred = model.predict_classes(test_input)
pred = tf.reshape(pred, [1280, 720]).numpy()

test = np.full((720, 1280, 3), 255, np.uint8)
for y in range(720):
    for x in range(1280):
        test[y][x] = pred[x][y]


key_map = {"1":"1", "2":"2", "3":"3", "4":"LEFT", "5":"OK", "6":"RIGHT"}
# virtual = np.load('test.npy')

camIndex=0
cap = cv2.VideoCapture(camIndex)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

mpHands = mp.solutions.hands  
Hands = mpHands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

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
    cv2.rectangle(overlay, (round(round(10/1280*width)), round(10/720*height)), (round(round(280/1280*width)), round(240/720*height)), (0, 0, 0), -1)
    i = 0
    for k, v in key_map.items():
        cv2.putText(overlay, f'{k} : {v}', (20, 40+(i*35)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        i+=1

    cv2.addWeighted(overlay, 0.7, output, 0.3, 0, output)
    return output

def draw_input(img, text):
    overlay = img.copy()
    output = img.copy()

    cv2.rectangle(overlay, (round(990/1280*width), round(900/720*height)), (round(1270/1280*width), round(950/720*height)), (0, 0, 0), -1)
    cv2.putText(overlay, text, (round(1010/1280*width), round(935/720*height)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    cv2.addWeighted(overlay, 0.7, output, 0.3, 0, output)
    return output

StoredVar = []

img = np.full((720, 1280, 3), 255, np.uint8)

for index in range(1, model.output.shape[1]):
    min_y = min(np.where(test==index)[0])
    min_x = min(np.where(test==index)[1])
    max_y = max(np.where(test==index)[0])
    max_x = max(np.where(test==index)[1])

    StoredVar.append(Store([round(((min_x+max_x)/2)/1280*width), round(((min_y+max_y)/2)/720*height)],[round(((max_x-min_x)/2)/1280*width), round(((max_y-min_y)/2)/720*height)], str(index)))

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
        if not l > 100:
            for button in StoredVar:
                x, y = button.pos
                w, h = button.size
                if x - w< lmList[12][0] < x + w and y - h < lmList[12][1] < y + h: # 버튼 영역 내에 손가락이 들어온 것을 탐지
                    try:
                        if flag == 0: # 클릭 한번 하면 손가락 뗄 뗴 까지 클릭 비활성화
                            print(key_map[button.text])
                            for s in key_map[button.text]:
                                keyboard.press(s)

                            pyautogui.press('enter')
                            print("Correct result: ", button.text)
                            print("Predict result:", np.argmax(model.predict([[x1/(width-1), y1/(height-1)]])))  
                            text = f'{key_map[button.text]} ({button.text})'
                            cv2.rectangle(img, (x - w - 5, y - h - 5), (x + w + 5, y + h + 5), (0, 255, 0), thickness=2)
                            # sd.Beep(2000, 100)
                            print('\a')
                            flag = 1
                    except Exception as e:
                        print(e)
        else: # 손가락 때면 클릭 활성화
            for button in StoredVar:
                temp_x, temp_y = button.pos
                temp_w, temp_h = button.size
                if temp_x - temp_w < lmList[12][0] < temp_x + temp_w and temp_y - temp_h < lmList[12][1] < temp_y + temp_h: # 버튼 영역 내에 중지 손가락 끝이 들어온 것을 탐지
                    cv2.rectangle(img, (temp_x - temp_w - 5, temp_y - temp_h - 5), (temp_x + temp_w + 5, temp_y + temp_h + 5), (0, 0, 255), thickness=2)
            flag = 0


    img = draw(img, StoredVar)
    img = draw_legend(img)
    img = draw_input(img, text)
    cv2.imshow("Scenario 1",img)

    if cv2.waitKey(1)==113: #Q=113
        break