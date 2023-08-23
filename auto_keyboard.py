import cv2
import numpy as np
import pandas as pd
import math
import mediapipe as mp
from pynput.keyboard import Controller
from screeninfo import get_monitors
import tensorflow as tf
import pyautogui
from tensorflow.keras.models import load_model

class Store():
    def __init__(self,pos,size,text):
        self.pos=pos
        self.size=size
        self.text=text

def draw(img, storedVar):
    for button in storedVar:
        x, y = button.pos
        w, h = button.size
        cv2.rectangle(img, (x - w, y - h), (x + w, y + h), (0, 255, 0), thickness=2)
        cv2.putText(img, button.text, (x-15, y+15), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)
    return img

def draw_legend(img):
    overlay = img.copy()
    output = img.copy()
    cv2.rectangle(overlay, (round(round(10/1280*width)), round(10/960*height)), (round(round(280/1280*width)), round(240/960*height)), (0, 0, 0), -1)
    i = 0
    for k, v in key_map.items():
        cv2.putText(overlay, f'{k} : {v}', (20, 40+(i*35)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        i+=1

    cv2.addWeighted(overlay, 0.7, output, 0.3, 0, output)
    return output

def draw_input(img, text):
    overlay = img.copy()
    output = img.copy()

    cv2.rectangle(overlay, (round(990/1280*width), round(900/960*height)), (round(1270/1280*width), round(950/960*height)), (0, 0, 0), -1)
    cv2.putText(overlay, text, (round(1010/1280*width), round(935/960*height)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    cv2.addWeighted(overlay, 0.7, output, 0.3, 0, output)
    return output

def vertical_symmetry(x, y, height):
    """
    세로축 (Y축) 대칭 변환
    :param x: 원본 X 좌표
    :param y: 원본 Y 좌표
    :param height: 대칭시킬 기준 선의 위치 (이미지의 높이)
    :return: 세로축에 대해 대칭된 새로운 (x, y) 좌표
    """
    new_y = height - y
    return x, new_y

def horizontal_symmetry(x, y, width):
    """
    가로축 (X축) 대칭 변환
    :param x: 원본 X 좌표
    :param y: 원본 Y 좌표
    :param width: 대칭시킬 기준 선의 위치 (이미지의 너비)
    :return: 가로축에 대해 대칭된 새로운 (x, y) 좌표
    """
    new_x = width - x
    return new_x, y

def drawROI(img, corners):
    cpy = img.copy()

    c1 = (192, 192, 255)
    c2 = (128, 128, 255)

    for pt in corners:
        cv2.circle(cpy, tuple(pt.astype(int)), 25, c1, -1, cv2.LINE_AA)

    cv2.line(cpy, tuple(corners[0].astype(int)), tuple(corners[1].astype(int)), c2, 2, cv2.LINE_AA)
    cv2.line(cpy, tuple(corners[1].astype(int)), tuple(corners[2].astype(int)), c2, 2, cv2.LINE_AA)
    cv2.line(cpy, tuple(corners[2].astype(int)), tuple(corners[3].astype(int)), c2, 2, cv2.LINE_AA)
    cv2.line(cpy, tuple(corners[3].astype(int)), tuple(corners[0].astype(int)), c2, 2, cv2.LINE_AA)

    disp = cv2.addWeighted(img, 0.3, cpy, 0.7, 0)

    return disp

def onMouse(event, x, y, flags, param):
    global srcQuad, dragSrc, ptOld, src

    if event == cv2.EVENT_LBUTTONDOWN:
        for i in range(4):
            if cv2.norm(srcQuad[i] - (x, y)) < 25: # type: ignore
                dragSrc[i] = True
                ptOld = (x, y)
                break

    if event == cv2.EVENT_LBUTTONUP:
        for i in range(4):
            dragSrc[i] = False

    if event == cv2.EVENT_MOUSEMOVE:
        for i in range(4):
            if dragSrc[i]:
                dx = x - ptOld[0]
                dy = y - ptOld[1]

                srcQuad[i] += (dx, dy)

                cpy = drawROI(src, srcQuad)
                cv2.imshow(window_name, cpy)
                ptOld = (x, y)
                break

def convert_position(pt1, pt2, pers):
    # 변환된 동차 좌표 계산
    transformed_pt1 = np.dot(pers, pt1)
    transformed_pt2 = np.dot(pers, pt2)

    # 변환된 유클리드 좌표 계산 (동차좌표를 유클리드 좌표로 변환)
    transformed_pt1 = transformed_pt1 / transformed_pt1[2]
    transformed_x1, transformed_y1 = round(transformed_pt1[0]), round(transformed_pt1[1])

    # 변환된 유클리드 좌표 계산 (동차좌표를 유클리드 좌표로 변환)
    transformed_pt2 = transformed_pt2 / transformed_pt2[2]
    transformed_x2, transformed_y2 = round(transformed_pt2[0]), round(transformed_pt2[1]) 

    return (transformed_x1, transformed_y1), (transformed_x2, transformed_y2)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

model = load_model("demo_keyboard_model")

print("가상 인터페이스 생성중...")
test_data = pd.read_hdf('1280x960.h5', 'df')
test_input = tf.convert_to_tensor(test_data, dtype=tf.float64)
pred = model.predict_classes(test_input)
pred = tf.reshape(pred, [1280, 960]).numpy()

test = np.full((960, 1280, 3), 255, np.uint8)
for y in range(960):
    for x in range(1280):
        test[y][x] = pred[x][y]

key_map = {"1":"1", "2":"2", "3":"3", "4":"LEFT", "5":"OK", "6":"RIGHT"}
keyboard = Controller()

'''
프로젝터 영역 자동 탐지

프로젝터로 빨강색 이미지를 출력하여 꼭짓점 탐지
'''
monitor1 = get_monitors()[1]
monitor2 = get_monitors()[2]

window_name = "Monitor"
window_name2 = "Projector"

cv2.namedWindow(window_name, cv2.WND_PROP_AUTOSIZE)
cv2.moveWindow(window_name, monitor1.x - 1, monitor1.y - 1)

cv2.namedWindow(window_name2, cv2.WND_PROP_FULLSCREEN)
cv2.moveWindow(window_name2, monitor2.x - 1, monitor2.y - 1)
cv2.setWindowProperty(window_name2, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

red_image = (np.ones((960, 1280, 3)) * [0, 0, 255]).astype(np.uint8)
cv2.imshow(window_name2, red_image)

# 웹캠으로부터 입력 받기
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

pers_corners = []

while True:
    # 이미지를 불러옵니다.
    _, img = cap.read()

    # 이미지를 HSV로 변환
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 빨간색의 범위를 정의
    lower_red_1 = np.array([0, 70, 50])
    upper_red_1 = np.array([10, 255, 255])
    lower_red_2 = np.array([170, 70, 50])
    upper_red_2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    mask = mask1 + mask2

    # 테두리 찾기
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 가장 큰 테두리 찾기
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        # 꼭짓점 찾기
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx_corners = cv2.approxPolyDP(largest_contour, epsilon, True)

        # 근사화된 꼭짓점의 수가 4인 경우만 그림
        if len(approx_corners) == 4:
            for corner in approx_corners:
                cv2.circle(img, tuple(corner[0]), 10, (0, 255, 0), -1)  # 꼭짓점 그리기
            # cap.release()
            pers_corners = approx_corners
            break
'''
가상 키패드 그리는 부분
'''
StoredVar = []

img = np.full((960, 1280, 3), 0, np.uint8)

for index in range(1, model.output.shape[1]):
    min_y = min(np.where(test==index)[0])
    min_x = min(np.where(test==index)[1])
    max_y = max(np.where(test==index)[0])
    max_x = max(np.where(test==index)[1])

    StoredVar.append(Store([round(((min_x+max_x)/2)/1280*width), round(((min_y+max_y)/2)/960*height)],[round(((max_x-min_x)/2)/1280*width), round(((max_y-min_y)/2)/960*height)], str(index)))

flag = 0
text = ''

img = draw(img, StoredVar)
img = draw_legend(img)
img = draw_input(img, text)

projector_img = img.copy()

mpHands = mp.solutions.hands
Hands = mpHands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

# 체크: 웹캠이 제대로 열렸는지 확인
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

dw = 1280 #640
dh = 960 #480 #round(dw * 297 / 210)  # A4 용지 크기: 210x297cm

# 모서리 점들의 좌표, 드래그 상태 여부
srcQuad = np.array([pers_corners[0][0], [pers_corners[1][0][0], height-pers_corners[1][0][1]], [width-pers_corners[2][0][0], height-pers_corners[2][0][1]], [width-pers_corners[3][0][0], pers_corners[3][0][1]]], np.float32)
dstQuad = np.array([[0, 0], [0, dh-1], [dw-1, dh-1], [dw-1, 0]], np.float32)

# 원본 영상을 표시할 윈도우 이름
# cv2.namedWindow(window_name2, 1)


# 원본 영상을 표시
# base = cv2.imread('key sample1.png')
# base = draw(img, StoredVar)
# base = draw_legend(base)
# base = draw_input(base, text)
# base = cv2.resize(base, dsize=(width, height), interpolation=cv2.INTER_LINEAR)

# 가상 키보드 이미지 출력창
cv2.imshow(window_name2, projector_img)

while True:
    # 웹캠으로부터 프레임을 읽어옴
    ret, src = cap.read()
    
    # 프레임 읽기 실패 시 종료
    if not ret:
        print("Failed to grab frame")
        break

    results=Hands.process(src)
    lmList=[]
    if results.multi_hand_landmarks:
        for img_in_frame in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(src, img_in_frame, mpHands.HAND_CONNECTIONS)
        for id,lm in enumerate(results.multi_hand_landmarks[0].landmark):
            h,w,c=src.shape
            cx,cy=int(lm.x*w),int(lm.y*h)
            lmList.append([cx,cy])

    projector_img = img.copy()

    # 투시 변환
    pers = cv2.getPerspectiveTransform(srcQuad, dstQuad)
    dst = cv2.warpPerspective(src, pers, (dw, dh), flags=cv2.INTER_CUBIC)
    
    cv2.imshow('dst', dst)

    if lmList:
        pt1, pt2 = convert_position(np.array([lmList[8][0],lmList[8][1],1]), np.array([lmList[4][0],lmList[4][1],1]), pers)
        x1, y1 = pt1
        x2, y2 = pt2
        l = math.hypot(x2-x1,y2-y1)
        # when clicked
        if not l > 100:
            for button in StoredVar:
                x, y = button.pos
                w, h = button.size
                if x - w< x1 < x + w and y - h < y1 < y + h: # 버튼 영역 내에 손가락이 들어온 것을 탐지
                    try:
                        if flag == 0: # 클릭 한번 하면 손가락 뗄 뗴 까지 클릭 비활성화
                            print(key_map[button.text])
                            for s in key_map[button.text]:
                                keyboard.press(s)

                            pyautogui.press('enter')
                            print("Correct result: ", button.text)
                            print("Predict result:", np.argmax(model.predict([[x1/(width-1), y1/(height-1)]])))  
                            text = f'{key_map[button.text]} ({button.text})'
                            cv2.rectangle(src, (x - w - 5, y - h - 5), (x + w + 5, y + h + 5), (0, 255, 0), thickness=2)
                            cv2.rectangle(projector_img, (x - w - 5, y - h - 5), (x + w + 5, y + h + 5), (0, 255, 0), thickness=2)
                            # sd.Beep(2000, 100)
                            print('\a')

                            flag = 1
                    except Exception as e:
                        print(e)
        else: # 손가락 때면 클릭 활성화
            for button in StoredVar:
                temp_x, temp_y = button.pos
                temp_w, temp_h = button.size
                # if temp_x - temp_w < lmList[8][0] < temp_x + temp_w and temp_y - temp_h < lmList[8][1] < temp_y + temp_h: # 버튼 영역 내에 중지 손가락 끝이 들어온 것을 탐지
                if temp_x - temp_w < x1 < temp_x + temp_w and temp_y - temp_h < y1 < temp_y + temp_h: # 버튼 영역 내에 중지 손가락 끝이 들어온 것을 탐지
                    cv2.rectangle(src, (temp_x - temp_w - 5, temp_y - temp_h - 5), (temp_x + temp_w + 5, temp_y + temp_h + 5), (0, 0, 255), thickness=2)
                    cv2.rectangle(projector_img, (temp_x - temp_w - 5, temp_y - temp_h - 5), (temp_x + temp_w + 5, temp_y + temp_h + 5), (0, 0, 255), thickness=2)
            flag = 0

    src = draw(src, StoredVar)
    src = draw_legend(src)
    src = draw_input(src, text)
    projector_img = draw_input(projector_img, text)
    
    cv2.line(src, tuple(pers_corners[0][0]), tuple(pers_corners[0][0]), (255,0,0), 10, cv2.LINE_AA)
    cv2.line(src, tuple(pers_corners[1][0]), tuple(pers_corners[1][0]), (0,255,0), 10, cv2.LINE_AA)
    cv2.line(src, tuple(pers_corners[2][0]), tuple(pers_corners[2][0]), (0,0,255), 10, cv2.LINE_AA)
    cv2.line(src, tuple(pers_corners[3][0]), tuple(pers_corners[3][0]), (255,255,255), 10, cv2.LINE_AA)

    cv2.imshow(window_name, src)
    cv2.imshow(window_name2, projector_img)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
