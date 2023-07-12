import cv2
import mediapipe as mp
import pandas as pd

file_name = '3'
cap = cv2.VideoCapture(f'./video_data/{file_name}.mp4')

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define output video codec and file name
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(f'skeleton_video/{file_name}.mp4', fourcc, fps, (width, height))

# Initialize MediaPipe Hand model
mp_hands = mp.solutions.hands # type: ignore
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils # type: ignore

df = pd.DataFrame(columns=['x', 'y'])

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if success:
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image.shape[1]
                    y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image.shape[0]
                    print(f'Index finger tip coordinates: ({x}, {y})')
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    df = df.append({'x':x, 'y':y}, ignore_index=True)

            # for y_ in range(0, 1000, 100):
            #     image = cv2.line(image, (0, y_), (1920, y_), (0, 255, 0), 5)
            
            out.write(image)
            cv2.imshow('MediaPipe Hands', image)
                # Wait for key press
            if cv2.waitKey(1) == ord('q'):
                break
        else:
            break

cap.release()
out.release()
cv2.destroyAllWindows()
df.to_csv(f'hand_extract_data/{file_name}.csv', index=False)
