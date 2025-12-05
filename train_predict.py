import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import os

DATA_FILE = 'hand_data.csv'

# --- 1. 모델 학습 ---
if not os.path.exists(DATA_FILE):
    print(f"오류: '{DATA_FILE}'이 없습니다. collect_data.py를 먼저 실행하세요.")
    exit()

print(">> 모델 학습 중...")
df = pd.read_csv(DATA_FILE)
X = df.drop('label', axis=1)
y = df['label']

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X, y)
print(">> 학습 완료! 웹캠 시작")

# --- 2. 실시간 예측 ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    h, w, c = image.shape

    if results.multi_hand_landmarks:
        # 손 감지됨: 예측 수행
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        landmarks_row = []
        for lm in hand_landmarks.landmark:
            landmarks_row.extend([lm.x, lm.y, lm.z])
        
        live_data = np.array([landmarks_row])
        prediction = model.predict(live_data)[0]
        
        # 결과 출력 (초록색)
        cv2.putText(image, f"Status: {prediction}", (50, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5, cv2.LINE_AA)
        
    else:
        # 손 없음: 대기 문구 출력 (빨간색)
        text = "Waiting for hand..."
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, 1.5, 3)[0]
        text_x = (w - text_size[0]) // 2
        text_y = (h + text_size[1]) // 2
        
        cv2.putText(image, text, (text_x, text_y), 
                    font, 1.5, (0, 0, 255), 3, cv2.LINE_AA)

    cv2.imshow('Hospital Sign Language System', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()