import cv2
import mediapipe as mp
import pandas as pd

# --- 설정: 병원용 제스처와 키 매핑 (영어) ---
GESTURE_MAP = {
    ord('1'): "Pain",    # 주먹 (아파요)
    ord('2'): "Help",    # 손바닥 (도와주세요)
    ord('3'): "Water",   # OK사인 (물 주세요)
    ord('4'): "Toilet",  # 검지 (화장실)
    ord('5'): "Yes"      # 엄지 (괜찮아요/감사)
}

DATA_FILE = 'hand_data.csv'

# MediaPipe 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
data_list = []

print(">> [병원 수화 데이터 수집기] 시작")
print(">> 1:Pain, 2:Help, 3:Water, 4:Toilet, 5:Yes")
print(">> 'q'를 누르면 저장 후 종료")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        
        if key in GESTURE_MAP:
            label = GESTURE_MAP[key]
            landmarks_row = []
            for lm in hand_landmarks.landmark:
                landmarks_row.extend([lm.x, lm.y, lm.z])
            
            data_list.append([label] + landmarks_row)
            print(f">> [{label}] 저장! (총 {len(data_list)}개)")

    cv2.imshow('Hospital Gesture Collector', image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# CSV 저장
if data_list:
    columns = ['label']
    for i in range(21):
        columns.extend([f'x{i}', f'y{i}', f'z{i}'])
        
    df = pd.DataFrame(data_list, columns=columns)
    df.to_csv(DATA_FILE, index=False)
    print(f"\n>> '{DATA_FILE}' 저장 완료!")
else:
    print("\n>> 저장된 데이터 없음")

cap.release()
cv2.destroyAllWindows()
hands.close()