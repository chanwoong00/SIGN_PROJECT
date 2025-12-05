이 프로젝트는 병원 내에서 의사소통이 어려운 환자들이 **간단한 손동작만으로 의료진에게 의사를 전달**할 수 있도록 돕는 **실시간 수화 인식 시스템 프로토타입**입니다.

**MediaPipe**를 이용해 손의 랜드마크를 추출하고, **k-NN** 머신러닝 알고리즘을 통해 5가지 필수 단어를 실시간으로 분류합니다.

---

## 🛠️ 개발 환경 설정 (Installation)

이 프로젝트는 **Python 3.10** 환경에서 개발되었습니다. 원활한 실행을 위해 `conda` 가상환경 사용을 권장합니다.

1. 가상환경 생성 및 활성화
```bash
conda create -n vision_project python=3.10
conda activate vision_project

2. 필수 라이브러리 설치 
pip install opencv-python mediapipe pandas scikit-learn


#파일설명#
collect_data.py - 데이터 수집

사용 키 매핑 (Key Mapping):

1: Pain (주먹) - "아파요"
2: Help (손바닥) - "도와주세요"
3: Water (OK사인) - "물 주세요"
4: Toilet (검지) - "화장실"
5: Yes (엄지) - "괜찮아요/감사"

종료: q 키를 누르고 종료해야 csv 파일 생성


train_predict.py - 실시간 인식

손이 감지되지 않으면 화면에 "Waiting for hand..." 대기 문구가 뜹니다.

손을 올리면 학습된 단어(Pain, Help 등)가 화면 중앙에 초록색 텍스트로 표시됩니다.

종료: q 키를 누르면 종료
