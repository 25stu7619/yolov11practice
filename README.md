# YOLOv11 완전 가이드 (Complete Guide)

<div align="center">

![YOLOv11](https://img.shields.io/badge/YOLOv11-2024-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)
![License](https://img.shields.io/badge/License-AGPL--3.0-red)

**최신 YOLO 시리즈의 차세대 객체 탐지 모델**

</div>

---

## 📑 목차

1. [YOLOv11 소개](#-yolov11-소개)
2. [주요 특징](#-주요-특징)
3. [설치 방법](#-설치-방법)
4. [빠른 시작](#-빠른-시작)
5. [모델 구조](#-모델-구조)
6. [사용 예제](#-사용-예제)
7. [학습 (Training)](#-학습-training)
8. [추론 (Inference)](#-추론-inference)
9. [모델 내보내기](#-모델-내보내기)
10. [성능 벤치마크](#-성능-벤치마크)
11. [하이퍼파라미터](#-하이퍼파라미터)
12. [Tips & Tricks](#-tips--tricks)
13. [FAQ](#-faq)
14. [참고 자료](#-참고-자료)

---

## 🚀 YOLOv11 소개

**YOLOv11**은 Ultralytics에서 2024년 9월에 출시한 최신 객체 탐지 모델입니다. YOLOv8 대비 **더 적은 파라미터**로 **더 높은 정확도**와 **더 빠른 속도**를 달성했습니다.

### 주요 개선사항
- 🎯 **정확도 향상**: mAP 2~3% 상승
- ⚡ **속도 개선**: 약 22% 추론 속도 향상
- 💾 **경량화**: 파라미터 수 19~22% 감소
- 🏗️ **아키텍처**: C3k2, C2PSA 모듈 도입
- 🔧 **안정성**: 더욱 안정적인 학습 프로세스

---

## ✨ 주요 특징

### 1. 다양한 태스크 지원
- **Object Detection**: 객체 탐지
- **Instance Segmentation**: 인스턴스 분할
- **Pose Estimation**: 포즈 추정
- **Oriented Object Detection (OBB)**: 회전 박스 탐지
- **Image Classification**: 이미지 분류

### 2. 5가지 모델 크기
| 모델 | 파라미터 | mAP50-95 | 속도 (T4) | 용도 |
|------|----------|----------|-----------|------|
| YOLOv11n | 2.6M | 39.5% | 1.0ms | 엣지 디바이스 |
| YOLOv11s | 9.4M | 47.0% | 1.7ms | 모바일 |
| YOLOv11m | 20.1M | 51.5% | 2.9ms | 일반 용도 |
| YOLOv11l | 25.3M | 53.4% | 4.1ms | 고성능 |
| YOLOv11x | 56.9M | 54.7% | 6.5ms | 최고 성능 |

### 3. 프레임워크 지원
- ✅ PyTorch
- ✅ ONNX
- ✅ TensorRT
- ✅ CoreML
- ✅ OpenVINO
- ✅ TensorFlow Lite

---

## 📦 설치 방법

### pip를 통한 설치 (권장)

```bash
# 최신 버전 설치
pip install ultralytics

# 특정 버전 설치
pip install ultralytics==8.3.0

# 개발 버전 설치 (최신 기능)
pip install git+https://github.com/ultralytics/ultralytics.git
```

### conda를 통한 설치

```bash
conda install -c conda-forge ultralytics
```

### 소스에서 설치

```bash
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics
pip install -e .
```

### 의존성 확인

```bash
pip install torch torchvision opencv-python numpy pillow pyyaml
```

**시스템 요구사항:**
- Python 3.8+
- PyTorch 1.8+
- CUDA 11.0+ (GPU 사용 시)

---

## 🏃 빠른 시작

### 1. CLI (Command Line Interface)

```bash
# 이미지 예측
yolo detect predict model=yolov11n.pt source='image.jpg'

# 웹캠 실시간 탐지
yolo detect predict model=yolov11n.pt source=0

# 비디오 예측
yolo detect predict model=yolov11n.pt source='video.mp4'

# 학습
yolo detect train data=coco.yaml model=yolov11n.pt epochs=100

# 검증
yolo detect val model=yolov11n.pt data=coco.yaml

# 모델 내보내기
yolo export model=yolov11n.pt format=onnx
```

### 2. Python API

```python
from ultralytics import YOLO

# 모델 로드
model = YOLO('yolov11n.pt')

# 이미지 예측
results = model('image.jpg')

# 결과 출력
for result in results:
    boxes = result.boxes  # Boxes object
    print(boxes.xyxy)     # 박스 좌표
    print(boxes.conf)     # 신뢰도
    print(boxes.cls)      # 클래스 ID

# 결과 저장
results[0].save('result.jpg')

# 결과 시각화
results[0].show()
```

---

## 🏗️ 모델 구조

### 전체 아키텍처

```
Input Image (640x640)
    ↓
[Backbone - Feature Extraction]
    ↓ C3k2 Blocks
    ↓ Down-sampling
    ↓ SPPF
    ↓
[Neck - Feature Fusion]
    ↓ C2PSA (Attention)
    ↓ PAN Structure
    ↓ Multi-scale Features
    ↓
[Head - Detection]
    ↓ Decoupled Head
    ↓ Classification + Regression
    ↓
Output (Boxes, Classes, Scores)
```

### 주요 모듈

#### 1. C3k2 (CSP Bottleneck with 2 convolutions - k2)
```python
# C3k2 구조 개념
class C3k2:
    def __init__(self, in_channels, out_channels):
        # Split channels
        # Bottleneck layers
        # Concat and Conv
        pass
```
**특징:**
- CSP 구조 기반
- 효율적인 특징 추출
- 파라미터 감소

#### 2. C2PSA (C2 with Partial Self-Attention)
```python
# C2PSA 구조 개념
class C2PSA:
    def __init__(self, channels):
        # Partial Self-Attention
        # Channel split
        # Attention on subset
        pass
```
**특징:**
- 부분 Self-Attention 사용
- 계산량 감소
- 장거리 의존성 포착

#### 3. SPPF (Spatial Pyramid Pooling - Fast)
- 다중 스케일 특징 추출
- 빠른 처리 속도
- 수용 영역(Receptive Field) 확대

---

## 💻 사용 예제

### 객체 탐지 (Object Detection)

```python
from ultralytics import YOLO
import cv2

# 모델 로드
model = YOLO('yolov11n.pt')

# 단일 이미지 예측
results = model('bus.jpg')

# 여러 이미지 예측
results = model(['image1.jpg', 'image2.jpg', 'image3.jpg'])

# 배치 예측 (더 빠름)
results = model(['image1.jpg', 'image2.jpg'], batch=2)

# 결과 처리
for result in results:
    # 박스 정보
    boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
    confidences = result.boxes.conf.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy()
    
    # 이미지에 그리기
    img = result.orig_img
    for box, conf, cls in zip(boxes, confidences, class_ids):
        x1, y1, x2, y2 = map(int, box)
        label = f'{model.names[int(cls)]} {conf:.2f}'
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imwrite('output.jpg', img)
```

### 비디오 스트림 처리

```python
from ultralytics import YOLO
import cv2

model = YOLO('yolov11n.pt')

# 웹캠 열기
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 예측 (stream=True로 메모리 효율성 향상)
    results = model(frame, stream=True)
    
    # 결과 시각화
    for result in results:
        annotated_frame = result.plot()
        cv2.imshow('YOLOv11', annotated_frame)
    
    # 'q' 키로 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 특정 클래스만 탐지

```python
from ultralytics import YOLO

model = YOLO('yolov11n.pt')

# 사람(class 0)과 자동차(class 2)만 탐지
results = model('street.jpg', classes=[0, 2])

# 또는 특정 클래스 제외
results = model('street.jpg', classes=[0, 1, 2, 3])  # 0~3번 클래스만
```

### 신뢰도 임계값 설정

```python
from ultralytics import YOLO

model = YOLO('yolov11n.pt')

# 신뢰도 0.5 이상만 표시
results = model('image.jpg', conf=0.5)

# IoU 임계값 조정 (NMS)
results = model('image.jpg', iou=0.7)

# 동시 설정
results = model('image.jpg', conf=0.4, iou=0.6)
```

### 이미지 크기 조정

```python
from ultralytics import YOLO

model = YOLO('yolov11n.pt')

# 기본 크기 (640)
results = model('image.jpg')

# 사용자 정의 크기 (더 큰 이미지 = 더 높은 정확도, 느린 속도)
results = model('image.jpg', imgsz=1280)

# 작은 이미지로 빠른 처리
results = model('image.jpg', imgsz=320)
```

---

## 🎓 학습 (Training)

### 기본 학습

```python
from ultralytics import YOLO

# 사전 학습된 모델에서 시작 (전이 학습)
model = YOLO('yolov11n.pt')

# 학습 시작
results = model.train(
    data='coco.yaml',      # 데이터셋 설정 파일
    epochs=100,            # 에포크 수
    imgsz=640,             # 이미지 크기
    batch=16,              # 배치 크기
    name='yolov11_custom', # 실험 이름
    device=0,              # GPU 번호 (0, 1, 2, ... 또는 'cpu')
)
```

### CLI로 학습

```bash
yolo detect train \
    data=coco.yaml \
    model=yolov11n.pt \
    epochs=100 \
    imgsz=640 \
    batch=16 \
    device=0 \
    project=runs/train \
    name=exp
```

### 커스텀 데이터셋 준비

#### 1. 데이터셋 구조

```
dataset/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── val/
│       ├── image3.jpg
│       └── image4.jpg
└── labels/
    ├── train/
    │   ├── image1.txt
    │   └── image2.txt
    └── val/
        ├── image3.txt
        └── image4.txt
```

#### 2. 라벨 형식 (YOLO Format)

```txt
# image1.txt
# class_id x_center y_center width height (정규화된 값 0~1)
0 0.5 0.5 0.3 0.4
1 0.3 0.7 0.2 0.15
```

#### 3. 데이터셋 YAML 파일

```yaml
# dataset.yaml
path: /path/to/dataset  # 데이터셋 루트 경로
train: images/train     # 학습 이미지 경로
val: images/val         # 검증 이미지 경로
test: images/test       # (선택) 테스트 이미지 경로

# 클래스 정의
names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  # ...

# 클래스 수
nc: 80
```

### 고급 학습 옵션

```python
from ultralytics import YOLO

model = YOLO('yolov11n.pt')

results = model.train(
    # 필수 파라미터
    data='custom.yaml',
    epochs=100,
    
    # 이미지 및 배치
    imgsz=640,
    batch=16,
    
    # 학습률
    lr0=0.01,              # 초기 학습률
    lrf=0.01,              # 최종 학습률 (lr0 * lrf)
    
    # 옵티마이저
    optimizer='SGD',       # SGD, Adam, AdamW
    momentum=0.937,
    weight_decay=0.0005,
    
    # 데이터 증강
    hsv_h=0.015,          # HSV-Hue 증강
    hsv_s=0.7,            # HSV-Saturation 증강
    hsv_v=0.4,            # HSV-Value 증강
    degrees=0.0,          # 이미지 회전 (±도)
    translate=0.1,        # 이미지 이동 (±분율)
    scale=0.5,            # 이미지 스케일 (±증감)
    shear=0.0,            # 이미지 전단 (±도)
    perspective=0.0,      # 이미지 원근 (±분율)
    flipud=0.0,           # 상하 뒤집기 (확률)
    fliplr=0.5,           # 좌우 뒤집기 (확률)
    mosaic=1.0,           # Mosaic 증강 (확률)
    mixup=0.0,            # MixUp 증강 (확률)
    
    # 정규화
    dropout=0.0,          # Dropout 비율
    
    # 기타
    patience=50,          # 조기 종료 patience
    save=True,            # 체크포인트 저장
    save_period=10,       # 매 N 에포크마다 저장
    workers=8,            # 데이터 로더 워커 수
    device=0,             # GPU 디바이스
    verbose=True,         # 상세 출력
    
    # 재개 및 전이 학습
    resume=False,         # 이전 학습 재개
    amp=True,             # Automatic Mixed Precision
    
    # 검증
    val=True,             # 매 에포크마다 검증
    
    # 프로젝트 관리
    project='runs/train',
    name='exp',
    exist_ok=False,       # 기존 프로젝트 덮어쓰기
)
```

### 다중 GPU 학습

```bash
# PyTorch DDP (권장)
yolo detect train data=coco.yaml model=yolov11n.pt epochs=100 device=0,1,2,3
```

```python
# Python에서
from ultralytics import YOLO

model = YOLO('yolov11n.pt')
model.train(data='coco.yaml', epochs=100, device=[0, 1, 2, 3])
```

### 학습 중단 및 재개

```python
from ultralytics import YOLO

# 학습 재개 (last.pt에서)
model = YOLO('runs/train/exp/weights/last.pt')
model.train(resume=True)

# 또는 CLI
# yolo detect train resume model=runs/train/exp/weights/last.pt
```

---

## 🔍 추론 (Inference)

### 다양한 소스에서 추론

```python
from ultralytics import YOLO

model = YOLO('yolov11n.pt')

# 1. 이미지 파일
results = model('image.jpg')

# 2. 이미지 URL
results = model('https://example.com/image.jpg')

# 3. NumPy 배열 (OpenCV)
import cv2
img = cv2.imread('image.jpg')
results = model(img)

# 4. PIL Image
from PIL import Image
img = Image.open('image.jpg')
results = model(img)

# 5. 비디오 파일
results = model('video.mp4')

# 6. 웹캠
results = model(0)  # 0은 기본 웹캠

# 7. RTSP/HTTP 스트림
results = model('rtsp://192.168.1.100:554/stream')

# 8. 디렉토리 (모든 이미지)
results = model('path/to/images/')

# 9. 와일드카드
results = model('path/to/*.jpg')
```

### 결과 처리 및 분석

```python
from ultralytics import YOLO

model = YOLO('yolov11n.pt')
results = model('image.jpg')

for result in results:
    # 원본 이미지
    orig_img = result.orig_img
    
    # 박스 정보
    boxes = result.boxes
    
    # 박스 좌표 (다양한 형식)
    xyxy = boxes.xyxy      # [x1, y1, x2, y2]
    xywh = boxes.xywh      # [x_center, y_center, width, height]
    xyxyn = boxes.xyxyn    # 정규화된 xyxy
    xywhn = boxes.xywhn    # 정규화된 xywh
    
    # 신뢰도 및 클래스
    conf = boxes.conf      # 신뢰도 점수
    cls = boxes.cls        # 클래스 ID
    
    # 클래스 이름
    names = result.names
    for c in cls:
        print(names[int(c)])
    
    # 결과 저장
    result.save('output.jpg')
    
    # 결과 표시
    result.show()
    
    # 주석이 달린 이미지
    annotated = result.plot()
    
    # JSON으로 내보내기
    json_results = result.tojson()
    
    # Pandas DataFrame으로
    df = result.pandas().xyxy[0]
    print(df)
```

### 추론 최적화

```python
from ultralytics import YOLO

model = YOLO('yolov11n.pt')

# Half precision (FP16) - GPU에서 더 빠름
model = YOLO('yolov11n.pt')
results = model('image.jpg', half=True)

# 최대 탐지 수 제한
results = model('image.jpg', max_det=100)

# 더 작은 이미지 크기로 빠른 추론
results = model('image.jpg', imgsz=320)

# NMS 파라미터 조정
results = model('image.jpg', conf=0.25, iou=0.45)

# 특정 클래스만 탐지
results = model('image.jpg', classes=[0, 2, 3])  # person, car, motorcycle

# 이미지 증강 테스트 (TTA) - 더 높은 정확도
results = model('image.jpg', augment=True)
```

### 배치 처리

```python
from ultralytics import YOLO
import glob

model = YOLO('yolov11n.pt')

# 이미지 목록 가져오기
images = glob.glob('path/to/images/*.jpg')

# 배치 처리 (더 빠름)
results = model(images, batch=8)

# 결과 저장
for i, result in enumerate(results):
    result.save(f'output_{i}.jpg')
```

---

## 📤 모델 내보내기

### 지원 형식

| 형식 | 명령 | 장점 | 용도 |
|------|------|------|------|
| PyTorch | `format=torchscript` | 원본 정확도 | Python 배포 |
| ONNX | `format=onnx` | 범용성 | 다양한 프레임워크 |
| TensorRT | `format=engine` | 최고 속도 | NVIDIA GPU |
| CoreML | `format=coreml` | iOS 최적화 | Apple 기기 |
| TFLite | `format=tflite` | 경량 | Android/임베디드 |
| OpenVINO | `format=openvino` | Intel CPU 최적화 | Intel 하드웨어 |

### 내보내기 예제

```python
from ultralytics import YOLO

model = YOLO('yolov11n.pt')

# ONNX로 내보내기
model.export(format='onnx')

# TensorRT로 내보내기 (동적 배치)
model.export(format='engine', dynamic=True)

# CoreML로 내보내기 (iOS)
model.export(format='coreml')

# TensorFlow Lite로 내보내기 (INT8 양자화)
model.export(format='tflite', int8=True)

# OpenVINO로 내보내기
model.export(format='openvino')
```

### CLI로 내보내기

```bash
# ONNX
yolo export model=yolov11n.pt format=onnx

# TensorRT (FP16)
yolo export model=yolov11n.pt format=engine half=True

# CoreML
yolo export model=yolov11n.pt format=coreml

# INT8 양자화된 TFLite
yolo export model=yolov11n.pt format=tflite int8=True
```

### 내보낸 모델 사용

```python
from ultralytics import YOLO

# ONNX 모델 로드 및 추론
model = YOLO('yolov11n.onnx')
results = model('image.jpg')

# TensorRT 엔진 사용
model = YOLO('yolov11n.engine')
results = model('image.jpg')
```

---

## 📊 성능 벤치마크

### COCO Dataset 성능

| 모델 | 크기<br>(pixels) | mAP50-95 | mAP50 | 파라미터<br>(M) | FLOPs<br>(G) | 속도<br>CPU (ms) | 속도<br>T4 (ms) |
|------|-----------------|----------|-------|----------------|--------------|-----------------|----------------|
| YOLOv11n | 640 | 39.5 | 56.1 | 2.6 | 6.5 | 56.1 | 1.5 |
| YOLOv11s | 640 | 47.0 | 63.6 | 9.4 | 21.5 | 90.0 | 2.5 |
| YOLOv11m | 640 | 51.5 | 68.0 | 20.1 | 68.0 | 183.2 | 4.7 |
| YOLOv11l | 640 | 53.4 | 70.0 | 25.3 | 86.9 | 238.6 | 6.2 |
| YOLOv11x | 640 | 54.7 | 71.3 | 56.9 | 194.9 | 462.8 | 11.3 |

### 세그멘테이션 성능

| 모델 | 크기 | mAP50-95 (box) | mAP50-95 (mask) | 속도 (ms) |
|------|------|----------------|-----------------|-----------|
| YOLOv11n-seg | 640 | 38.9 | 32.0 | 1.8 |
| YOLOv11s-seg | 640 | 46.6 | 37.8 | 2.9 |
| YOLOv11m-seg | 640 | 51.5 | 41.5 | 5.1 |
| YOLOv11l-seg | 640 | 53.4 | 42.9 | 6.9 |
| YOLOv11x-seg | 640 | 54.7 | 43.8 | 12.0 |

### 포즈 추정 성능

| 모델 | 크기 | mAP50-95 | mAP50 | 속도 (ms) |
|------|------|----------|-------|-----------|
| YOLOv11n-pose | 640 | 50.0 | 81.0 | 1.7 |
| YOLOv11s-pose | 640 | 58.9 | 86.4 | 2.6 |
| YOLOv11m-pose | 640 | 64.9 | 89.4 | 4.9 |
| YOLOv11l-pose | 640 | 66.1 | 89.9 | 6.4 |
| YOLOv11x-pose | 640 | 69.5 | 91.1 | 11.0 |

---

## ⚙️ 하이퍼파라미터

### 학습 하이퍼파라미터

```yaml
# 기본 학습 설정
lr0: 0.01              # 초기 학습률
lrf: 0.01              # 최종 학습률 (lr0 * lrf)
momentum: 0.937        # SGD 모멘텀/Adam beta1
weight_decay: 0.0005   # 가중치 감쇠
warmup_epochs: 3.0     # 워밍업 에포크
warmup_momentum: 0.8   # 워밍업 초기 모멘텀
warmup_bias_lr: 0.1    # 워밍업 초기 바이어스 lr
box: 7.5               # 박스 손실 가중치
cls: 0.5               # 클래스 손실 가중치
dfl: 1.5               # DFL 손실 가중치
pose: 12.0             # 포즈 손실 가중치 (pose-only)
kobj: 1.0              # 키포인트 객체 손실 가중치 (pose-only)
label_smoothing: 0.0   # 라벨 스무딩 (epsilon)
nbs: 64                # 명목 배치 크기
overlap_mask: True     # 마스크 오버랩 학습 (segment)
mask_ratio: 4          # 마스크 다운샘플 비율 (segment)
dropout: 0.0           # 분류 Dropout (val/train 0.0)
val: True              # 학습 중 검증
```

### 증강 하이퍼파라미터

```yaml
hsv_h: 0.015          # 이미지 HSV-Hue 증강 (fraction)
hsv_s: 0.7            # 이미지 HSV-Saturation 증강 (fraction)
hsv_v: 0.4            # 이미지 HSV-Value 증강 (fraction)
degrees: 0.0          # 이미지 회전 (+/- deg)
translate: 0.1        # 이미지 이동 (+/- fraction)
scale: 0.5            # 이미지 스케일 (+/- gain)
shear: 0.0            # 이미지 전단 (+/- deg)
perspective: 0.0      # 이미지 원근 (+/- fraction), range 0-0.001
flipud: 0.0           # 이미지 상하 뒤집기 (probability)
fliplr: 0.5           # 이미지 좌우 뒤집기 (probability)
bgr: 0.0              # BGR 채널 뒤집기 (probability)
mosaic: 1.0           # 이미지 모자이크 (probability)
mixup: 0.0            # 이미지 믹스업 (probability)
copy_paste: 0.0       # 세그먼트 복사-붙여넣기 (probability)
auto_augment: randaugment  # 자동 증강 정책 (randaugment, autoaugment, augmix)
erasing: 0.4          # 분류 학습 중 랜덤 지우기 (probability, 분류 전용)
crop_fraction: 1.0    # 분류 이미지 자르기 비율 (fraction, 분류 전용)
```

---

## 💡 Tips & Tricks

### 1. 모델 선택 가이드

**작은 객체 탐지:**
- 더 큰 이미지 크기 사용 (`imgsz=1280`)
- YOLOv11l 또는 YOLOv11x 추천

**실시간 추론:**
- YOLOv11n 또는 YOLOv11s 사용
- 작은 이미지 크기 (`imgsz=640` 또는 `imgsz=416`)
- TensorRT로 내보내기

**높은 정확도:**
- YOLOv11x 사용
- 큰 이미지 크기 (`imgsz=1280`)
- TTA (Test Time Augmentation) 활성화

**엣지 디바이스:**
- YOLOv11n 사용
- INT8 양자화
- TFLite 또는 CoreML로 내보내기

### 2. 학습 개선 팁

**데이터셋:**
- 클래스당 최소 1500개 이미지 권장
- 다양한 조명, 각도, 배경 사용
- 데이터 증강 활용

**하이퍼파라미터:**
- 학습률: 배치 크기에 비례하여 조정
- Mosaic 증강: 작은 객체 탐지에 효과적
- MixUp: 과적합 방지

**전이 학습:**
- 항상 사전 학습된 가중치로 시작
- 적은 데이터: 더 적은 에포크, 높은 학습률
- 많은 데이터: 더 많은 에포크, 낮은 학습률

### 3. 추론 최적화

**속도 향상:**
```python
# FP16 사용
results = model('image.jpg', half=True)

# 배치 처리
results = model(images, batch=8)

# 더 작은 이미지 크기
results = model('image.jpg', imgsz=416)

# NMS 최적화
results = model('image.jpg', conf=0.5, iou=0.7, max_det=100)
```

**정확도 향상:**
```python
# 더 큰 이미지 크기
results = model('image.jpg', imgsz=1280)

# TTA 사용
results = model('image.jpg', augment=True)

# 더 낮은 신뢰도 임계값
results = model('image.jpg', conf=0.001)
```

### 4. 커스텀 데이터셋 팁

**라벨링:**
- [Roboflow](https://roboflow.com/) 사용
- [Label Studio](https://labelstud.io/) 사용
- [CVAT](https://www.cvat.ai/) 사용

**데이터 분할:**
- Train: 70-80%
- Validation: 10-20%
- Test: 10-20%

**데이터 품질:**
- 명확한 객체 경계
- 일관된 라벨링 규칙
- 오탐 최소화

### 5. 디버깅

```python
# 학습 시각화
from ultralytics import YOLO

model = YOLO('yolov11n.pt')
results = model.train(
    data='custom.yaml',
    epochs=100,
    plots=True,  # 학습 플롯 생성
    verbose=True  # 상세 로그
)

# TensorBoard
# tensorboard --logdir runs/train
```

**학습 문제 해결:**
- 손실이 감소하지 않음 → 학습률 낮추기
- 과적합 → 데이터 증강, Dropout 증가
- 낮은 mAP → 더 긴 학습, 더 큰 모델

---

## ❓ FAQ

### Q1: YOLOv11과 YOLOv8의 차이점은?
**A:** YOLOv11은 더 적은 파라미터로 더 높은 정확도를 제공합니다. 약 22% 빠른 추론 속도와 2-3% 높은 mAP를 달성했습니다.

### Q2: 커스텀 데이터셋으로 학습하는 방법은?
**A:** 데이터를 YOLO 형식으로 준비하고 YAML 파일을 만든 후, `model.train(data='custom.yaml')`로 학습하세요.

### Q3: 어떤 모델을 선택해야 하나요?
**A:** 
- 실시간/모바일: YOLOv11n 또는 YOLOv11s
- 일반 용도: YOLOv11m
- 높은 정확도: YOLOv11l 또는 YOLOv11x

### Q4: GPU 메모리 부족 오류가 발생합니다.
**A:** 배치 크기를 줄이거나 (`batch=8` → `batch=4`) 이미지 크기를 줄이세요 (`imgsz=640` → `imgsz=416`).

### Q5: 학습 속도를 높이는 방법은?
**A:** 
- 더 작은 이미지 크기 사용
- 배치 크기 증가 (GPU 메모리가 충분한 경우)
- Mixed Precision Training (`amp=True`)
- 다중 GPU 사용

### Q6: 작은 객체 탐지를 개선하려면?
**A:**
- 더 큰 이미지 크기 (`imgsz=1280`)
- Mosaic 증강 활성화
- 더 큰 모델 사용 (YOLOv11l, YOLOv11x)

### Q7: 모델을 모바일 기기에 배포하려면?
**A:**
- iOS: CoreML로 내보내기 (`format=coreml`)
- Android: TFLite로 내보내기 (`format=tflite`)
- INT8 양자화로 크기 감소

### Q8: 추론 결과를 JSON으로 저장하려면?
**A:**
```python
results = model('image.jpg')
json_data = results[0].tojson()
import json
with open('results.json', 'w') as f:
    f.write(json_data)
```

### Q9: 특정 클래스만 학습할 수 있나요?
**A:** 네, 데이터셋 YAML 파일에서 필요한 클래스만 정의하면 됩니다.

### Q10: 전이 학습 vs 처음부터 학습?
**A:** 거의 항상 전이 학습을 권장합니다. 사전 학습된 가중치는 더 빠른 수렴과 더 나은 성능을 제공합니다.

---

## 📚 참고 자료

### 공식 문서
- [Ultralytics 공식 문서](https://docs.ultralytics.com/)
- [YOLOv11 GitHub](https://github.com/ultralytics/ultralytics)
- [Ultralytics Hub](https://hub.ultralytics.com/)

### 데이터셋 크기별 하이퍼파라미터

#### 작은 데이터셋 (< 1000 이미지)

```yaml
# 과적합 방지에 중점
lr0: 0.001              # 낮은 학습률
epochs: 200             # 더 많은 에포크
patience: 50            # 긴 patience
dropout: 0.1            # Dropout 추가
label_smoothing: 0.1    # 라벨 스무딩
mosaic: 1.0
mixup: 0.2              # MixUp 활성화
copy_paste: 0.1
# 강한 증강
degrees: 15.0
translate: 0.2
scale: 0.5
hsv_h: 0.02
hsv_s: 0.8
hsv_v: 0.5
```

#### 중간 데이터셋 (1000-10000 이미지)

```yaml
# 기본 설정 사용
lr0: 0.01
epochs: 100
patience: 50
mosaic: 1.0
mixup: 0.0
# 적당한 증강
degrees: 0.0
translate: 0.1
scale: 0.5
```

#### 큰 데이터셋 (> 10000 이미지)

```yaml
# 빠른 수렴에 중점
lr0: 0.01               # 표준 학습률
epochs: 300             # 충분한 학습
patience: 100
mosaic: 1.0
mixup: 0.0
# 약한 증강 (데이터가 충분)
degrees: 0.0
translate: 0.1
scale: 0.5
close_mosaic: 10        # 마지막 10 에포크 mosaic 끄기
```

### 학습률 스케줄러

```yaml
# Linear warmup + Cosine annealing (기본)
warmup_epochs: 3.0
cos_lr: True            # Cosine LR scheduler

# Linear warmup + Linear decay
warmup_epochs: 3.0
cos_lr: False

# One-cycle policy
optimizer: 'Adam'
lr0: 0.001
lrf: 0.1
```

---

## 💡 Tips & Tricks

### 1. 모델 선택 가이드

#### 사용 케이스별 추천

| 사용 케이스 | 추천 모델 | 이미지 크기 | 이유 |
|------------|----------|-----------|------|
| **라즈베리파이** | YOLOv11n | 320-416 | 최소 파라미터, CPU 최적화 |
| **Jetson Nano** | YOLOv11n/s | 416-640 | 제한된 GPU 메모리 |
| **Jetson Xavier** | YOLOv11s/m | 640 | 적절한 성능/속도 균형 |
| **모바일 (iOS)** | YOLOv11s | 640 | CoreML 최적화 |
| **모바일 (Android)** | YOLOv11s | 640 | TFLite INT8 |
| **실시간 CCTV** | YOLOv11m | 640-1280 | 정확도와 속도 균형 |
| **드론 영상** | YOLOv11s/m | 640 | 경량, 배터리 효율 |
| **자율주행** | YOLOv11l/x | 1280 | 높은 정확도 필수 |
| **의료 영상** | YOLOv11x | 1280 | 최고 정확도 |
| **소매 분석** | YOLOv11m | 640 | 실시간 + 정확도 |
| **스포츠 분석** | YOLOv11l | 1280 | 빠른 움직임 추적 |
| **산업 검사** | YOLOv11l/x | 1280 | 정밀 탐지 필요 |
| **얼굴 인식** | YOLOv11m | 640 | 중간 크기 객체 |
| **차량 번호판** | YOLOv11l | 1280 | 작은 텍스트 읽기 |

### 2. 성능 개선 전략

#### 정확도 향상

**1. 데이터 품질 개선**
```python
# 데이터 검증
from ultralytics import YOLO

model = YOLO('yolov11n.pt')

# 데이터셋 분석
model.val(data='custom.yaml', split='train')

# 잘못된 라벨 찾기
# - mAP가 매우 낮은 이미지
# - 높은 FP/FN 이미지
```

**2. 더 큰 모델 사용**
```python
# n -> s -> m -> l -> x
model = YOLO('yolov11x.pt')  # 최고 성능
```

**3. 더 큰 이미지 크기**
```python
model.train(
    data='custom.yaml',
    imgsz=1280,  # 기본 640에서 증가
    epochs=100
)
```

**4. 더 긴 학습**
```python
model.train(
    data='custom.yaml',
    epochs=300,   # 100에서 증가
    patience=100  # 조기 종료 늦춤
)
```

**5. 데이터 증강 강화**
```python
model.train(
    data='custom.yaml',
    epochs=100,
    # 강한 증강
    mosaic=1.0,
    mixup=0.2,
    copy_paste=0.1,
    degrees=15.0,
    translate=0.2,
    scale=0.5,
    hsv_h=0.02,
    hsv_s=0.8,
    hsv_v=0.5
)
```

**6. 앙상블**
```python
from ultralytics import YOLO
import numpy as np

# 여러 모델 로드
models = [
    YOLO('yolov11m.pt'),
    YOLO('yolov11l.pt'),
    YOLO('yolov11x.pt')
]

# 결과 수집
all_results = []
for model in models:
    results = model('image.jpg')
    all_results.append(results[0].boxes)

# NMS로 병합 (직접 구현 필요)
# WBF (Weighted Boxes Fusion) 권장
```

#### 속도 향상

**1. 더 작은 모델**
```python
model = YOLO('yolov11n.pt')  # 가장 빠름
```

**2. 더 작은 이미지**
```python
results = model('image.jpg', imgsz=416)  # 640 -> 416
```

**3. TensorRT 사용**
```python
# 내보내기
model.export(format='engine', half=True)

# 사용 (5-10배 빠름)
trt_model = YOLO('yolov11n.engine')
results = trt_model('image.jpg')
```

**4. 배치 처리**
```python
# 단일 처리
for img in images:
    results = model(img)  # 느림

# 배치 처리
results = model(images, batch=16)  # 빠름
```

**5. FP16 사용**
```python
results = model('image.jpg', half=True, device=0)
```

**6. NMS 최적화**
```python
results = model('image.jpg',
                conf=0.5,      # 높은 임계값
                iou=0.7,       # 높은 IoU
                max_det=100)   # 최대 탐지 제한
```

### 3. 일반적인 문제 해결

#### 문제: 낮은 mAP

**원인 및 해결:**

1. **데이터 부족**
   - 해결: 클래스당 최소 1500개 이미지
   - 데이터 증강 강화
   - 온라인 데이터셋 추가

2. **잘못된 라벨**
   - 해결: 라벨 검증
   - 경계 박스 정확도 확인
   - 일관된 라벨링 기준

3. **클래스 불균형**
   - 해결: 클래스 가중치 조정
   - 오버샘플링
   - Focal Loss 사용

4. **너무 짧은 학습**
   - 해결: 더 많은 에포크
   - patience 증가

5. **부적절한 하이퍼파라미터**
   - 해결: 학습률 조정
   - 증강 파라미터 튜닝

#### 문제: 과적합

**증상:**
- Train mAP 높음, Val mAP 낮음
- Train loss 계속 감소, Val loss 증가

**해결:**

```python
model.train(
    data='custom.yaml',
    epochs=100,
    # 과적합 방지
    dropout=0.1,
    label_smoothing=0.1,
    # 강한 증강
    mosaic=1.0,
    mixup=0.2,
    degrees=15.0,
    translate=0.2,
    # 정규화
    weight_decay=0.001,
    # 더 작은 모델
    model='yolov11s.pt'
)
```

#### 문제: 학습이 수렴하지 않음

**증상:**
- Loss가 감소하지 않음
- mAP가 매우 낮음

**해결:**

1. **학습률 낮추기**
```python
model.train(
    data='custom.yaml',
    lr0=0.001,  # 0.01에서 감소
    lrf=0.001
)
```

2. **워밍업 늘리기**
```python
model.train(
    data='custom.yaml',
    warmup_epochs=5.0  # 3.0에서 증가
)
```

3. **배치 크기 증가**
```python
model.train(
    data='custom.yaml',
    batch=32  # 16에서 증가
)
```

4. **데이터 확인**
- 라벨 형식 검증
- 경로 확인
- 이미지 로드 테스트

#### 문제: GPU 메모리 부족

**해결:**

```python
# 1. 배치 크기 감소
model.train(batch=8)  # 16 -> 8

# 2. 이미지 크기 감소
model.train(imgsz=416)  # 640 -> 416

# 3. 더 작은 모델
model = YOLO('yolov11n.pt')

# 4. 그래디언트 누적 (구현 필요)
# 5. 혼합 정밀도
model.train(amp=True)

# 6. 워커 수 감소
model.train(workers=4)

# 7. 캐시 비우기
import torch
torch.cuda.empty_cache()
```

#### 문제: 작은 객체 탐지 실패

**해결:**

```python
model.train(
    data='custom.yaml',
    imgsz=1280,        # 더 큰 이미지
    mosaic=1.0,        # Mosaic 활성화
    copy_paste=0.1,    # Copy-paste
    model='yolov11l.pt'  # 더 큰 모델
)

# 추론 시
results = model('image.jpg',
                imgsz=1280,
                conf=0.3,      # 낮은 임계값
                augment=True)  # TTA
```

### 4. 커스텀 데이터셋 준비 체크리스트

**✅ 데이터 수집**
- [ ] 클래스당 최소 1500개 이미지 (이상적으로 5000+)
- [ ] 다양한 조명 조건
- [ ] 다양한 각도와 거리
- [ ] 다양한 배경
- [ ] 실제 사용 환경과 유사

**✅ 라벨링**
- [ ] 일관된 라벨링 기준
- [ ] 정확한 경계 박스
- [ ] 겹치는 객체 처리
- [ ] 부분적으로 가려진 객체 포함
- [ ] 어려운 케이스 포함

**✅ 데이터 분할**
- [ ] Train: 70-80%
- [ ] Validation: 10-20%
- [ ] Test: 10-20%
- [ ] 분할 후 클래스 분포 확인

**✅ 형식 확인**
- [ ] YOLO 형식 (class x y w h)
- [ ] 정규화된 좌표 (0-1)
- [ ] 파일명 일치 (image.jpg <-> image.txt)
- [ ] YAML 파일 작성

**✅ 검증**
- [ ] 샘플 이미지 시각화
- [ ] 라벨 정확도 확인
- [ ] 클래스 분포 분석
- [ ] 이상치 제거

### 5. 프로덕션 배포 가이드

#### 클라우드 배포 (AWS/GCP/Azure)

**1. Docker 컨테이너**

```dockerfile
# Dockerfile
FROM ultralytics/ultralytics:latest

COPY yolov11n.pt /app/model.pt
COPY app.py /app/app.py

WORKDIR /app

CMD ["python", "app.py"]
```

```python
# app.py - Flask API
from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np

app = Flask(__name__)
model = YOLO('model.pt')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    results = model(img)
    
    # 결과 변환
    detections = []
    for box in results[0].boxes:
        detections.append({
            'class': model.names[int(box.cls[0])],
            'confidence': float(box.conf[0]),
            'bbox': box.xyxy[0].tolist()
        })
    
    return jsonify(detections)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

**2. FastAPI 서버**

```python
# main.py
from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI()
model = YOLO('yolov11n.pt')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    
    results = model(img)
    
    detections = []
    for box in results[0].boxes:
        detections.append({
            'class': model.names[int(box.cls[0])],
            'confidence': float(box.conf[0]),
            'bbox': box.xyxy[0].tolist()
        })
    
    return {"detections": detections}

# 실행: uvicorn main:app --host 0.0.0.0 --port 8000
```

#### 엣지 디바이스 배포

**1. Raspberry Pi**

```python
# rpi_inference.py
from ultralytics import YOLO
import cv2

# TFLite 모델 사용 (CPU 최적화)
model = YOLO('yolov11n.tflite')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 추론
    results = model(frame, imgsz=320)  # 작은 크기
    
    # 표시
    annotated = results[0].plot()
    cv2.imshow('YOLOv11', annotated)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

**2. NVIDIA Jetson**

```python
# jetson_inference.py
from ultralytics import YOLO

# TensorRT 엔진 사용 (GPU 가속)
model = YOLO('yolov11n.engine')

# 실시간 처리
results = model(0, stream=True)  # 웹캠

for result in results:
    result.show()
```

#### 모바일 배포

**iOS (Swift):**

```swift
import CoreML
import Vision

class YOLOv11Detector {
    let model: VNCoreMLModel
    
    init() {
        let mlModel = try! yolov11n()
        self.model = try! VNCoreMLModel(for: mlModel.model)
    }
    
    func detect(image: CGImage, completion: @escaping ([VNRecognizedObjectObservation]) -> Void) {
        let request = VNCoreMLRequest(model: model) { request, error in
            guard let results = request.results as? [VNRecognizedObjectObservation] else {
                completion([])
                return
            }
            completion(results)
        }
        
        request.imageCropAndScaleOption = .scaleFill
        
        let handler = VNImageRequestHandler(cgImage: image, options: [:])
        try? handler.perform([request])
    }
}
```

**Android (Kotlin):**

```kotlin
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer

class YOLOv11Detector(modelPath: String) {
    private val interpreter: Interpreter
    
    init {
        interpreter = Interpreter(File(modelPath))
    }
    
    fun detect(bitmap: Bitmap): List<Detection> {
        // 전처리
        val input = preprocessImage(bitmap)
        
        // 추론
        val output = Array(1) { Array(8400) { FloatArray(84) } }
        interpreter.run(input, output)
        
        // 후처리
        return postprocess(output[0])
    }
    
    private fun preprocessImage(bitmap: Bitmap): ByteBuffer {
        // 640x640으로 리사이즈 및 정규화
        // ...
    }
    
    private fun postprocess(output: Array<FloatArray>): List<Detection> {
        // NMS 적용 및 Detection 객체 생성
        // ...
    }
}
```

### 6. 실전 예제

#### 예제 1: 실시간 교통 분석

```python
from ultralytics import YOLO
import cv2
from collections import defaultdict

model = YOLO('yolov11m.pt')

# 관심 영역 (ROI) 정의
roi_line = [(300, 400), (900, 400)]

# 카운터
vehicle_count = defaultdict(int)
tracked_ids = set()

# 비디오 처리
cap = cv2.VideoCapture('traffic.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 추적 모드 (객체 ID 유지)
    results = model.track(frame, persist=True, classes=[2, 3, 5, 7])  # car, motorcycle, bus, truck
    
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        classes = results[0].boxes.cls.int().cpu().tolist()
        
        for box, track_id, cls in zip(boxes, track_ids, classes):
            x1, y1, x2, y2 = box
            center_y = (y1 + y2) / 2
            
            # ROI 라인을 넘었는지 확인
            if track_id not in tracked_ids and center_y > roi_line[0][1]:
                vehicle_count[model.names[cls]] += 1
                tracked_ids.add(track_id)
    
    # ROI 라인 그리기
    cv2.line(frame, roi_line[0], roi_line[1], (0, 255, 0), 2)
    
    # 카운트 표시
    y_offset = 30
    for vehicle_type, count in vehicle_count.items():
        cv2.putText(frame, f"{vehicle_type}: {count}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        y_offset += 40
    
    cv2.imshow('Traffic Analysis', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

#### 예제 2: PPE (개인 보호 장비) 감지

```python
from ultralytics import YOLO
import cv2

# 커스텀 PPE 모델
model = YOLO('ppe_yolov11m.pt')  # 헬멧, 조끼, 안전화 등

# 클래스 정의
REQUIRED_PPE = ['helmet', 'vest', 'safety_shoes']
ALERT_THRESHOLD = 0.5

def check_ppe_compliance(results, frame):
    """PPE 착용 여부 확인"""
    detected_ppe = set()
    non_compliant = False
    
    for box in results[0].boxes:
        cls = int(box.cls[0])
        class_name = model.names[cls]
        conf = float(box.conf[0])
        
        if conf > ALERT_THRESHOLD:
            detected_ppe.add(class_name)
            
            # 박스 그리기
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = (0, 255, 0) if class_name in REQUIRED_PPE else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # 미착용 항목 확인
    missing_ppe = set(REQUIRED_PPE) - detected_ppe
    if missing_ppe:
        non_compliant = True
        warning = f"WARNING: Missing {', '.join(missing_ppe)}"
        cv2.putText(frame, warning, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return frame, non_compliant

# 비디오 처리
cap = cv2.VideoCapture('construction_site.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame)
    annotated_frame, alert = check_ppe_compliance(results, frame)
    
    if alert:
        # 알림 전송 (이메일, SMS 등)
        pass
    
    cv2.imshow('PPE Detection', annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

#### 예제 3: 얼굴 감지 및 블러 처리 (프라이버시)

```python
from ultralytics import YOLO
import cv2

model = YOLO('yolov11n.pt')

def blur_faces(frame, results):
    """얼굴 영역 블러 처리"""
    for box in results[0].boxes:
        cls = int(box.cls[0])
        
        # 사람(class 0)만 처리
        if cls == 0:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # 얼굴 영역 추정 (상체 상단 1/3)
            face_h = (y2 - y1) // 3
            face_region = frame[y1:y1+face_h, x1:x2]
            
            # 블러 적용
            if face_region.size > 0:
                blurred = cv2.GaussianBlur(face_region, (99, 99), 30)
                frame[y1:y1+face_h, x1:x2] = blurred
    
    return frame

# 비디오 처리
cap = cv2.VideoCapture('input.mp4')
out = cv2.VideoWriter('output_blurred.mp4', 
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      30, 
                      (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame, classes=[0])  # person만
    blurred_frame = blur_faces(frame, results)
    
    out.write(blurred_frame)
    cv2.imshow('Privacy Protection', blurred_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
```

---

## ❓ FAQ

### Q1: YOLOv11과 YOLOv8의 차이점은?
**A:** YOLOv11은 YOLOv8 대비:
- 파라미터 19-22% 감소
- mAP 2-3% 향상
- 추론 속도 약 22% 개선
- 새로운 C3k2, C2PSA 모듈 도입
- API는 동일하여 쉬운 마이그레이션

### Q2: 커스텀 데이터셋으로 학습하는 방법은?
**A:** 
1. YOLO 형식으로 데이터 준비 (images/, labels/)
2. YAML 파일 작성 (경로, 클래스 정의)
3. `model.train(data='custom.yaml')` 실행
4. 최소 클래스당 1500개 이미지 권장

### Q3: 어떤 모델 크기를 선택해야 하나요?
**A:**
- **실시간/모바일**: YOLOv11n 또는 YOLOv11s
- **일반 용도**: YOLOv11m
- **높은 정확도**: YOLOv11l 또는 YOLOv11x
- **엣지 디바이스**: YOLOv11n + INT8 양자화

### Q4: GPU 메모리 부족 오류가 발생합니다.
**A:** 
```python
# 해결 방법:
model.train(
    batch=8,    # 배치 크기 감소 (16 -> 8)
    imgsz=416,  # 이미지 크기 감소 (640 -> 416)
    workers=4,  # 워커 수 감소
    amp=True    # Mixed Precision
)
```

### Q5: 학습 속도를 높이는 방법은?
**A:**
- 다중 GPU 사용: `device=[0,1,2,3]`
- Mixed Precision: `amp=True`
- 더 큰 배치: `batch=32` (GPU 허용 시)
- 더 많은 워커: `workers=16`
- 더 작은 이미지: `imgsz=416`

### Q6: 작은 객체 탐지를 개선하려면?
**A:**
```python
model.train(
    imgsz=1280,        # 더 큰 이미지
    mosaic=1.0,        # Mosaic 증강
    copy_paste=0.1,    # Copy-paste
    model='yolov11l.pt'  # 더 큰 모델
)

# 추론 시
results = model('image.jpg', imgsz=1280, conf=0.3)
```

### Q7: 모델을 모바일 기기에 배포하려면?
**A:**
```python
# iOS
model.export(format='coreml', int8=True)

# Android
model.export(format='tflite', int8=True)

# 크기: PyTorch (10MB) -> TFLite INT8 (2.5MB)
```

### Q8: 추론 결과를 JSON으로 저장하려면?
**A:**
```python
results = model('image.jpg')
json_data = results[0].tojson()

import json
with open('results.json', 'w') as f:
    json.dump(json.loads(json_data), f, indent=2)
```

### Q9: 특정 클래스만 학습할 수 있나요?
**A:** 네, YAML 파일에서 원하는 클래스만 정의하면 됩니다:
```yaml
names:
  0: person
  1: car
nc: 2
```

### Q10: 전이 학습 vs 처음부터 학습?
**A:** **항상 전이 학습 권장!**
```python
# 전이 학습 (권장)
model = YOLO('yolov11n.pt')
model.train(data='custom.yaml')

# 처음부터 (비권장)
model = YOLO('yolov11n.yaml')  # 구조만
model.train(data='custom.yaml', epochs=500)
```

### Q11: 학습 중 과적합을 방지하려면?
**A:**
```python
model.train(
    data='custom.yaml',
    dropout=0.1,
    label_smoothing=0.1,
    mosaic=1.0,
    mixup=0.2,
    weight_decay=0.001,
    patience=30
)
```

### Q12: 여러 GPU로 학습하는 방법은?
**A:**
```bash
# DDP (권장)
yolo detect train data=custom.yaml model=yolov11n.pt device=0,1,2,3

# 또는 Python
model.train(data='custom.yaml', device=[0,1,2,3])
```

### Q13: 학습을 재개하려면?
**A:**
```python
# 자동 재개
model = YOLO('runs/detect/train/weights/last.pt')
model.train(resume=True)
```

### Q14: 탐지 신뢰도를 높이려면?
**A:**
```python
# 신뢰도 임계값 조정
results = model('image.jpg', conf=0.7)  # 기본 0.25

# NMS IoU 조정
results = model('image.jpg', conf=0.5, iou=0.7)
```

### Q15: YOLOv11로 영상 추적이 가능한가요?
**A:** 네!
```python
# 객체 추적 (BoT-SORT, ByteTrack)
results = model.track('video.mp4', persist=True)

for result in results:
    if result.boxes.id is not None:
        track_ids = result.boxes.id.int().cpu().tolist()
        # ID별 추적 처리
```

---

## 📚 참고 자료

### 공식 문서
- [Ultralytics 공식 문서](https://docs.ultralytics.com/)
- [YOLOv11 GitHub](https://github.com/ultralytics/ultralytics)
- [Ultralytics Hub](https://hub.ultralytics.com/)
- [API Reference](https://docs.ultralytics.com/reference/)

### 데이터셋
- [COCO Dataset](https://cocodataset.org/)
- [Open Images](https://storage.googleapis.com/openimages/web/index.html)
- [Roboflow Universe](https://universe.roboflow.com/)
- [ImageNet](https://www.image-net.org/)
- [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/)

### 튜토리얼 및 가이드
- [YOLOv11 Quick Start](https://docs.ultralytics.com/quickstart/)
- [Custom Training Guide](https://docs.ultralytics.com/modes/train/)
- [Model Export Guide](https://docs.ultralytics.com/modes/export/)
- [Prediction Guide](https://docs.ultralytics.com/modes/predict/)
- [Validation Guide](https://docs.ultralytics.com/modes/val/)

### 라벨링 도구
- [Roboflow](https://roboflow.com/) - 웹 기반, 자동 변환
- [Label Studio](https://labelstud.io/) - 오픈소스
- [CVAT](https://www.cvat.ai/) - 비디오 지원
- [LabelImg](https://github.com/HumanSignal/labelImg) - 간단한 데스크톱
- [Makesense.ai](https://www.makesense.ai/) - 온라인 무료

### 커뮤니티
- [Ultralytics Discord](https://discord.gg/ultralytics)
- [GitHub Discussions](https://github.com/ultralytics/ultralytics/discussions)
- [Stack Overflow - YOLO](https://stackoverflow.com/questions/tagged/yolo)
- [Reddit r/computervision](https://www.reddit.com/r/computervision/)

### 논문 및 연구
- [YOLOv11 Technical Report](https://docs.ultralytics.com/) (출시 예정)
- [YOLOv8 Paper](https://arxiv.org/abs/...)
- [YOLOv7](https://arxiv.org/abs/2207.02696)
- [YOLO Series Overview](https://arxiv.org/search/?query=YOLO&searchtype=all)

### 블로그 및 기사
- [Ultralytics Blog](https://www.ultralytics.com/blog)
- [Roboflow Blog - YOLO](https://blog.roboflow.com/tag/yolo/)
- [Towards Data Science - YOLO](https://towardsdatascience.com/tagged/yolo)

### 비디오 튜토리얼
- [Ultralytics YouTube](https://www.youtube.com/ultralytics)
- [YOLOv11 Tutorial Playlist](https://www.youtube.com/playlist?list=...)

### 도구 및 라이브러리
- [PyTorch](https://pytorch.org/)
- [OpenCV](https://opencv.org/)
- [TensorRT](https://developer.nvidia.com/tensorrt)
- [ONNX](https://onnx.ai/)
- [CoreML Tools](https://github.com/apple/coremltools)

---

## 📝 라이선스

YOLOv11은 두 가지 라이선스로 제공됩니다:

### AGPL-3.0 License
- **오픈소스 사용**: 무료
- **조건**: 소스 코드 공개 필요
- **용도**: 연구, 교육, 개인 프로젝트

### Enterprise License
- **상업적 사용**: 유료
- **조건**: 소스 코드 비공개 가능
- **용도**: 상업 제품, SaaS, 클로즈드 소스

자세한 내용은 [Ultralytics 라이선스 페이지](https://ultralytics.com/license)를 참조하세요.

---

## 🤝 기여

기여는 언제나 환영합니다!

### 기여 방법
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing`)
5. Open a Pull Request

### 코드 스타일
- PEP 8 준수
- 타입 힌트 사용
- Docstring 작성
- 테스트 코드 포함

### 이슈 리포팅
- [GitHub Issues](https://github.com/ultralytics/ultralytics/issues)
- 명확한 제목과 설명
- 재현 가능한 예제
- 환경 정보 (OS, Python, PyTorch 버전)

---

## 🙏 감사의 말

이 프로젝트는 다음 분들의 기여로 만들어졌습니다:
- Ultralytics 팀
- YOLO 커뮤니티
- 오픈소스 기여자들

---

## 📧 문의

- **Email**: hello@ultralytics.com
- **GitHub**: [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- **Discord**: [Ultralytics 커뮤니티](https://discord.gg/ultralytics)
- **Twitter**: [@ultralytics](https://twitter.com/ultralytics)
- **LinkedIn**: [Ultralytics](https://www.linkedin.com/company/ultralytics/)

---

## 📈 업데이트 로그

### v1.0.0 (2024-11)
- YOLOv11 완전 가이드 초판 작성
- YOLOv8 비교 추가
- 용어 정리 추가
- 실전 예제 추가

---

<div align="center">

**마지막 업데이트**: 2024년 11월

**제작**: Ultralytics

**버전**: YOLOv11 (2024)

---

Made with ❤️ by the Ultralytics Team

[⬆ 맨 위로](#yolov11-완전-가이드-complete-guide)

</div>셋
- [COCO Dataset](https://cocodataset.org/)
- [Open Images](https://storage.googleapis.com/openimages/web/index.html)
- [Roboflow Universe](https://universe.roboflow.com/)

### 튜토리얼
- [YOLOv11 Quick Start](https://docs.ultralytics.com/quickstart/)
- [Custom Training Guide](https://docs.ultralytics.com/modes/train/)
- [Model Export Guide](https://docs.ultralytics.com/modes/export/)

### 커뮤니티
- [Ultralytics Discord](https://discord.gg/ultralytics)
- [GitHub Discussions](https://github.com/ultralytics/ultralytics/discussions)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/yolo)

### 논문
- [YOLOv11 Technical Report](https://arxiv.org/abs/...)
- [YOLOv8 Paper](https://arxiv.org/abs/...)

---

## 📝 라이선스

YOLOv11은 두 가지 라이선스로 제공됩니다:

- **AGPL-3.0 License**: 오픈소스 사용
- **Enterprise License**: 상업적 사용

자세한 내용은 [Ultralytics 라이선스 페이지](https://ultralytics.com/license)를 참조하세요.

---

## 🤝 기여

기여는 언제나 환영합니다! 이슈나 풀 리퀘스트를 자유롭게 제출해주세요.

---

## 📧 문의

- **Email**: hello@ultralytics.com
- **GitHub Issues**: [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics/issues)
- **Discord**: [Ultralytics 커뮤니티](https://discord.gg/ultralytics)

---

**마지막 업데이트**: 2024년 11월

**제작**: Ultralytics

**버전**: YOLOv11 (2024)
