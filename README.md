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

### 데이터셋
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
# YOLOv11 vs YOLOv8 비교 및 용어 정리

## 📊 YOLOv11 vs YOLOv8 주요 차이점

| 구분 | YOLOv8 | YOLOv11 | 개선사항 |
|------|--------|---------|----------|
| **출시일** | 2023년 1월 | 2024년 9월 | - |
| **백본 구조** | CSPDarknet with C2f | C3k2, C2PSA | 더 효율적인 특징 추출 |
| **파라미터 수 (N)** | 3.2M | 2.6M | 약 19% 감소 |
| **파라미터 수 (S)** | 11.2M | 9.4M | 약 16% 감소 |
| **파라미터 수 (M)** | 25.9M | 20.1M | 약 22% 감소 |
| **mAP (N)** | 37.3% | 39.5% | +2.2% 향상 |
| **mAP (S)** | 44.9% | 47.0% | +2.1% 향상 |
| **mAP (M)** | 50.2% | 51.5% | +1.3% 향상 |
| **추론 속도** | 기준 | 약 22% 빠름 | 속도 개선 |
| **Neck 구조** | PAN (Path Aggregation Network) | C2PSA 기반 개선된 구조 | 다중 스케일 특징 융합 강화 |
| **Head 구조** | Decoupled Head | Decoupled Head (개선) | 분류/회귀 분리 최적화 |
| **학습 안정성** | 양호 | 개선됨 | 더 안정적인 수렴 |

## 🏆 성능 비교 (COCO Dataset)

### 모델별 상세 비교

| 모델 | 파라미터 | FLOPs | mAP50-95 | 속도 (ms) | 용도 |
|------|----------|-------|----------|-----------|------|
| **YOLOv8n** | 3.2M | 8.7G | 37.3% | 1.2 | 경량 엣지 디바이스 |
| **YOLOv11n** | 2.6M | 6.5G | 39.5% | 1.0 | 경량 엣지 디바이스 (개선) |
| **YOLOv8s** | 11.2M | 28.6G | 44.9% | 2.1 | 모바일/임베디드 |
| **YOLOv11s** | 9.4M | 21.5G | 47.0% | 1.7 | 모바일/임베디드 (개선) |
| **YOLOv8m** | 25.9M | 78.9G | 50.2% | 3.6 | 일반 용도 |
| **YOLOv11m** | 20.1M | 68.0G | 51.5% | 2.9 | 일반 용도 (개선) |
| **YOLOv8l** | 43.7M | 165.2G | 52.9% | 5.5 | 고성능 |
| **YOLOv11l** | 25.3M | 86.9G | 53.4% | 4.1 | 고성능 (경량화) |
| **YOLOv8x** | 68.2M | 257.8G | 53.9% | 7.8 | 최고 성능 |
| **YOLOv11x** | 56.9M | 194.9G | 54.7% | 6.5 | 최고 성능 (최적화) |

*속도는 NVIDIA T4 GPU 기준*

## 🔧 주요 아키텍처 개선사항

### 1. C3k2 모듈
- **YOLOv8**: C2f (CSP Bottleneck with 2 convolutions)
- **YOLOv11**: C3k2 (개선된 CSP 구조)
- **특징**: 더 효율적인 특징 추출, 파라미터 감소

### 2. C2PSA (C2 with Partial Self-Attention)
- **새로운 어텐션 메커니즘** 도입
- **부분 Self-Attention**으로 계산량 감소
- **장거리 의존성** 모델링 개선

### 3. SPPF (Spatial Pyramid Pooling - Fast)
- 두 모델 모두 사용하지만, YOLOv11에서 최적화

---

## 📚 YOLO 용어 정리

### 기본 개념

| 용어 | 설명 | 예시/참고 |
|------|------|-----------|
| **Object Detection** | 이미지에서 객체의 위치(bbox)와 클래스를 동시에 예측 | 사람, 자동차, 고양이 탐지 |
| **Bounding Box (BBox)** | 객체를 둘러싸는 직사각형 영역 | (x, y, width, height) |
| **IoU** | Intersection over Union, 예측 박스와 정답 박스의 겹침 비율 | 0.0 ~ 1.0 값 |
| **NMS** | Non-Maximum Suppression, 중복 박스 제거 | IoU 임계값 기반 |
| **Anchor Box** | 사전 정의된 박스 크기/비율 (YOLOv5 이하) | YOLOv8+는 Anchor-Free |
| **Anchor-Free** | 앵커 박스 없이 직접 객체 위치 예측 | YOLOv8, YOLOv11 |

### 성능 지표

| 용어 | 설명 | 계산 방법 |
|------|------|-----------|
| **Precision** | 예측한 객체 중 실제 객체의 비율 | TP / (TP + FP) |
| **Recall** | 실제 객체 중 정확히 탐지한 비율 | TP / (TP + FN) |
| **mAP** | mean Average Precision, 모든 클래스의 AP 평균 | Σ AP / 클래스 수 |
| **mAP50** | IoU 0.5 기준의 mAP | PASCAL VOC 방식 |
| **mAP50-95** | IoU 0.5~0.95의 mAP 평균 | COCO 방식 (더 엄격) |
| **FPS** | Frames Per Second, 초당 처리 프레임 수 | 1000 / inference_time(ms) |
| **Latency** | 단일 이미지 추론 시간 | 밀리초(ms) 단위 |

### 모델 구조 용어

| 용어 | 설명 | 역할 |
|------|------|------|
| **Backbone** | 입력 이미지에서 특징을 추출하는 네트워크 | CSPDarknet, C3k2 등 |
| **Neck** | 다양한 스케일의 특징을 융합 | PAN, FPN, C2PSA |
| **Head** | 최종 탐지 결과를 출력 | 클래스 분류 + BBox 회귀 |
| **CSP** | Cross Stage Partial, 특징맵을 분할하여 처리 | 계산량 감소 |
| **PAN** | Path Aggregation Network | Bottom-up 경로 추가 |
| **FPN** | Feature Pyramid Network | Top-down 특징 융합 |
| **SPPF** | Spatial Pyramid Pooling Fast | 다중 스케일 풀링 |

### 학습 관련 용어

| 용어 | 설명 | 기본값 예시 |
|------|------|-------------|
| **Epoch** | 전체 데이터셋을 한 번 학습 | 100~300 epochs |
| **Batch Size** | 한 번에 처리하는 이미지 수 | 16, 32, 64 |
| **Learning Rate** | 가중치 업데이트 크기 | 0.01 (초기값) |
| **Image Size** | 입력 이미지 크기 | 640x640 (기본) |
| **Augmentation** | 데이터 증강 기법 | Mosaic, Flip, Scale 등 |
| **Mosaic** | 4개 이미지를 하나로 합성 | YOLOv4에서 도입 |
| **MixUp** | 두 이미지를 혼합 | 일반화 성능 향상 |
| **Warmup** | 초기 학습률을 점진적으로 증가 | 처음 3 epochs |

### 손실 함수 (Loss Functions)

| 용어 | 설명 | 용도 |
|------|------|------|
| **CIoU Loss** | Complete IoU Loss | BBox 회귀 |
| **DFL** | Distribution Focal Loss | 박스 정밀도 개선 |
| **BCE Loss** | Binary Cross Entropy | 클래스 분류 |
| **Focal Loss** | 클래스 불균형 해결 | 어려운 샘플에 집중 |

### 배포 및 최적화

| 용어 | 설명 | 장점 |
|------|------|------|
| **ONNX** | Open Neural Network Exchange | 프레임워크 독립적 |
| **TensorRT** | NVIDIA의 추론 최적화 엔진 | GPU 가속 |
| **OpenVINO** | Intel의 추론 최적화 | CPU 최적화 |
| **CoreML** | Apple의 ML 프레임워크 | iOS/macOS 배포 |
| **TFLite** | TensorFlow Lite | 모바일/임베디드 |
| **INT8 Quantization** | 8비트 정수로 양자화 | 모델 크기/속도 개선 |
| **FP16** | 16비트 부동소수점 | 정확도 유지하며 경량화 |

### 데이터셋 형식

| 용어 | 설명 | 사용처 |
|------|------|--------|
| **COCO Format** | JSON 기반 어노테이션 | MS COCO 데이터셋 |
| **YOLO Format** | 텍스트 기반 (class x y w h) | YOLO 시리즈 학습 |
| **Pascal VOC** | XML 기반 어노테이션 | VOC 데이터셋 |
| **Labelme** | JSON 어노테이션 도구 | 커스텀 데이터셋 제작 |

### 모델 변형

| 변형 | 설명 | 특징 |
|------|------|------|
| **n (nano)** | 가장 작은 모델 | 엣지 디바이스 |
| **s (small)** | 소형 모델 | 모바일 |
| **m (medium)** | 중형 모델 | 일반 용도 |
| **l (large)** | 대형 모델 | 고성능 요구 |
| **x (xlarge)** | 최대 모델 | 최고 정확도 |
| **-seg** | Segmentation 모델 | Instance Segmentation |
| **-pose** | Pose Estimation 모델 | 키포인트 탐지 |
| **-cls** | Classification 모델 | 이미지 분류 |

---

## 🎯 YOLOv11 선택 가이드

### 사용 케이스별 추천

| 사용 케이스 | 추천 모델 | 이유 |
|------------|----------|------|
| **라즈베리파이, Jetson Nano** | YOLOv11n | 최소 파라미터, 빠른 추론 |
| **모바일 앱 (iOS/Android)** | YOLOv11s | 정확도와 속도 균형 |
| **실시간 CCTV 분석** | YOLOv11m | 적절한 정확도, 실시간 가능 |
| **드론 영상 분석** | YOLOv11s/m | 경량, 배터리 효율적 |
| **자율주행 (고성능)** | YOLOv11l/x | 높은 정확도 요구 |
| **산업 검사 (품질관리)** | YOLOv11m/l | 정밀한 탐지 필요 |
| **의료 영상 분석** | YOLOv11x | 최고 정확도 |

### YOLOv8에서 YOLOv11로 마이그레이션

```python
# YOLOv8 코드
from ultralytics import YOLO
model = YOLO('yolov8n.pt')

# YOLOv11 코드 (동일한 API!)
from ultralytics import YOLO
model = YOLO('yolov11n.pt')

# 사용법은 완전히 동일
results = model('image.jpg')
```

**주요 변경사항:**
- API는 동일 (Ultralytics 통합)
- 모델 가중치만 변경: `yolov8n.pt` → `yolov11n.pt`
- 하이퍼파라미터 조정 권장 (학습률, augmentation 등)

---

## 📈 선택 기준

### YOLOv11을 선택해야 하는 경우:
✅ 최신 성능이 필요할 때  
✅ 파라미터 효율성이 중요할 때  
✅ 실시간 추론 속도가 중요할 때  
✅ 엣지 디바이스 배포 시  

### YOLOv8을 유지해야 하는 경우:
✅ 이미 YOLOv8로 잘 작동하는 시스템이 있을 때  
✅ 검증된 안정성이 필요할 때  
✅ 특정 프레임워크와의 호환성 문제가 있을 때  

---

## 📖 참고 자료

- [YOLOv11 공식 문서](https://docs.ultralytics.com/)
- [YOLOv8 vs YOLOv11 벤치마크](https://github.com/ultralytics/ultralytics)
- [COCO Dataset](https://cocodataset.org/)
- [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)

---

**마지막 업데이트**: 2024년 11월
