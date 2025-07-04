# Recipe-Recommendation-DeepLearning
식재료 기반 딥러닝 레시피 추천 시스템

## 프로젝트 개요
이 프로젝트는 사용자가 가진 남은 식재료를 효율적으로 활용하고, 음식물 쓰레기를 줄이는 데 기여하기 위해 기획되었습니다. 사용자가 직접 찍은 사진 속 식재료를 딥러닝 모델로 인식하고, 인식된 식재료를 기반으로 맞춤형 레시피를 추천해주는 시스템을 개발합니다. 특히, YOLO기반의 객체 인식 모델과 LLM(Large Language Model)을 활용하여 더욱 정확하고 유용한 레시피 추천 서비스를 제공하는 것을 목표로 합니다.

## 목차
1. [프로젝트 목표](# -프로젝트-목표)
2. [데이터 수집 및 라벨링](# -데이터-수집-및-라벨링)
3. [모델링(YOLO)](# -모델링-yolo)
4. [LLM 모델 적용](# -llm-모델-적용)
5. [예상되는 기대효과 및 한계](# -예상되는-기대효과-및-한계)

## 1.프로젝트 목표

### 문제 정의
많은 사람들이 다음과 같은 문제에 직면하고 있습니다:
* 남은 식재료 활용의 어려움 : 냉장고에 남은 식재료가 있어도 어떤 요리를 해야 할지 막막하여 방치되는 경우가 많습니다.
* 음식물 쓰레기 증가 : 사다 놓은 식재료가 기간 내에 소비되지 못하고 버려져 불필요한 음식물 쓰레기가 발생하는 경우가 빈번합니다. 이는 경제적 손실뿐만 아니라 환경 문제로도 이어집니다.

### 프로젝트 목표
위와 같은 문제들을 해결하기 위해, 본 프로젝트는 다음과 같은 목표를 설정하였습니다:
* 식재료 인식 및 레시피 추천 사용자가 스마트폰으로 냉장고 속 식재료를 촬영하면, 사진 속 식재료를 정확하게 인식합니다.
* YOLO 기반 딥러닝 모델 활용 : 객체 인식에 최적화된 YOLO(You Only Look Once) 모델을 활용하여 식재료를 빠르고 정확하게 탐지합니다.
* LLM 연동 : 인식된 식재료를 기반으로 다양한 레시피를 생성하고 추천하기 위해 LLM을 연동하여 사용자에게 맞춤형 요리 아이디어를 제공합니다.
* 효율적인 식재료 소비 유도 : 사용자에게 실용적인 레시피를 제공함으로써 식재료 활용도를 높이고, 음식물 쓰레기를 줄이는 데 기여합니다.

## 2. 데이터 수집 및 라벨링

### 데이터셋 구축 과정
본 프로젝트의 핵심인 식재료 인식 모델 학습을 위해 고품질의 식재료 이미지 데이터셋을 구축하였습니다.
초기에는 직접 웹 크롤링을 통해 데이터를 수집하고 LabelImg를 사용하여 라벨링을 진행할 계획이었습니다. 하지만 LabelImg 사용 중 클래스명 꼬임 현상과 같은 이슈가 발생하여, 효율적인 데이터 확보와 라벨링 품질 유지를 위해 Roboflow 플랫폼으로 전환하여 공개 데이터셋을 활용하고, 직접 크롤링한 한국 식재료 데이터셋도 Roboflow를 통해 관리하는 방향으로 전략을 변경하였습니다.

#### 2.1. Roboflow 공개 데이터셋 활용
주로 다양한 채소와 과일 이미지를 포함하고 있는 Roboflow Universe의 공개 데이터셋을 활용하였습니다.
* 출처 : Combined Vegetables & Fruits Object Detection Dataset by Yolo
* 구성 : Train : 34,008개 이미지
         Validation : 4,619개 이미지
         Test : 3,358개 이미지
         총계 : 41,985개 이미지

#### 2.2. 직접 크롤링한 한국 식재료 데이터셋
한국 요리에 자주 사용되는 식재료의 다양성을 확보하기 위해 직접 웹 크롤링을 통해 이미지를 수집하고 라벨링하여 데이터셋을 보강하였습니다.
* 출처 : 웹 크롤링을 통해 직접 수집 및 라벨링 (My First Project Object Detection Dataset by kim)
* 구성 : Train: 635개 이미지
         Validation: 151개 이미지
         Test: 993개 이미지
         총계 : 1,779개 이미지

#### 2.3. 최종 통합 데이터셋
위 두 데이터셋을 통합하여 최종 학습에 사용된 데이터셋의 규모는 다음과 같습니다.
* Train: 34,643개 이미지 (34,008 + 635)
* Validation: 4,770개 이미지 (4,619 + 151)
* Test: 3,565개 이미지 (3,358 + 993)
* 전체 총계 : 42,978개 이미지

### 라벨링 클래스
본 프로젝트에서는 총 50가지의 다양한 식재료 클래스를 인식하도록 모델을 학습시켰습니다. 학습에 사용된 식재료 클래스는 다음과 같습니다: 
``` 
'almond', 'apple', 'asparagus', 'avocado', 'banana', 'beans', 'beet', 'bell pepper',
'blackberry', 'blueberry', 'broccoli', 'brussels sprouts', 'cabbage', 'carrot',
'cauliflower', 'celery', 'cherry', 'corn', 'cucumber', 'egg', 'eggplant', 'garlic',
'grape', 'green bean', 'green onion', 'hot pepper', 'kiwi', 'lemon', 'lettuce', 'lime',
'mandarin', 'mushroom', 'onion', 'orange', 'pattypan squash', 'pea', 'peach', 'pear',
'pineapple', 'potato', 'pumpkin', 'radish', 'raspberry', 'strawberry', 'tomato',
'vegetable marrow', 'watermelon', 'kimchi', 'seaweed', 'tobu' 
```

### 라벨링 툴 및 과정
데이터 라벨링은 객체 탐지 모델 학습의 핵심 단계로, 정확하고 일관된 라벨링이 모델 성능에 큰 영향을 미칩니다.

* LabelImg 시도 및 한계 : 프로젝트 초기에는 오프라인 라벨링 툴인 LabelImg를 사용하여 직접 수집한 이미지에 대한 라벨링을 시도했습니다. 하지만 대규모 데이터셋 관리 및 클래스명 일관성 유지 과정에서 클래스명 꼬임, 버전 관리의 어려움 등 여러 이슈에 직면했습니다.
* Roboflow로의 전환 : 이러한 문제점을 해결하고 라벨링 효율성을 높이기 위해 Roboflow 플랫폼을 도입했습니다. Roboflow는 웹 기반의 강력한 라벨링 기능을 제공하며, 특히 YOLO 형식의 라벨링 데이터를 효율적으로 관리하고 내보낼 수 있는 장점이 있었습니다. 직접 크롤링한 한국 식재료 데이터셋 또한 Roboflow를 통해 라벨링 및 검수 과정을 거쳤습니다.
* 라벨링 형식 : 모든 라벨링은 YOLO 모델 학습에 필요한 바운딩 박스(Bounding Box) 형식으로 진행되었으며, 각 이미지에 해당하는 `.txt` 파일로 저장되었습니다.

![Image](https://github.com/user-attachments/assets/efde98a3-eeaf-4ba8-8b96-9570a04056bd)  
* 설명 : Labellmg Labeling 예시
![Image](https://github.com/user-attachments/assets/30b74ac6-e1db-46e6-a643-e8c71cd00e2d)
* 설명 : Roboflow Labeling 예시


  
## 3. 모델링 (YOLO)

본 프로젝트에서는 식재료 객체 인식을 위해 **YOLO (You Only Look Once)** 모델을 활용하였습니다. YOLO는 실시간 객체 탐지에 최적화된 딥러닝 모델로, 빠르고 정확한 식재료 인식을 가능하게 합니다. 초기에는 YOLOv5를 사용하여 모델을 구축하고 학습했으며, 이후 YOLOv8l로 확장하여 성능을 개선하였습니다.

### 3.1. YOLOv5 모델 학습 및 추론

식재료 객체 인식 모델 구축을 위해 Ultralytics에서 제공하는 YOLOv5 프레임워크를 활용하였습니다. GitHub 저장소를 클론하여 기본적인 학습 및 추론 스크립트를 사용하고, 커스텀 데이터셋에 맞춰 모델을 학습시켰습니다.

#### 3.1.1. 학습 환경 및 설정

* 모델 버전: YOLOv5 (Ultralytics 공식 GitHub 저장소 활용)
* 학습 데이터: [2.3. 최종 통합 데이터셋](#23-최종-통합-데이터셋)에서 구축한 42,978개의 식재료 이미지 데이터셋
* 환경: Python, PyTorch
* 사용된 스크립트: train.py

train.py 코드 스니펫:

```python
import os

# YOLOv5 설치 및 학습
# os.system('pip install -qr https://github.com/ultralytics/yolov5/releases/latest/download/requirements.txt')
os.system('python yolov5/train.py --img 640 --batch 16 --epochs 50 --data data.yaml --weights yolov5s.pt --project runs/train --name exp')
```
주요 학습 파라미터 설명:
* img 640: 학습 시 이미지 크기를 640x640 픽셀로 설정합니다.
* batch 16: 한 번에 처리할 이미지의 개수(배치 크기)를 16으로 설정합니다.
* epochs 50: 전체 데이터셋을 50번 반복하여 학습합니다.
* data data.yaml: 데이터셋의 경로, 클래스 정보 등이 정의된 data.yaml 파일을 지정합니다.
* weights yolov5s.pt: 사전 학습된 yolov5s.pt 가중치를 사용하여 학습을 시작합니다. 이는 학습 시간을 단축하고 성능을 향상시키는 데 도움이 됩니다.
* project runs/train: 학습 결과가 저장될 상위 디렉토리를 runs/train으로 지정합니다.
* name exp: 현재 학습 세션의 이름을 exp로 지정하여 runs/train/exp 경로에 결과가 저장되도록 합니다.

#### 3.1.2. 모델 추론 및 검증
학습된 모델의 성능을 확인하고 실제 이미지에 대한 객체 탐지 추론을 수행하기 위해 detect.py 및 val.py 스크립트를 활용했습니다.

predict.py 코드 스니펫:

```python

import os

# test 이미지 폴더에 대해 추론 수행
os.system('python yolov5/detect.py --weights runs/train/exp/weights/best.pt --img 640 --conf 0.25 --source yolov5/data_jy/test --save-txt --save-conf --project runs/food_ingredients --exist-ok True')
os.system('python yolov5/val.py --weights runs/train/exp/weights/best.pt --data data.yaml --img 640')
```
주요 추론 및 검증 파라미터 설명:
* weights runs/train/exp/weights/best.pt: 학습을 통해 얻은 최적의 가중치 파일(best.pt)을 사용합니다.
* img 640: 추론 시 이미지 크기를 640x640 픽셀로 설정합니다.
* conf 0.25: 객체 탐지 결과의 신뢰도 임계값을 0.25로 설정합니다. 이 값보다 낮은 신뢰도를 가진 탐지 결과는 무시됩니다.
* source yolov5/data_jy/test: 추론을 수행할 이미지의 경로를 지정합니다. 여기서는 test 폴더의 이미지들을 사용합니다.
* save-txt: 탐지된 객체의 바운딩 박스 좌표와 클래스 정보를 텍스트 파일로 저장합니다.
* save-conf: 탐지된 객체의 신뢰도 점수를 함께 저장합니다.
* project runs/food_ingredients: 추론 결과가 저장될 상위 디렉토리를 지정합니다.
* exist-ok True: 이미 해당 프로젝트 디렉토리가 존재해도 덮어쓰지 않고 진행하도록 합니다.
* yolov5/val.py: 학습된 모델의 성능 지표(mAP, Precision, Recall 등)를 정량적으로 평가하기 위해 사용되는 검증 스크립트입니다.

#### 3.1.3. 데이터셋 구성 파일 (data.yaml)
YOLOv5 학습을 위해 데이터셋 구성 정보를 담은 data.yaml 파일을 사용했습니다. 이 파일은 다음과 같은 주요 정보를 포함합니다:

```yaml

train: /mnt/d/jypark/yolov8_test/data/train/images
val: /mnt/d/jypark/yolov8_test/data/valid/images
test: /mnt/d/jypark/yolov8_test/data/test/images

nc: 50
names: ['almond', 'apple', 'asparagus', 'avocado', 'banana', 'beans', 'beet', 'bell pepper', 
'blackberry', 'blueberry', 'broccoli', 'brussels sprouts', 'cabbage', 'carrot', 
'cauliflower', 'celery', 'cherry', 'corn', 'cucumber', 'egg', 'eggplant', 'garlic', 
'grape', 'green bean', 'green onion', 'hot pepper', 'kiwi', 'lemon', 'lettuce', 'lime', 
'mandarin', 'mushroom', 'onion', 'orange', 'pattypan squash', 'pea', 'peach', 'pear', 
'pineapple', 'potato', 'pumpkin', 'radish', 'raspberry', 'strawberry', 'tomato', 
'vegetable marrow', 'watermelon', 'kimchi', 'seaweed', 'tobu']

roboflow:
  workspace: yolo-jpkho
  project: combined-vegetables-fruits
  version: 8
  license: CC BY 4.0
  url: https://universe.roboflow.com/yolo-jpkho/combined-vegetables-fruits/dataset/8
```
이 설정 파일을 통해 YOLOv5 모델이 학습 과정에서 올바른 데이터셋과 클래스 정보를 참조할 수 있도록 했습니다. 특히 Roboflow에서 제공하는 'Combined Vegetables & Fruits' 데이터셋(버전 8)을 기반으로 하여, 총 50개의 식재료 클래스에 대한 객체 탐지 모델을 학습시켰습니다.


### 3.2. YOLOv8 모델 학습 및 추론
YOLOv5를 통해 기본적인 모델 구축 및 학습을 경험한 후, 더 향상된 성능과 기능을 제공하는 YOLOv8 모델을 도입하여 식재료 인식 모델의 성능을 고도화했습니다. 이 과정에서 다양한 YOLOv8 모델(yolov8l, yolov8n 등)을 실험하고 최적의 모델을 탐색했습니다.

#### 3.2.1. run.py를 통한 학습 및 추론 (yolov8l 중심)
run.py 스크립트는 YOLOv8l 모델의 학습 및 추론, 검증을 통합적으로 수행하기 위해 사용되었습니다.

학습 코드:

```python


from ultralytics import YOLO

model = YOLO('yolov8l.yaml') # yolov8l 모델 구조 로드
model.train(data='/mnt/d/jypark/yolov8_test/data.yaml', epochs=50, imgsz=640, batch=16)
```
주요 학습 파라미터 설명:
* model = YOLO('yolov8l.yaml'): YOLOv8 모델 중 yolov8l (large) 버전을 사용하여 모델을 초기화합니다. 이는 더 큰 모델로, 복잡한 패턴 학습에 유리합니다.
* data='/mnt/d/jypark/yolov8_test/data.yaml': 학습에 사용할 데이터셋의 data.yaml 파일 경로를 지정합니다.
* epochs=50: 전체 데이터셋을 50번 반복하여 학습합니다.
* imgsz=640: 학습 및 추론 시 이미지 크기를 640x640 픽셀로 설정합니다.
* batch=16: 한 번에 처리할 이미지의 개수(배치 크기)를 16으로 설정합니다.

추론 및 검증 코드:

```python

from ultralytics import YOLO

model = YOLO('runs/detect/train/weights/best.pt') # 학습된 모델 가중치 로드
model.predict(source='/mnt/d/jypark/yolov8_test/data/test/images', conf=0.25, save=True, save_txt=True, save_conf=True, project='runs/detect', name='test', exist_ok=True)
model.val(data='data.yaml', split='test', save_json=True, save_txt=True, save_dir='runs/detect/val')
```
주요 추론 및 검증 파라미터 설명:
* model = YOLO('runs/detect/train/weights/best.pt'): 학습이 완료된 모델의 best.pt 가중치 파일을 로드하여 추론 및 검증에 사용합니다.
* source='/mnt/d/jypark/yolov8_test/data/test/images': 추론을 수행할 이미지의 경로를 지정합니다.
* conf=0.25: 객체 탐지 결과의 신뢰도 임계값을 0.25로 설정합니다.
* save=True: 추론 결과를 이미지로 저장합니다.
* save_txt=True: 탐지된 객체의 바운딩 박스 좌표와 클래스 정보를 텍스트 파일로 저장합니다.
* save_conf=True: 탐지된 객체의 신뢰도 점수를 함께 저장합니다.
* project='runs/detect', name='test', exist_ok=True: 추론 결과 저장 경로 및 설정입니다.
* model.val(...): 모델의 성능을 정량적으로 평가하기 위한 검증 명령입니다.
* data='data.yaml', split='test': 검증에 사용할 데이터셋과 스플릿(test set)을 지정합니다.
* save_json=True, save_txt=True, save_dir='runs/detect/val': 검증 결과를 JSON, TXT 파일로 저장하고 저장 디렉토리를 지정합니다.

#### 3.2.2. Jupyter Notebook (test0620.ipynb)을 통한 학습 과정 (yolov8n 실험)
test0620.ipynb는 주로 Google Colab 환경에서 YOLOv8 모델의 학습을 진행하고, 데이터 경로를 동적으로 설정하는 과정을 포함합니다. 이 노트북을 통해 yolov8n (nano) 모델을 실험적으로 학습시켰습니다.

* 실험 목적: yolov8l 모델 학습에 앞서, 더 가벼운 yolov8n 모델의 성능과 학습 효율성을 탐색하기 위해 실험을 진행했습니다.
* 환경 설정: Google Colab 환경에서 Google Drive를 마운트하여 데이터셋에 접근했으며, data.yaml 파일의 데이터 경로를 Colab 환경에 맞게 동적으로 수정하여 학습을 진행했습니다.
* 학습 파라미터: yolov8n.yaml 모델 구조를 사용했으며, epochs=50, imgsz=640, batch=16으로 학습을 수행했습니다.
* 결과 및 인사이트: yolov8n 모델을 성공적으로 학습시키고, 경량 모델로서의 성능과 학습 효율성을 확인할 수 있었습니다. 이는 향후 서비스 배포 시 모델 최적화 및 리소스 효율성을 고려하는 데 중요한 통찰력을 제공했습니다.

환경 설정 및 데이터 경로 동적 변경 코드:

```python

!pip install ultralytics
import os
from google.colab import drive
drive.mount('/content/drive') # Google Drive 마운트
data_dir = '/content/drive/MyDrive/my_ws/Object_Detection/food_dataset2'
data_yaml= '/content/drive/MyDrive/my_ws/Object_Detection/food_dataset2/data.yaml'
import torch
import yaml

print('변경된 yaml 파일 :')
with open(data_yaml) as f:
    film = yaml.load(f, Loader=yaml.FullLoader)
    # display(film) # Jupyter 환경에서 내용 확인용

film['train'] = '/content/drive/MyDrive/my_ws/Object_Detection/food_dataset2/train/images'
film['val'] = '/content/drive/MyDrive/my_ws/Object_Detection/food_dataset2/test/images'

with open(data_yaml, 'w') as f:
    yaml.dump(film, f, default_flow_style=False)
```
주요 환경 설정 및 경로 변경 설명:
* !pip install ultralytics: 필요한 라이브러리를 설치합니다.
* drive.mount('/content/drive'): Google Colab 환경에서 Google Drive를 마운트하여 데이터셋 및 모델에 접근할 수 있도록 합니다.
* data_dir, data_yaml: Google Drive 내의 데이터셋 및 data.yaml 파일의 경로를 정의합니다.
* yaml.load(...), film['train'] = ..., film['val'] = ...: data.yaml 파일을 읽어와 학습 및 검증 데이터셋의 경로를 Google Colab 환경에 맞게 동적으로 수정합니다. 이는 Colab에서 Google Drive의 데이터를 효율적으로 참조하기 위함입니다.
* yaml.dump(...): 변경된 data.yaml 내용을 다시 저장합니다.

YOLOv8n 모델 학습 코드:

```python

from ultralytics import YOLO

model = YOLO('yolov8n.yaml')  # yolov8n 모델 구조 선택
model.train(data='/content/drive/MyDrive/my_ws/Object_Detection/food_dataset2/data.yaml', epochs=50, imgsz=640, batch=16)
```
주요 학습 파라미터 설명:
* model = YOLO('yolov8n.yaml'): YOLOv8 모델 중 yolov8n (nano) 버전을 사용하여 모델을 초기화합니다. yolov8n은 yolov8l보다 작고 빠르며, 경량 모델 구축 및 빠른 실험에 적합합니다.
* data='...': 동적으로 설정된 data.yaml 파일을 사용하여 학습 데이터셋을 지정합니다.
* epochs=50, imgsz=640, batch=16: 학습 반복 횟수, 이미지 크기, 배치 크기를 설정합니다.

학습된 모델 가중치 저장 코드:

```python

import shutil
import os
import glob

weight_paths = sorted(glob.glob('/content/runs/detect/train*/weights/best.pt'), key=os.path.getmtime)

if weight_paths:
    best_weight_path = weight_paths[-1]  # 가장 최근 best.pt
    save_path = '/content/drive/MyDrive/my_ws/food_best.pt'
    shutil.copy(best_weight_path, save_path)
    print(f"✅ best.pt 저장 완료 → {save_path}")
else:
    print("❌ best.pt를 찾지 못했습니다.")
```
주요 가중치 저장 설명:
* glob.glob(...): Colab 환경에서 학습 완료 후 생성된 best.pt 가중치 파일의 경로를 찾습니다. train*/weights/best.pt 패턴을 사용하여 가장 최근 학습된 모델의 가중치를 식별합니다.
* best_weight_path = weight_paths[-1]: 여러 학습 세션이 있을 경우, 가장 최근에 생성된 best.pt 파일을 선택합니다.
* save_path = '/content/drive/MyDrive/my_ws/food_best.pt': 찾은 best.pt 파일을 Google Drive의 지정된 경로로 복사하여 영구적으로 저장합니다. 이는 Colab 세션 종료 후에도 학습된 모델을 보존하고 재활용하기 위함입니다.

## 4. LLM 모델 적용

본 프로젝트는 식재료 객체 인식 모델에서 한 단계 더 나아가, 인식된 식재료를 기반으로 맞춤형 레시피를 제안하기 위해 **거대 언어 모델(Large Language Model, LLM)**을 통합하였습니다. 이를 통해 사용자는 냉장고 속 재료만으로도 다양한 요리 아이디어를 얻고, 음식물 낭비를 줄이는 데 도움을 받을 수 있습니다.

### 4.1. LLM 모델 선정 및 핵심 모듈 구현

레시피 추천 기능을 구현하기 위해 **Microsoft Phi-1.5** 모델을 선정하여 활용했습니다. Phi-1.5는 Microsoft에서 개발한 13억 개의 파라미터를 가진 트랜스포머 기반의 경량 언어 모델로, 복잡한 추론 작업에 강점을 보이며 고품질의 합성 데이터셋으로 학습되어 특정 도메인 작업에 효율적으로 활용될 수 있습니다.

LLM과의 상호작용을 체계적으로 관리하고 재사용성을 높이기 위해 `RecipeRecommender` 클래스를 별도의 핵심 모듈로 구현했습니다. 이 클래스는 LLM 모델 로드, 동적 프롬프트 생성, 레시피 추천 기능을 캡슐화합니다.

RecipeRecommender 클래스 코드 스니펫:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class RecipeRecommender:
    def __init__(self, model_name="microsoft/phi-1_5", device=None):
        # 모델 로드 및 디바이스 설정 (GPU 사용 가능 시 CUDA, 아니면 CPU)
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

    def build_prompt(self, ingredients):
        # 식재료 목록을 기반으로 LLM 프롬프트 생성
        ingredients_str = ", ".join(ingredients)
        prompt = (
                  "You are a recipe recommendation AI. Given the following ingredients, suggest a simple recipe.\n\n"
                  "Ingredients: {ingredients_str}\n\n" # <-- 이 부분은 f-string을 위해 {ingredients_str}로 수정
                  "Format:\n- Dish name:\n- Ingredients:\n- Steps:\n\n"
                  "Answer:"
        )
        return prompt.strip()

    def recommend_recipe(self, ingredients, max_tokens=250):
        # 생성된 프롬프트를 LLM에 전달하여 레시피 추천
        prompt = self.build_prompt(ingredients)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        output = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7, # 생성의 무작위성 조절
            top_p=0.9,       # 상위 확률 분포에서 샘플링
            pad_token_id=self.tokenizer.eos_token_id
        )

        result = self.tokenizer.decode(output[0], skip_special_tokens=True)
        response = result.replace(prompt, "").strip() # 프롬프트 부분 제거
        return response
```
주요 특징 및 역할:
* 모듈화: LLM 관련 로직을 RecipeRecommender 클래스 내부에 캡슐화하여 코드의 재사용성과 유지보수성을 높였습니다.
* 디바이스 최적화: GPU(CUDA) 사용 가능 여부를 자동으로 감지하여 모델을 최적의 하드웨어에 로드하도록 설정했습니다.
* 동적 프롬프트 생성: 사용자가 입력한 식재료에 따라 레시피 요청 프롬프트를 동적으로 구성하여 LLM에 전달합니다.
* 레시피 생성 파라미터 제어: max_new_tokens, temperature, top_p 등의 파라미터를 조절하여 LLM의 레시피 생성 방식(길이, 다양성, 일관성)을 제어할 수 있습니다.

### 4.2. Streamlit 기반 웹 애플리케이션 통합
사용자 친화적인 인터페이스를 제공하기 위해 Streamlit 프레임워크를 활용하여 웹 애플리케이션을 구현했습니다. 이를 통해 사용자는 웹 환경에서 쉽게 식재료 사진을 업로드하고, 실시간으로 레시피를 추천받을 수 있습니다.

Streamlit UI 화면 예시
![Image](https://github.com/user-attachments/assets/d862d194-609d-49db-a5f4-3f98e6b00b9e)
![Image](https://github.com/user-attachments/assets/0b71a472-2e66-43f9-83d3-f68fc85e6711)
![Image](https://github.com/user-attachments/assets/528794f7-eaf6-46cb-9692-32c11ac52e7b)


통합 아키텍처:
* 이미지 입력: 사용자가 Streamlit 웹 UI를 통해 식재료 사진을 업로드합니다.
* YOLOv5 객체 인식: load_yolo_model 함수를 통해 로드된 YOLOv5 모델이 사진 속 식재료를 탐지하고 목록을 추출합니다.
* LLM 모듈 호출: 추출된 식재료 목록은 RecipeRecommender 클래스의 recommend_recipe 메서드에 전달됩니다.
* LLM 레시피 생성: RecipeRecommender 내부에서 Phi-1.5 모델이 레시피를 생성합니다.
* 레시피 출력: 생성된 레시피는 Streamlit 웹 페이지에 사용자에게 보기 좋은 형태로 제공됩니다.

Streamlit 웹 애플리케이션 핵심 코드 스니펫:

```python

import streamlit as st
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. 모델 로드 (YOLO 및 LLM) - @st.cache_resource를 통해 모델 로딩 최적화
@st.cache_resource
def load_yolo_model():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='./best.pt', force_reload=True)
    return model

@st.cache_resource
def load_llm_model():
    model_name = "microsoft/phi-1_5"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

yolo_model = load_yolo_model()
tokenizer, llm_model = load_llm_model()

# 2. Streamlit UI 및 레시피 추천 로직
st.title("🍳 사진 기반 레시피 추천기")
uploaded_file = st.file_uploader("Upload food image:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Detecting Ingredients..."):
        results = yolo_model(image)
        labels = results.names
        detected = results.pred[0][:, -1].cpu().numpy()
        ingredients = list(set([labels[int(i)] for i in detected]))

    if ingredients:
        st.success(f"Detected Ingredients: {', '.join(ingredients)}")

        # LLM에게 전달할 프롬프트 구성 (RecipeRecommender 클래스의 build_prompt 로직과 유사)
        prompt = (
                  f"You are a recipe recommendation AI. Given the following ingredients, suggest a simple recipe.\n\n"
                  f"Ingredients: {', '.join(ingredients)}. \n\n"
                  f"Format:\n- Dish name:\n- Ingredients:\n- Steps:\n\n"
                  f"Answer:"
        )

        with st.spinner("Creating Recipe..."):
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = llm_model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=True,
                top_p=0.95,
                temperature=0.8,
                pad_token_id=tokenizer.eos_token_id
            )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            cleaned_output = generated_text.replace(prompt, "").strip()

        st.markdown("🍳 **Recommended Recipe:**")
        st.markdown(cleaned_output)
    else:
        st.warning("The ingredient was not recognized. Please try another image.")
```
Streamlit 웹 애플리케이션 코드 설명:
* 모델 로딩 최적화: `@st.cache_resource` 데코레이터를 사용하여 YOLO 및 LLM 모델을 한 번만 로드하도록 최적화했습니다. 이는 Streamlit 앱의 성능을 크게 향상시키는 중요한 기법입니다.
* 사용자 인터페이스 구성:`st.title()`, `st.file_uploader()` 등의 Streamlit 컴포넌트를 활용하여 직관적인 사용자 인터페이스를 구성했습니다. 특히 이미지 업로드 기능은 사용자가 식재료 사진을 쉽게 제공할 수 있도록 합니다.
* 실시간 피드백: `st.spinner()`, `st.success()`, `st.warning()` 등을 통해 사용자에게 처리 과정과 결과를 실시간으로 알려주어 사용자 경험을 향상시켰습니다.
* YOLO-LLM 파이프라인 구현:
  1. YOLO 모델로 이미지에서 식재료를 탐지
  2. 탐지된 식재료 목록을 추출
  3. LLM 프롬프트에 동적으로 삽입
  4. LLM이 맞춤형 레시피를 생성하는 전체 파이프라인을 구현
* 중복 제거 로직: `list(set([...]))` 구문을 통해 동일한 식재료가 여러 번 탐지되더라도 중복 없이 한 번만 레시피에 반영되도록 처리했습니다.
* LLM 생성 파라미터 최적화: `temperature=0.8`, `top_p=0.95` 등의 파라미터를 조정하여 레시피 생성의 창의성과 일관성 사이의 균형을 맞추었습니다.
* 결과 후처리:`cleaned_output = generated_text.replace(prompt, "").strip()` 코드를 통해 LLM이 생성한 텍스트에서 프롬프트 부분을 제거하고 실제 레시피만 깔끔하게 추출하여 사용자에게 제공합니다.

### 4.3. LLM 적용을 통한 성과 및 인사이트
동적 레시피 생성: 고정된 데이터베이스의 레시피를 제공하는 대신, 사용자의 입력(식재료 사진)에 따라 LLM이 실시간으로 새로운 레시피를 생성함으로써 추천의 유연성과 다양성을 확보했습니다.
사용자 경험 향상: 직관적인 웹 인터페이스를 통해 딥러닝 모델의 복잡한 기능을 일반 사용자도 쉽게 활용할 수 있도록 구현하여, 접근성과 편의성을 높였습니다.
모듈화된 시스템 설계: RecipeRecommender 클래스와 Streamlit 웹 UI를 분리하여, 각 컴포넌트의 역할과 책임을 명확히 하고 시스템의 확장성 및 유지보수성을 고려한 설계를 적용했습니다.
AI 모델 통합 역량: 객체 탐지 모델(YOLOv5)과 언어 모델(Phi-1.5)이라는 두 가지 이종의 AI 모델을 하나의 애플리케이션으로 성공적으로 통합하여, 복합적인 AI 시스템을 설계하고 구현하는 역량을 보여주었습니다.
도메인 특화 프롬프트 엔지니어링: 'Dish name', 'Ingredients', 'Steps'와 같은 특정 형식을 지정하는 프롬프트 엔지니어링을 통해 LLM이 원하는 형식의 레시피를 생성하도록 유도했습니다.

## 5. 예상되는 기대효과 및 한계
본 프로젝트는 식재료 인식 기반 레시피 추천 시스템을 통해 사용자에게 실질적인 가치를 제공하고, 향후 다양한 방향으로의 확장 가능성을 가지고 있습니다. 동시에, 현재 구현 단계에서의 한계점들을 명확히 인식하고 있습니다.

### 5.1. 기대 효과 및 확장성
* 음식물 낭비 감소 및 식비 절약: 사용자가 보유한 식재료를 효율적으로 활용하도록 유도하여 불필요한 음식물 쓰레기를 줄이고, 식비 절약에 기여할 수 있습니다.
* 레시피 앱 연동 및 쇼핑몰 장바구니 자동 생성: 기존 레시피 애플리케이션과의 연동을 통해 사용자 편의성을 높이고, 인식된 식재료 외 부족한 재료를 쇼핑몰 장바구니에 자동으로 담아주는 기능으로 확장하여 구매 경험을 개선할 수 있습니다.
* IoT 연계 실시간 추천 시스템: 냉장고 내부에 카메라를 설치하여 실시간으로 식재료를 인식하고, 이를 기반으로 레시피를 추천하는 IoT(사물 인터넷) 연계 서비스로 발전시킬 수 있습니다.
* 심화된 레시피 추천: 음식의 양을 인식하여 만들 수 있는 요리의 양을 예측하는 등, 더욱 정교하고 개인화된 레시피 추천 기능으로 심화 발전 가능성이 있습니다.

### 5.2. 한계점
* 데이터셋의 지역적 한계: 라벨링 시간의 제약으로 인해 한국에서 주로 소비되는 식재료 사진을 충분히 확보하지 못했습니다. 이는 모델의 한국형 식재료 인식 정확도에 영향을 미칠 수 있습니다.
* 모델 학습 및 테스트 시간: YOLO 모델의 학습 및 추론에 상대적으로 많은 시간이 소요되어, 다양한 하이퍼파라미터 튜닝이나 모델 테스트를 충분히 진행하지 못했습니다.
* 실시간 영상 인식 성능: 현재 모델은 이미지 기반 인식에 최적화되어 있어, 실시간 영상 데이터에서의 객체 인식률은 다소 떨어지는 한계점을 보입니다. 이는 향후 모델 최적화 및 경량화 연구를 통해 개선될 필요가 있습니다.
