# 데이터 탐색적 분석

```
# (Optional) 가상환경 설치하기
pip install virtualenv
# 가상환경 생성하기
virtualenv venv
# 가상환경 활성화하기
. venv/bin/activate

# 필요한 라이브러리 설치하기
pip install -r requirements.txt

# jupyter notebook 백그라운드에서 실행하기
nohup jupyter notebook &

# 웹브라우저를 통해 http://localhost:8000로 접속한 후,
# 01_EDA/EDA.ipynb 파일을 실행한다.
```

# Baseline 모델

```
cd 02_Baseline

# input/ 폴더에 경진대회 데이터를 넣는다
cd code

## 01. Baseline 모델
python main.py

# 개선 실험 재현
## 02. 운전자별 교차 검증
python main.py --weights None --random-split 0 --data-augment 0 --learning-rate 1e-4

## 03. ImageNet 기학습 모델
python main.py --weights imagenet --random-split 0 --data-augment 0 --learning-rate 1e-5

## 04. 실시간 데이터 어그멘테이션
python main.py --weights imagenet --random-split 0 --data-augment 1 --learning-rate 1e-4

## 05. 랜덤 교차 검증
python main.py --weights imagenet --random-split 1 --data-augment 1 --learning-rate 1e-4

## 06. 다양한 CNN 모델 학습 (ResNet50)
python main.py --weights imagenet --random-split 0 --data-augment 1 --learning-rate 1e-4 --model resnet50

## 07. 앙상블
# 앙상블을 수행할 파일을 rsc/ensemble 폴더에 이동한다
cp ../subm/<ResNet50모델 결과 파일 경로>/ens.csv ../rsc/ensemble/resnet50.csv
cp ../subm/<VGG19모델 결과 파일 경로>/ens.csv ../rsc/ensemble/vgg19.csv
cp ../subm/<VGG16모델 결과 파일 경로>/ens.csv ../rsc/ensemble/vgg16.csv
python ../tools/ensemble.py

## 08. Semi-Supervised Learning
# Semi-Supervised Learning용 훈련 데이터 구축 
python ../tools/prepare_data_for_semi_supervised.py
python main.py --weights imagenet --random-split 1 --data-augment 1 --learning-rate 1e-4 --semi-train ../input/<semi-supervised 학습 데이터 경로> --model resnet50
```
