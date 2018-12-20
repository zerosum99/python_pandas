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

# input/ 폴더에 경진대회 데이터를 넣은 후,
cd code

# Baseline 모델 실행
python baseline.py
```

# 승자의 코드

```
cd ../03_Winners_Code

# input/ 폴더에 경진대회 데이터를 넣은 후,
cd code

# 승자의 코드 재현
python clean.py
python main.py
```
