import pandas as pd
import numpy as np
import os

# 표 5-9의 앙상블 결과를 얻은 csv 파일을 지정한다 (파일 이름은 독자마다 다를 수 있음)
test_pred_fname = 'FIX_ME'
test_pred = pd.read_csv(test_pred_fname)
test_pred_probs = test_pred.iloc[:, :-1]
test_pred_probs_max = np.max(test_pred_probs.values, axis=1)

# 확률값 구간별로 몇개의 파일이 존재하는지 출력한다
for thr in range(1,10):
  thr = thr / 10.
  count = sum(test_pred_probs_max > thr)
  print('# Thre : {} | count : {} ({}%)'.format(thr, count, 1. * count / len(test_pred_probs_max)))

# 확률값 기준치을 0.90으로 지정한다
print('=' * 50)
threshold = 0.90
count = {}
print('# Extracting data with threshold : {}'.format(threshold))

# 기존의 훈련 데이터를 semi_train_{} 디렉토리로 복사한다
cmd = 'cp -r input/train input/semi_train_{}'.format(os.path.basename(test_pred_fname))
os.system(cmd)

# 기존의 훈련 데이터를 semi_train_{} 디렉토리로 복사한다
for i, row in test_pred.iterrows():
  img = row['img']
  row = row.iloc[:-1]
  if np.max(row) > threshold:
    label = row.values.argmax()
    cmd = 'cp input/test/imgs/{} input/semi_train_{}/c{}/{}'.format(img, os.path.basename(test_pred_fname), label, img)
    os.system(cmd)
    count[label] = count.get(label, 0) + 1

# 클래스별 추가된 테스트 데이터의 통계를 출력한다
print('# Added semi-supservised labels: \n{}'.format(count))
