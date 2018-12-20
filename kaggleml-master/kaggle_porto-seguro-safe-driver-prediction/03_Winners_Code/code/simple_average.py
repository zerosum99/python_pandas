import pandas as pd
# 각 모델의 결과물을 읽어온다
keras5_test = pd.read_csv("../model/keras5_pred.csv")
lgbm3_test = pd.read_csv("../model/lgbm3_pred_avg.csv")

def get_rank(x):
    return pd.Series(x).rank(pct=True).values

# 두 예측값의 단순 평균을 최종 앙상블 결과물로 저장한다
pd.DataFrame({'id': keras5_test['id'], 'target': get_rank(keras5_test['target']) * 0.5 + get_rank(keras5_test['target']) * 0.5}).to_csv("../model/simple_average.csv", index = False)
