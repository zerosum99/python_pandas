"""
train with semi data
"""
from resnet import ResModel
from senet import SeModel
from vgg2d import vgg2d
from vgg1d import vgg1d
from trainer import train_model
from data import preprocess_mel, preprocess_mfcc, preprocess_wav

list_2d = [('mel', preprocess_mel), ('mfcc', preprocess_mfcc)]
BAGGING_NUM=3

# 모델 학습을 실행하는 함수이다.
def train_and_predict(cfg_dict, preprocess_list):
    for p, preprocess_fun in preprocess_list:
        cfg = cfg_dict.copy()
        cfg['preprocess_fun'] = preprocess_fun
        cfg['CODER'] += '_%s' %p
        cfg['bagging_num'] = BAGGING_NUM
        # 테스트 데이터에 대한 예측값 경로를 지정한다
        cfg['semi_train_path'] = "sub/base_average.csv"
        print("training ", cfg['CODER'])
        train_model(**cfg)

# Semi-Supervised 학습을 위한 설정값이다.
res_config = {
    'model_class': ResModel,
    'is_1d': False,
    'reshape_size': None,
    'BATCH_SIZE': 32,
    # 모델 학습 epoch을 125로 늘렸다.
    'epochs': 125,
    'CODER': 'resnet_semi'
}

print("train resnet.........")
train_and_predict(res_config, list_2d)

se_config = {
    'model_class': SeModel,
    'is_1d': False,
    'reshape_size': 128,
    'BATCH_SIZE': 16,
    'epochs': 125,
    'CODER': 'senet_semi'
}

print("train senet..........")
train_and_predict(se_config, list_2d)

vgg2d_config = {
    'model_class': vgg2d,
    'is_1d': False,
    'reshape_size': 128,
    'BATCH_SIZE': 32,
    'epochs': 125,
    'CODER': 'vgg2d_semi'
}

print("train vgg2d...........")
train_and_predict(vgg2d_config, list_2d)

vgg1d_config = {
    'model_class': vgg1d,
    'is_1d': True,
    'reshape_size': None,
    'BATCH_SIZE': 32,
    'epochs': 125,
    'CODER': 'vgg1d_semi'
}

print("train vgg1d on raw features..........")
train_and_predict(vgg1d_config, [('raw', preprocess_wav)])



