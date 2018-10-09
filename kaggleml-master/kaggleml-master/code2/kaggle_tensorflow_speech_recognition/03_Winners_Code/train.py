# ResNet 모델이 정의된 모델 함수를 읽어온다
from resnet import ResModel
# 모델 학습용 함수 trainer를 읽어온다
from trainer import train_model
# 데이터 전처리용 함수를 읽어온다
from data import preprocess_mel, preprocess_mfcc

# ResNet 모델에는 mel, mfcc 로 전처리된 입력값을 받는 모델을 각각 학습한다 
list_2d = [('mel', preprocess_mel), ('mfcc', preprocess_mfcc)]
# 동일한 모델을 4개 학습하여, 4모델의 결과물의 평균값을 최종 결과물로 사용한다 (bagging 앙상블)
BAGGING_NUM=4

# 모델을 학습하고, 최종 모델을 기반으로 테스트 데이터에 대한 예측 결과물을 저장하는 도구 함수이다
def train_and_predict(cfg_dict, preprocess_list):
    # 전처리 방식에 따라 각각 다른 모델을 학습한다
    for p, preprocess_fun in preprocess_list:
        # 모델 학습의 설정값 (config)를 정의한다
        cfg = cfg_dict.copy()
        cfg['preprocess_fun'] = preprocess_fun
        cfg['CODER'] += '_%s' %p
        cfg['bagging_num'] = BAGGING_NUM
        print("training ", cfg['CODER'])
        # 모델을 학습한다!
        train_model(**cfg)

# ResNet 모델 학습 설정값이다
res_config = {
    'model_class': ResModel,
    'is_1d': False,
    'reshape_size': None,
    'BATCH_SIZE': 32,
    'epochs': 100,
    'CODER': 'resnet'
}

print("train resnet.........")
train_and_predict(res_config, list_2d)

se_config = {
    'model_class': SeModel,
    'is_1d': False,
    'reshape_size': 128,
    'BATCH_SIZE': 16,
    'epochs': 100,
    'CODER': 'senet'
}

print("train senet..........")
train_and_predict(se_config, list_2d)

dense_config = {
    'model_class': densenet121,
    'is_1d': False,
    'reshape_size': 128,
    'BATCH_SIZE': 16,
    'epochs': 100,
    'CODER': 'dense'
}

print("train densenet.........")
train_and_predict(dense_config, list_2d)

vgg2d_config = {
    'model_class': vgg2d,
    'is_1d': False,
    'reshape_size': 128,
    'BATCH_SIZE': 32,
    'epochs': 100,
    'CODER': 'vgg2d'
}

print("train vgg2d...........")
train_and_predict(vgg2d_config, list_2d)

vgg1d_config = {
    'model_class': vgg1d,
    'is_1d': True,
    'reshape_size': None,
    'BATCH_SIZE': 32,
    'epochs': 100,
    'CODER': 'vgg1d'
}

print("train vgg1d on raw features..........")
train_and_predict(vgg1d_config, [('raw', preprocess_wav)])

vggmel_config = {
    'model_class': vggmel,
    'is_1d': True,
    'reshape_size': None,
    'BATCH_SIZE': 64,
    'epochs': 100,
    'CODER': 'vgg1d'
}

print("train vgg1d on mel features..........")
train_and_predict(vggmel_config, [('mel', preprocess_mel)])


