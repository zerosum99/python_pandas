import torch
import torch.nn.functional as F
from torch.nn import MaxPool2d

# ResNet 모델이 정의된 파이썬 클래스
class ResModel(torch.nn.Module):

    # 모델 구조를 정의하기 위한 준비 작업을 한다
    def __init__(self):
        super(ResModel, self).__init__()

        # 12-class 분류 문제이며, 모델에 사용하는 채널 수를 128로 지정한다
        n_labels = 12
        n_maps = 128

        # 1채널 입력값을 n_maps(128)채널로 출력하는 3x3 Conv 커널을 사전에 정의한다
        self.conv0 = torch.nn.Conv2d(1, n_maps, (3, 3), padding=(1, 1), bias=False)

        # 입력과 출력 채널이 n_maps(128)인 3x3 Conv 커널을 9개를 사전에 정의한다
        self.n_layers = n_layers = 9
        self.convs = torch.nn.ModuleList([torch.nn.Conv2d(n_maps, n_maps, (3, 3), padding=1, dilation=1, bias=False) for _ in range(n_layers)])

        # max-pooling 계층을 사전에 정의한다
        self.pool = MaxPool2d(2, return_indices=True)

        # batch_normalization과 conv 모듈을 사전에 정의한다
        for i, conv in enumerate(self.convs):
            self.add_module("bn{}".format(i + 1), torch.nn.BatchNorm2d(n_maps, affine=False))
            self.add_module("conv{}".format(i + 1), conv)

        # n_maps(128)을 입력으로 받아 n_labels(12)를 출력하는 최종 선형 계층을 사전에 정의한다
        self.output = torch.nn.Linear(n_maps, n_labels)

    # 모델의 결과값을 출력하는 forward() 함수이다
    def forward(self, x):
        # 9계층의 Conv 모듈과 최종 선형 계층 총 10계층 모델이다
        for i in range(self.n_layers + 1):
            # 입력값 x를 conv 모듈에 적용 후, relu activation을 통과시킨다
            y = F.relu(getattr(self, "conv{}".format(i))(x))

            # residual 모듈 생성을 위한 코드이다. i가 짝수일 때, x는 y + old_x의 합으로 residual 연산이 수행된다.
            if i == 0:
                old_x = y
            if i > 0 and i % 2 == 0:
                x = y + old_x
                old_x = x
            else:
                x = y

            # 2번째 계층부터는 batch_normalization을 적용한다
            if i > 0:
                x = getattr(self, "bn{}".format(i))(x)

            # max_pooling은 사용하지 않도록 설정한다
            pooling = False
            if pooling:
                x_pool, pool_indices = self.pool(x)
                x = self.unpool(x_pool, pool_indices, output_size=x.size())

        # view() 함수를 통해 x의 크기를 조정한다
        x = x.view(x.size(0), x.size(1), -1)
        # 2번째 dimension에 대해서 평균값을 구한다 
        x = torch.mean(x, 2)
        # 마지막 선형 계층을 통과한 결과값을 반환한다
        return self.output(x)
