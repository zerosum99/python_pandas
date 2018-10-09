from glob import glob
import pandas as pd
import os

resnet_weight = 0.33
vgg19_weight = 0.33
vgg16_weight = 0.33
resnet50_test = pd.read_csv('rsc/ensemble/resnet50.csv', index_col=-1).values
vgg19_test = pd.read_csv('rsc/ensemble/vgg19.csv', index_col=-1).values
vgg16_test = pd.read_csv('rsc/ensemble/vgg16.csv', index_col=-1).values

ensemble = resnet50_test * resnet_weight + \
        vgg19_test * vgg19_weight + \
        vgg16_test * vgg16_weight

ensemble = pd.DataFrame(ensemble, columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
test_id = [os.path.basename(fl) for fl in glob('input/test/imgs/*.jpg')]
ensemble.loc[:, 'img'] = pd.Series(test_id, index=ensemble.index)
ensemble.to_csv('subm/resnet50.{}_vgg19.{}_vgg16_{}.csv'.format(resnet_weight, vgg19_weight, vgg16_weight), index=False)
