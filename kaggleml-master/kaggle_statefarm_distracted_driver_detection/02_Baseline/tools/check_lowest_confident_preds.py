import pandas as pd
import numpy as np
import sys

fl = sys.argv[1]
pred = pd.read_csv(fl)
labels = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']
pred['max'] = pred[labels].apply(max, axis=1)
pred['label'] = pred[labels].apply(np.argmax, axis=1)
pred = pred.sort_values(by=['max'])

print(pred.head())
lowest_preds = []
for i, row in pred.iterrows():
    lowest_preds.append((row['img'], row['max'], row['label']))

import pickle
pickle.dump(lowest_preds, open('{}.lowest_preds.pkl'.format(fl), 'wb'))
