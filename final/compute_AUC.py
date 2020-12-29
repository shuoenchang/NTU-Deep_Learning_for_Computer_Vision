import pandas as pd
from sklearn import metrics
import numpy as np
import sys

argv = sys.argv
df = pd.read_csv(argv[1])
y = df['labels'].to_numpy()
pred = df['preds'].to_numpy()


fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)

print(metrics.auc(fpr, tpr))