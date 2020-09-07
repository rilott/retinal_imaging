import os
import sys
import argparse

import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn.metrics import roc_curve, auc
from sklearn.datasets import make_classification
from sklearn.preprocessing import label_binarize

from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_inceptionv3
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as preprocess_inception_resnetv2
from tensorflow.keras.models import load_model

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from data import load_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser()
parser.add_argument('image')
parser.add_argument('model')
parser.add_argument('--batch-size', default=32, help='Batch size')
parser.add_argument('--preprocess', choices=['inceptionv3','inception_resnetv2'])
parser.add_argument('--classes', default=['ABCA4', 'USH2A'], help='Classes', nargs='+')
args = parser.parse_args()

path = args.image
print('Loading model')
model = load_model(args.model)
print('Model loaded')

if args.preprocess == 'inceptionv3':
    preprocess = preprocess_inceptionv3
elif args.preprocess == 'inception_resnetv2':
    preprocess = preprocess_inception_resnetv2
else:
    preprocess = None
print('Preprocess set to', preprocess)

print('Loading all images into memory..')
x_test, y_test, _ = load_data(args.image, classes=args.classes, rescale=False, preprocess=preprocess, verbose=True)

print('Predicting labels for images..')
y_pred_keras = model.predict(x_test)
y_score = y_pred_keras
print('Done')

# Class stats
n_classes = len(args.classes)
labels = args.classes


# Plot linewidth.
lw = 2

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
print('Computing ROC..')
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
print('Plotting..')
plt.figure(1)

for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], lw=lw, # color=color
             label='{0}'
             ''.format(labels[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()