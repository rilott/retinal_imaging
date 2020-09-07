
import argparse

from data import load_data

from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as preprocess_resnet

parser = argparse.ArgumentParser()
parser.add_argument('path')
args = parser.parse_args()

x, y, s = load_data(args.path, channels=3, rescale=False, preprocess=None)
print(x.shape)
print()

print('No scaling, no preprocess')
print(x[0, 100, 100, :])
print()

x, y, s = load_data(args.path, channels=3, rescale=True, preprocess=None)

print('Scaling, no preprocess')
print(x[0, 100, 100, :])
print()

x, y, s = load_data(args.path, channels=3, rescale=False, preprocess=preprocess_input)

print('No scaling, preprocess inceptionv3')
print(x[0, 100, 100, :])
print()

x, y, s = load_data(args.path, channels=3, rescale=False, preprocess=preprocess_resnet)

print('No scaling, preprocess inception_resnetv2')
print(x[0, 100, 100, :])
print()
