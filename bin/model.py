from __future__ import print_function

import argparse
import random
import glob
import os
import logging
import time
import sys
import datetime
import math
import json
from collections import defaultdict
from itertools import product

# This is annoying, but needed to suppress the cpp log messages :(
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from keras import activations
from keras import backend as k
from keras.applications import vgg16 as kvgg16
from keras.callbacks import Callback, ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, MaxPooling2D, Dropout, Flatten, Dense, merge, GlobalAveragePooling2D, GlobalMaxPooling2D, BatchNormalization, Activation, Conv2D
from keras.layers.core import Lambda
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy, binary_crossentropy
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import cv2
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
#matplotlib.use('TkAgg')

#import seaborn as sns
#import pandas as pd
#from heatmappy import Heatmapper
#from vis.utils import utils
#from vis.visualization import visualize_cam
#from vis.visualization import visualize_saliency, overlay
#import cv
#from PIL import Image, ImageDraw

img_rows = 256
img_cols = 256
batch_size = 32
smooth = 1.
lr = 1e-6

timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')

out_json=[{"info": " The ABCA4 protein is required for effective dimerization and processing of vitamin A, a necessary component to the normal functioning and survival of photoreceptors. Therefore accumulation of unprocessed vitamin A dimers is one of the earliest measurable events preceding retinal degeneration (Mihai and Washington 2014) detectable by patterns of hyper autofluorescence in OCT around the fovea. ", "gene": "ABCA4"}, {"info": "USH2A is thought to be essential for maintaining the structural integrity of the membrane hosting sensorial cells such as photoreceptors in the retina and cochlear hair cells in the inner ear. If both copies of the gene are non-functional, this can lead to Usher syndrome, causing both retinal and hearing impairment. However, there are many patients with loss of function in USH2A which have retinal only manifestation. The retinal disease typically resembles Retinitis Pigmentosa due to the presence of 'bone spicules' in the periphery caused by the death of photoreceptors cells, detectable as zones of hypo autofluorescence.", "gene": "USH2A"}]

#cropped_rows = 128
#cropped_cols = 496
#pixel_overlap = 8
#num_train = int(75*((img_rows - cropped_rows)/pixel_overlap * (img_cols - cropped_cols)/pixel_overlap))  # out of 79 at the moment
train_val_split = 0.8


def w_categorical_crossentropy(weights):
    def loss(y_true, y_pred):
        nb_cl = len(weights)
        final_mask = K.zeros_like(y_pred[:, 0])
        y_pred_max = K.max(y_pred, axis=1, keepdims=True)
        y_pred_max_mat = K.equal(y_pred, y_pred_max)
        for c_p, c_t in product(range(nb_cl), range(nb_cl)):
            final_mask += (weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t])
        return K.categorical_crossentropy(y_pred, y_true) * final_mask
    return loss

def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    weights = K.variable(weights)
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    return loss

def trainGenerator(batch_size, data, labels):
  sl_data = data[0:num_train, :]
  sl_mask = mask[0:num_train, :]
  while True:
    # randomize order at the beginning of each epoch
    p = np.random.permutation(len(sl_data))
    sl_data = sl_data[p, ]
    sl_mask = sl_mask[p, ]
    for i in xrange(0, sl_data.shape[0], batch_size):
      batch_x = np.copy(sl_data[i:i + batch_size])
      batch_y = np.copy(sl_mask[i:i + batch_size])
      if batch_x.shape[0] != batch_size: break
      yield (batch_x, batch_y)


def validGenerator(batch_size, data, labels):
  sl_data = data[num_train:, :]
  sl_mask = mask[num_train:, :]
  while True:
    # randomize order at the beginning of each epoch
    p = np.random.permutation(len(sl_data))
    sl_data = sl_data[p, ]
    sl_mask = sl_mask[p, ]
    for i in xrange(0, sl_data.shape[0], batch_size):
      batch_x = np.copy(sl_data[i:i + batch_size])
      batch_y = np.copy(sl_mask[i:i + batch_size])
      if batch_x.shape[0] != batch_size: break
      yield (batch_x, batch_y)


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.valid = []
        self.lastiter = 0
    def on_epoch_end(self, batch, logs={}):
        self.lastiter += 1
        print(batch, logs)
        with open("./runs/%s/train.txt" % timestamp, "a") as fout:
            for metric in ["loss", "acc"]:
                fout.write("train\t%d\t%s\t%.6f\n" % (self.lastiter, metric, logs.get(metric)))
        with open("./runs/%s/valid.txt" % timestamp, "a") as fout:
            for metric in ["val_loss", "val_acc"]:
                fout.write("train\t%d\t%s\t%.6f\n" % (self.lastiter, metric, logs.get(metric)))
    def on_batch_end(self, batch, logs={}):
        self.lastiter += 1
        with open("./runs/%s/train.txt" % timestamp, "a") as fout:
            for metric in ["loss"]:
                fout.write("train\t%d\t%s\t%.6f\n" % (self.lastiter, metric, logs.get(metric)))


def make_parallel(model, gpu_count):
    def get_slice(data, idx, parts):
        shape = tf.shape(data)
        size = tf.concat(0, [shape[:1] // parts, shape[1:]])
        stride = tf.concat(0, [shape[:1] // parts, shape[1:] * 0])
        start = stride * idx
        return tf.slice(data, start, size)
    outputs_all = []
    for i in range(len(model.outputs)):
        outputs_all.append([])
    # Place a copy of the model on each GPU, each getting a slice of the batch
    for i in range(gpu_count):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i) as scope:
                inputs = []
                # Slice each input into a piece for processing on this GPU
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_n = Lambda(get_slice, output_shape=input_shape, arguments={'idx': i, 'parts': gpu_count})(x)
                    inputs.append(slice_n)
                outputs = model(inputs)
                if not isinstance(outputs, list):
                    outputs = [outputs]
                # Save all the outputs for merging back together later
                for l in range(len(outputs)):
                    outputs_all[l].append(outputs[l])
    # merge outputs on CPU
    if gpu_count == 1:
        return Model(input=model.inputs, output=outputs)
    with tf.device('/cpu:0'):
        merged = []
        for outputs in outputs_all:
            merged.append(merge(outputs, mode='concat', concat_axis=0))
        return Model(input=model.inputs, output=merged)

def NASNetLarge(class_weights, input_size, lr):
    num_class = len(class_weights)
    base_model = keras.applications.nasnet.NASNetLarge(include_top=False, input_shape=[input_size,input_size,3], weights='imagenet', pooling='max', classes=num_class)
    print(base_model.layers[-1])
    x = base_model.layers[-1].output
    x = Dense(num_class, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=x)
    return model


def vgg16(class_weights, input_size, feature_layer, lr, with_dropout=False):
    num_class = len(class_weights)
    base_model = kvgg16.VGG16(include_top=False, weights='imagenet', input_shape=[input_size,input_size,3], classes=num_class)

    if with_dropout:
        # Sort out the output layer
        #y = base_model.get_layer(feature_layer).output
        #y = GlobalAveragePooling2D()(y)
        #y = Dense(num_class, activation='softmax')(y)

        # Store the fully connected layers
        fc1 = base_model.layers[-3]
        fc2 = base_model.layers[-2]
        #predictions = base_model.get_layer(feature_layer).output
        #predictions = GlobalAveragePooling2D()(predictions)
        #predictions = Dense(num_class, activation='softmax')(predictions)

        output_layer = base_model.get_layer(feature_layer).output

        # Create the dropout layers
        dropout1 = Dropout(0.85)
        dropout2 = Dropout(0.85)

        # Reconnect the layers
        x = dropout1(fc1.output)
        x = fc2(x)
        x = dropout2(x)
        x = GlobalMaxPooling2D()(x)
        x = Dense(num_class, activation='softmax')(x)




        #predictors = predictions(x)
        #predictors = Dense(num_class, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=x)
    else:
        x = base_model.get_layer(feature_layer).output
        x = GlobalAveragePooling2D()(x)
        x = Dense(num_class, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=x)

    optimizer = Adam(lr=lr)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def vgg_simple_yuka(include_top=True, input_tensor=None, input_shape=None, pooling=None, classes=2, weights_file=None):
        #width,height,depth,=input_shape
        #initialize the model
        model = Sequential()
        inputShape = (img_rows, img_cols, 1)
        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first": inputShape = (depth, height, width)
        # use VGG16 up to the 3rd pooling layer
        # first set of CONV -> RELU -> Pool layers
        model.add(Conv2D(64, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        #model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        #model.add(Dropout(0.25))
        # second set of CONV => RELU => POOL layers
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        #model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        #model.add(Dropout(0.5))
        # third set of CONV => RELU => POOL layers
        model.add(Conv2D(256, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(Conv2D(256, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(Conv2D(256, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        #model.add(Dropout(0.5))
        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(1000))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))
        model.add(Dense(1000))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))
        if classes == 1: #regression
            model.add(Dense(1))
            model.add(Activation("linear"))
        else:
            # softmax classifier
            model.add(Dense(classes))
            model.add(Activation("softmax"))
        # return the constructed network architecture
        if weights_file: model.load_weights(weights_file)
        return model


def vgg_simple_yue(include_top=True, input_tensor=None, input_shape=None, pooling=None, classes=2, weights_file=None):
    inputs = Input((img_rows, img_cols, 1))
    # Block 1
    #x = BatchNormalization()(inputs)
    x = Convolution2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
    #x = BatchNormalization()(x)
    x = Convolution2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    #x= Dropout(0.5)(x)
    # Block 2
    x = Convolution2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    #x = BatchNormalization()(x)
    x = Convolution2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    # Block 3
    x = Convolution2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    #x = BatchNormalization()(x)
    x = Convolution2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    #x = BatchNormalization()(x)
    x = Convolution2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    # Classification block
    if include_top:
        x = Flatten(name='flatten')(x)
        x = Dense(1024, activation='relu', name='fc1')(x)
        x = Dropout(0.5)(x)
        x = Dense(1024, activation='relu', name='fc2')(x)
        x = Dense(classes)(x)
        x = Activation('softmax')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)
    # Create model.
    model = Model(inputs, x, name='vgg_simple_yue')
    if weights_file: model.load_weights(weights_file)
    return model


def _create_weights_folder():
    if not os.path.exists("./runs"):
        os.mkdir('runs')
    os.mkdir("./runs/%s/" % timestamp)
    os.mkdir("./runs/%s/weights" % timestamp)

def fixaxes(x):
    x=np.swapaxes(x,0,3)
    x=np.swapaxes(x,0,1)
    return x

def onehot(labels):
    b=np.zeros((len(labels),2))
    b[np.arange(len(labels)),labels]=1
    return b


def png2numpy(f,verbose=True,img_rows=256,img_cols=256):
    if verbose: print('input file',f)
    img=cv2.imread(f,cv2.IMREAD_GRAYSCALE)
    img=cv2.resize(img,(img_rows,img_cols)).astype(np.float32)
    img=np.array([img])
    img=img.astype('float32')
    return img

def load_image(input_file,verbose=False):
    imgs=[png2numpy(input_file,verbose=verbose)]
    imgs=fixaxes(np.stack(imgs,axis=1).astype('float32'))
    return imgs

def load_images(directory,verbose=False):
    imgs=[]
    labels=[]
    imgs_names=[]
    for i, label in enumerate(sorted( os.listdir(directory))):
        pngs=glob.glob(directory+'/'+label+'/*.png')
        imgs_names+=pngs
        if verbose: print(i,label,len(pngs))
        imgs+=[png2numpy(f,verbose=verbose) for f in pngs]
        labels+=[i]*len(pngs)
    imgs=fixaxes(np.stack(imgs,axis=1).astype('float32'))
    labels=onehot(labels)
    i=np.random.permutation(len(imgs_names))
    imgs=imgs[i,]
    labels=labels[i,]
    imgs_names=np.array(imgs_names)[i].tolist()
    return imgs, labels, imgs_names

def train_and_predict(train_dir,valid_dir, output_dir, normalise=True,augment=True,epochs=100,class_weights=[1,1],lr=1e-5,batch_size=32):
    train_labels=sorted(os.listdir(train_dir))
    valid_labels=sorted(os.listdir(valid_dir))
    assert train_labels==valid_labels
    print('Labels:', train_labels)
    labels=train_labels
    #train dataset
    train_imgs, train_labels,_,=load_images(train_dir)
    #valid dataset
    valid_imgs, valid_labels,_,=load_images(valid_dir)
    print('Train imgs shape:', train_imgs.shape)
    print('Train labels shape:', train_labels.shape)
    print('Number per class', np.sum(train_labels,axis=0))
    print('Valid imgs shape:', valid_imgs.shape)
    print('Valid labels shape:', valid_labels.shape)
    print('Number per class:', np.sum(valid_labels,axis=0))
    mean=None
    std=None
    #normalise
    if normalise:
        # mean for data centering
        mean = np.mean(np.concatenate((train_imgs,valid_imgs),axis=0))
        # std for data normalization
        std = np.std(np.concatenate((train_imgs,valid_imgs),axis=0))
        train_imgs -= mean
        train_imgs /= std
        valid_imgs -= mean
        valid_imgs /= std
    _create_weights_folder()
    #model = simple_vgg()
    #model=vgg_simple_yuka()
    model=vgg16(class_weights, 256, "block5_conv3", lr, with_dropout=True)
    #model=NASNetLarge(class_weights, 256, lr)
    print(train_imgs.shape)
    train_imgs = np.repeat(train_imgs, 3, axis=3)
    valid_imgs = np.repeat(valid_imgs, 3, axis=3)

    print(model.summary())
    model.compile(optimizer=Adam(lr=lr), loss=categorical_crossentropy, metrics=['accuracy'])
    #model.compile(optimizer=Adam(lr=lr,decay=lr/epochs), loss=binary_crossentropy, metrics=['accuracy'])
    #model.compile(optimizer=Adam(lr=lr), loss=weighted_categorical_crossentropy(np.array(class_weights)), metrics=['accuracy'])
    print('-' * 30)
    print('Loading and preprocessing train data...')
    print('-' * 30)
    #train_ush2a_files=glob.glob(train_ush2a_folder+'/*.png')
    # with open("/data/pepple/runs/%s/params.txt" % timestamp, "w") as fout:
    with open("./runs/%s/params.txt" % timestamp, "w") as fout:
        fout.write("mean\t%.9f\n" % mean)
        fout.write("std\t%.9f\n" % std)
        fout.write("lr\t%.9f\n" % lr)
    #filepath = "./runs/%s/weights/weights-improvement-{epoch:03d}-{val_loss:.8f}.hdf5" % timestamp
    filepath = "./runs/%s/weights/weights.hdf5" % timestamp
    checkpoint = ModelCheckpoint(filepath)
    with open("./runs/%s/model.json" % timestamp ,'w') as model_out:
         model_out.write(model.to_json())
    history = LossHistory()
    print('-' * 30)
    print('Fitting model...')
    print('-' * 30)
    if augment:
        datagen = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=5,
            width_shift_range=0.05,
            height_shift_range=0.05,
                    zoom_range=[0.1,3],
                    rescale=0.5,
            zca_whitening=False,
            horizontal_flip=True,
                    vertical_flip=True)
        #datagen = ImageDataGenerator()
        datagen.fit(train_imgs)
        #model.fit_generator(trainGenerator(batch_size=batch_size, data=train_imgs, labels=train_labels), nb_worker=1, validation_data=validGenerator(batch_size=batch_size, data=imgs_train, mask=imgs_mask_train), samples_per_epoch=3000, nb_epoch=500, verbose=1, nb_val_samples=2500, callbacks=[history, checkpoint])  # 3600384
        #model.fit_generator(datagen.flow(train_imgs,train_labels,batch_size=batch_size), validation_data=(valid_imgs,valid_labels), callbacks=[history, checkpoint], epochs=epochs, shuffle=True, use_multiprocessing=True,class_weight=[1,2])
        #lr_finder = LRFinder(min_lr=1e-5, max_lr=1)
        tmp_name = generate_model_name('vgg16', epochs, batch_size, lr)
        checkpoint = ModelCheckpoint(os.path.join(output_dir, tmp_name+'-best.hd5'), monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
        early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=epochs/3, verbose=1, mode='auto')

        hist = model.fit_generator(datagen.flow(train_imgs,train_labels,batch_size=batch_size), validation_data=(valid_imgs,valid_labels), callbacks=[ checkpoint, early], epochs=epochs, shuffle=True, verbose=1)
        print('Evaluating')
        model.evaluate(valid_imgs, y = valid_labels, batch_size=batch_size, verbose=1)
        return model, hist
    else:
        #model.fit(train_imgs, train_labels, epochs=100, validation_data=(valid_imgs, valid_labels), verbose=1, callbacks=[history, checkpoint], class_weights=(0.6,0.4))
        hist = model.fit(train_imgs, train_labels, epochs=100, validation_data=(valid_imgs, valid_labels), verbose=1, callbacks=[history, checkpoint])
        return model, hist

def predict(model,predict_folder,lr,batch_size):
    predict_images,actual_labels,predict_images_names,=load_images(predict_folder,verbose=True)
    predictions=model.predict(predict_images, batch_size=None, verbose=1)
    print(predictions.shape)
    print(predictions)
    print(predictions.argmax(axis=1).shape)
    labels=[label for label in (sorted( os.listdir(predict_folder)))]
    print(labels)
    errors=np.sum(abs(predictions-actual_labels),axis=1).tolist()
    errors=pd.DataFrame({   'file':predict_images_names,
                'errors':errors,
                'predicted_label':np.array(labels)[np.array(predictions.argmax(axis=1))].tolist(),
                'actual_label':np.array(labels)[np.array(actual_labels.argmax(axis=1))].tolist()})
    errors=errors.sort_values('errors',ascending=False)
    print(errors.to_csv(index=False))

def predict_json(model,image_file,json_output,lr,batch_size):
    image=load_image(image_file,verbose=True)
    predictions=model.predict(image, batch_size=None, verbose=1)
    print(json_output)
    print(predictions.shape)
    print(predictions)
    for i,x in enumerate(out_json):
        print(x)
        x['probability']=str(predictions[0][i])
        print(x)
    print(predictions.argmax(axis=1).shape)
    out_json_sorted=sorted(out_json, key=lambda x: float(x['probability']), reverse=True)
    with open(json_output, 'w') as outfile:
            json.dump(out_json_sorted, outfile)



def iter_occlusion(image, size=8):
    occlusion = np.full((1, size * 5, size * 5), [0.5], np.float32)
    occlusion_center = np.full((1, size, size), [0.5], np.float32)
    occlusion_padding = size * 2
    # print('padding...')
    image_padded = np.pad(image, ( (0,0), (occlusion_padding, occlusion_padding), (occlusion_padding, occlusion_padding) ), 'constant', constant_values = 0.0)
    print('Occlusion center shape:', occlusion_center.shape)
    print('Occlusion shape:', occlusion.shape)
    print('Image shape:', image.shape)
    print('Image shaped padded:',image_padded.shape)
    for x in range(occlusion_padding, image.shape[1]+occlusion_padding, size):
        for y in range(occlusion_padding, image.shape[2]+occlusion_padding, size):
            tmp = image_padded.copy()
            x_s=(y-occlusion_padding)
            x_e=(y+occlusion_center.shape[1]+occlusion_padding)
            y_s=(x-occlusion_padding)
            y_e=(x+occlusion_center.shape[2]+occlusion_padding)
            print('tmp shape:',tmp.shape)
            print('tmp index shape:',tmp[:,x_s:x_e,y_s:y_e])
            print(occlusion.shape)
            tmp[:,x_s:x_e,y_s:y_e]=occlusion
            x_s=x
            x_e=(x+occlusion_center.shape[1])
            y_s=y
            y_e=(y+occlusion_center.shape[2])
            tmp[:,x_s:x_e,y_s:y_e]=occlusion_center
            x_s=x-occlusion_padding
            x_e=tmp.shape[1]-occlusion_padding
            y_s=y-occlusion_padding
            y_e=tmp.shape[2]-occlusion_padding
            yield x_s, y_s, tmp[:,x_s:x_e,y_s:y_e]


def occlusion(f,model,occlusion_size=8):
    #predict_images,predict_labels,predict_images_names,=load_images(predict_folder,verbose=False)
    #predictions=model.predict(predict_images, batch_size=None, verbose=0, steps=None)
    #print(np.concatenate((predictions,predict_labels),axis=1))
    #errors=np.sum(abs(predictions-predict_labels),axis=1).tolist()
    #errors=pd.DataFrame({'file':predict_images_names,'errors':errors})
    #errors=errors.sort_values('errors',ascending=False)
    #print(errors.to_csv(index=False))
    image=png2numpy(f)
    heatmap = np.zeros((img_rows, img_cols), np.float32)
    class_pixels = np.zeros((img_rows, img_cols), np.int16)
    counters = defaultdict(int)
    for n, (x, y, img_float) in enumerate(iter_occlusion(image, size=occlusion_size)):
        X = img_float.reshape(1, img_rows, img_cols, 1)
        out = model.predict(X)
        print('#{}: {} @ {} (correct class: {})'.format(n, np.argmax(out), np.amax(out), out[0][correct_class]))
        #print('x {} - {} | y {} - {}'.format(x, x + occlusion_size, y, y + occlusion_size))
        heatmap[y:y + occlusion_size, x:x + occlusion_size] = out[0][correct_class]
        class_pixels[y:y + occlusion_size, x:x + occlusion_size] = np.argmax(out)
        counters[np.argmax(out)] += 1
    pred = model.predict(inp)
    print('Correct class: {}'.format(correct_class))
    print('Predicted class: {} (prob: {})'.format(np.argmax(pred), np.amax(out)))
    print('Predictions:')
    for class_id, count in counters.items():
        print('{}: {}'.format(class_id, count))

def saliency(image_path, model):
    categoriy_index=0
    nb_classes=2
    target_layer = lambda x: target_category_loss(x, category_index, nb_classes)
    model.add(Lambda(target_layer, output_shape = target_category_loss_output_shape))
    print(model.layers[-1])
    loss = K.sum(model.layers[-1].output)
    print(loss)
    conv_output =  [l for l in model.layers[0].layers if l.name is layer_name][0].output
    print(conv_output)
    grads = normalize(K.gradients(loss, conv_output)[0])
    gradient_function = K.function([model.layers[0].input], [conv_output, grads])
    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]
    weights = np.mean(grads_val, axis = (0, 1))
    cam = np.ones(output.shape[0 : 2], dtype = np.float32)
    for i, w in enumerate(weights):
        cam += w * output[:, :, i]
    cam = cv2.resize(cam, (224, 224))
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)
    #Return to BGR [0..255] from the preprocessed image
    image = image[0, :]
    image -= np.min(image)
    image = np.minimum(image, 255)
    cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(image)
    cam = 255 * cam / np.max(cam)
    return np.uint8(cam), heatmap

def saliency2(image_path,model,filter_indices):
    #img = utils.load_img(image_path, target_size=(256, 256))
    img = png2numpy(image_path)
    #layer_idx = utils.find_layer_idx(model, 'predictions')
    layer_idx=-1
    print(layer_idx)
    # Swap softmax with linear
    print(model.layers[layer_idx].activation)
    model.layers[layer_idx].activation = activations.linear
    print(model.layers[layer_idx].activation)
    model = utils.apply_modifications(model)
    modifier='guided'
    #out_img=os.path.basename(image_path).replace('.png','')+'_'+str(modifier)+'_saliency.png'
    out_img=image_path.replace('.png','')+'_'+str(modifier)+'_saliency.png'
    #out_img=image_path
    plt.figure()
    f, ax = plt.subplots(1, 1)
    #plt.suptitle("vanilla" if modifier is None else modifier)
    # 20 is the imagenet index corresponding to `ouzel`
    #img_inp=png2numpy(image_path)
    img=np.reshape(img,(256,256,1))
    print(img)
    #img=np.delete(img,2,1)
    #print(img.shape)
    #img=img[:,:,]
    print(img.shape)
    cv.fromarray(img)
    #cv2.imwrite('test.png',img)
    img=np.expand_dims(img,axis=0)
    print('img',img.shape)
    #print(img.shape)
    #grads = visualize_saliency(model, layer_idx, filter_indices=0, seed_input=img, backprop_modifier=modifier)
    model.summary()
    grads = visualize_saliency(model, layer_idx, filter_indices=filter_indices, seed_input=img, backprop_modifier=modifier)
    #print('grads',grads.shape)
    print('grads',grads.shape)
    print(grads)
    #print('grads',np.min(grads, axis=2))
    #print('grads',np.mean(grads,axis=2))
    #print('grads',np.max(grads,axis=2))
    #grads=(grads*255./np.max(grads)).astype(np.uint8)
    #Image.fromarray(grads).save(out_img)
    #continue
    # Lets overlay the heatmap onto original image.
    #jet_heatmap = np.uint8(cm.jet(grads)[..., :1] * 255)
    #print(jet_heatmap.shape)
    #img = utils.load_img(image_path, target_size=(256, 256))
    #img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    print(img.shape)
    #jet_heatmap=np.delete(jet_heatmap,3,1)
    #print(jet_heatmap.shape)
    #ax.imshow(overlay(jet_heatmap, img))
    #ax.imshow(grads,cmap='jet')
    ax.imshow(grads,cmap='jet')
    f.savefig(out_img, format='PNG')

def grad_cam(image_path,model,filter_indices):
    #img = utils.load_img(image_path, target_size=(256, 256))
    img = png2numpy(image_path)
    #layer_idx = utils.find_layer_idx(model, 'predictions')
    layer_idx=-1
    print(layer_idx)
    # Swap softmax with linear
    print(model.layers[layer_idx].activation)
    model.layers[layer_idx].activation = activations.linear
    print(model.layers[layer_idx].activation)
    model = utils.apply_modifications(model)
    modifier='guided'
    out_img=os.path.basename(image_path).replace('.png','')+'_'+str(modifier)+'_gradcam.png'
    plt.figure()
    f, ax = plt.subplots(1, 1)
    #plt.suptitle("vanilla" if modifier is None else modifier)
    # 20 is the imagenet index corresponding to `ouzel`
    #img_inp=png2numpy(image_path)
    img=np.reshape(img,(256,256,1))
    print(img)
    #img=np.delete(img,2,1)
    #print(img.shape)
    #img=img[:,:,]
    print(img.shape)
    cv.fromarray(img)
    #cv2.imwrite('test.png',img)
    img=np.expand_dims(img,axis=0)
    print('img',img.shape)
    #print(img.shape)
    #grads = visualize_saliency(model, layer_idx, filter_indices=0, seed_input=img, backprop_modifier=modifier)
    model.summary()
    #grads = visualize_saliency(model, layer_idx, filter_indices=filter_indices, seed_input=img, backprop_modifier=modifier)
    grads = visualize_cam(model, layer_idx, filter_indices=filter_indices, seed_input=img, backprop_modifier=modifier)
    # Lets overlay the heatmap onto original image.
    #jet_heatmap = np.uint8(cm.jet(grads)[..., :3] * 255)
    #print('grads',grads.shape)
    print('grads',grads.shape)
    print(grads)
    #print('grads',np.min(grads, axis=2))
    #print('grads',np.mean(grads,axis=2))
    #print('grads',np.max(grads,axis=2))
    #grads=(grads*255./np.max(grads)).astype(np.uint8)
    #Image.fromarray(grads).save(out_img)
    #continue
    # Lets overlay the heatmap onto original image.
    #jet_heatmap = np.uint8(cm.jet(grads)[..., :1] * 255)
    #print(jet_heatmap.shape)
    #img = utils.load_img(image_path, target_size=(256, 256))
    #img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    #jet_heatmap=np.delete(jet_heatmap,3,1)
    #print(jet_heatmap.shape)
    #ax.imshow(overlay(jet_heatmap, img))
    #ax.imshow(grads,cmap='jet')
    #ax.imshow(grads,cmap='jet')
    srcBGR = cv2.imread(image_path)
    destRGB = cv2.cvtColor(srcBGR, cv2.COLOR_BGR2RGB)
    print(grads.shape)
    print(destRGB.shape)
    ax.imshow(overlay(grads, destRGB))
    f.savefig(out_img, format='PNG')



def occlusion_exp(image_path, occluding_size, occluding_pixel, occluding_stride, model):
    print(image_path)
    img=png2numpy(image_path)
    print(img.shape)
    #out_img=os.path.basename(image_path).replace('.png','')+'_heatmap.png'
    out_img=image_path.replace('.png','')+'_heatmap.png'
    out_img=image_path
    print('out_img:',out_img)
    image=Image.open(image_path)
    image=image.convert('RGB')
    drw=ImageDraw.Draw(image,'RGBA')
    #image = cv2.imread(image_path)
    #im = cv2.resize(image, (img_rows, img_cols)).astype(np.float32)
    #im[:,:,0] -= 103.939
    #im[:,:,1] -= 116.779
    #im[:,:,2] -= 123.68
    #im = im.transpose((2,0,1))
    out = model.predict(fixaxes(np.expand_dims(img, axis=0)))
    out = out[0]
    # Getting the index of the winning class:
    m = max(out)
    index_object = [i for i, j in enumerate(out) if j == m]
    _, height, width,  = img.shape
    output_height = int(math.ceil((height-occluding_size) / occluding_stride + 1))
    output_width = int(math.ceil((width-occluding_size) / occluding_stride + 1))
    heatmap = np.zeros((output_height, output_width))
    prob_zero=[]
    print(output_height, output_width)
    for h in xrange(output_height):
        for w in xrange(output_width):
            # Occluder region:
            h_start = h * occluding_stride
            w_start = w * occluding_stride
            h_end = min(height, h_start + occluding_size)
            w_end = min(width, w_start + occluding_size)
            # Getting the image copy, applying the occluding window and classifying it again:
            input_image = img.copy()
            input_image[:,h_start:h_end, w_start:w_end] =  occluding_pixel
            #im = cv2.resize(input_image, (imgs_rows, img_cols)).astype(np.float32)
            #im = im.transpose((2,0,1))
            #im = np.expand_dims(im, axis=0)
            out = model.predict(fixaxes(np.expand_dims(input_image,axis=0)))
            out = out[0]
            print('scanning position (%s, %s)'%(h,w))
            # It's possible to evaluate the VGG-16 sensitivity to a specific object.
            # To do so, you have to change the variable "index_object" by the index of
            # the class of interest. The VGG-16 output indices can be found here:
            # https://github.com/HoldenCaulfieldRye/caffe/blob/master/data/ilsvrc12/synset_words.txt
            prob = (out[index_object])
            print(prob)
            heatmap[h,w] = prob
            # draw rectangle on image
            drw.rectangle([w_start,h_start,w_end,h_end],(100,0,150,prob*100))
    #heatmapper = Heatmapper()
    #heatmap = heatmapper.heatmap_on_img(prob_zero, I)
    #f = plt.figure()
    #plt.savefig(sns.heatmap(heatmap,xticklabels=False, yticklabels=False))
    #p = np.asarray(I).astype('float')
    #w, h = I.size
    #y, x = np.mgrid[0:h, 0:w]
    #f.add_subplot(1, 2, 1)  # this line outputs images side-by-side
    #overlay=Image.open('heatmap.png')
    #overlay=overlay.convert('RGBA')
    #f.add_subplot(1, 2, 2)
    #plt.imshow(img)
    #plt.show()
    #img=Image.blend(background,overlay,0.5)
    image.save(out_img, 'PNG')

def generate_model_name(name, epochs, batch_size, lr):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    name = 'model-{}-'.format(name) + timestr + '-{}epochs-{}batch_size-{}lr'.format(epochs, batch_size, lr)
    return name

def save_model(directory, model, name, epochs, batch_size, lr):
    # could also use model.save(filename)
    # and then keras.models.load_model(filename)
    # weights are then stored with model in same file
    m_json = model.to_json()
    filename = os.path.join(directory, generate_model_name(name, epochs, batch_size, lr))
    with open(filename+'.json', 'w') as model_file:
        model_file.write(m_json)
    model.save_weights(filename+'.h5')

def plot_history(history):
    plt.subplot(1,2,1)
    plt.plot(history.history["accuracy"])
    plt.plot(history.history['val_accuracy'])
    plt.title("model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Accuracy","Validation Accuracy"])
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Loss","Validation Loss"])
    plt.show()

def evaluate(model, test_data, test_labels, batch_size, verbose=True):
    model.evaluate(test_data, y = test_labels, batch_size=batch_size, verbose=verbose)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('--debug', action='store_true', help='Enable debug')
    parser.add_argument('--epochs', dest='epochs', type=int, help='number of epochs to run', default=100)
    parser.add_argument('--lr', dest='lr', type=float, help='learning rate', default=1e-5)
    parser.add_argument('--batch_size', dest='batch_size', type=int, help='batch size', default=32)
    parser.add_argument('--img_rows', dest='img_rows', type=int, help='img rows', default=256)
    parser.add_argument('--img_cols', dest='img_cols', type=int, help='img cols', default=256)
    parser.add_argument('--smooth', dest='smooth', type=int, help='smooth', default=1.)
    parser.add_argument('--augment', dest='augment', type=bool, help='augmentations', default=True)
    parser.add_argument('--predict', dest='predict', type=str, help='')
    parser.add_argument('--predict_json', dest='predict_json', type=str, help='')
    parser.add_argument('--json_output', dest='json_output', type=str, help='JSON output')
    parser.add_argument('--weights', dest='weights', type=str, help='hdf5 file containing weights')
    parser.add_argument('--train', dest='train', type=str, help='')
    parser.add_argument('--valid', dest='valid', type=str, help='')
    parser.add_argument('--occlusion', dest='occlusion', type=str, help='')
    parser.add_argument('--saliency', dest='saliency', type=str, help='')
    parser.add_argument('--grad_cam', dest='grad_cam', type=str, help='')
    parser.add_argument('--filter_indices', dest='filter_indices', type=int, help='')
    parser.add_argument('--model', dest='model', type=str, help='Existing model file in JSON format')
    parser.add_argument('--output-dir', dest='output_dir', type=str, help='Location to save trained model')
    args = parser.parse_args()

    # Turn down the tf logging, it's really verbose by default
    # (needs to be before tensorflow import)
    if args.debug:
        logging.getLogger('tensorflow').setLevel(logging.DEBUG)
    else:
        logging.getLogger('tensorflow').setLevel(logging.ERROR)

    # Set tf to grow into GPU memory, not pre-allocate
    gpus = tf.config.experimental.list_physical_devices('GPU')

    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


    if args.model and args.weights:
        pass
    if args.train and args.valid:
        m, hist = train_and_predict(args.train, args.valid, args.output_dir, epochs=args.epochs,augment=args.augment, lr=args.lr,batch_size=args.batch_size)
        save_model(args.output_dir, m, 'vgg16', args.epochs, args.batch_size, args.lr)
        m.save(os.path.join(args.output_dir, generate_model_name('vgg16', args.epochs, args.batch_size, args.lr)))
        plot_history(hist)
    elif args.predict_json and args.weights:
        model = vgg_simple_yuka(weights_file=args.weights)
        json_string = model.to_json()
        print(json_string)
        print(json_string)
        print(model.summary())
        predict_json(model=model,image_file=args.predict_json,json_output=args.json_output,lr=args.lr,batch_size=args.batch_size)
        saliency2(args.predict_json, model, 0)
        #saliency2(args.predict_json, model, 1)
        #occlusion_exp(image_path=args.predict_json, occluding_size=64, occluding_pixel=0, occluding_stride=4, model=model)
    elif args.predict and args.weights:
        model = vgg_simple_yuka(weights_file=args.weights)
        json_string = model.to_json()
        print(json_string)
        print(model.summary())
        predict(model=model,predict_folder=args.predict,lr=args.lr,batch_size=args.batch_size)
    elif args.occlusion and args.weights:
        #Occlusion_exp(image_path, occluding_size, occluding_pixel, occluding_stride)
        #occlusion(f=args.occlusion,weights_file=args.weights)
        model=simple_vgg(weights_file=args.weights)
        model.compile(optimizer=Adam(lr=lr), loss=categorical_crossentropy, metrics=['accuracy'])
        occlusion_exp(image_path=args.occlusion, occluding_size=64, occluding_pixel=0, occluding_stride=4, model=model)
    elif args.saliency and args.weights:
        model=simple_vgg(weights_file=args.weights)
        model.compile(optimizer=Adam(lr=lr), loss=categorical_crossentropy, metrics=['accuracy'])
        model.summary()
        saliency2(args.saliency, model, args.filter_indices)
    elif args.grad_cam and args.weights:
        model=simple_vgg(weights_file=args.weights)
        model.compile(optimizer=Adam(lr=lr), loss=categorical_crossentropy, metrics=['accuracy'])
        model.summary()
        grad_cam(args.grad_cam, model, args.filter_indices)




