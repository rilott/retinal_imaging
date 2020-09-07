"""
Attempt at using a much smaller module, following tutorial at:
https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
"""
import argparse
import os
import sys
import time

import matplotlib.pyplot as plt
import talos
from talos.utils import lr_normalizer

from collections import Counter
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dropout, Flatten, Dense
from keras.utils import plot_model
#from tensorflow.keras.optimizers import Adam, RMSprop
#from tensorflow.keras.optimizers.schedules import ExponentialDecay
#.from tensorflow.keras.callbacks import LearningRateScheduler
#from keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
#from keras.optimizers import Adam
from tensorflow.keras.optimizers import Adam, RMSprop

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
class LearningRateDecay:
    def plot(self, epochs, title="Learning Rate Schedule"):
        # compute the set of learning rates for each corresponding
        # epoch
        lrs = [self(i) for i in epochs]
        # the learning rate schedule
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(epochs, lrs)
        plt.title(title)
        plt.xlabel("Epoch #")
        plt.ylabel("Learning Rate")

class StepDecay(LearningRateDecay):
    def __init__(self, initAlpha=0.01, factor=0.25, dropEvery=10):
        # store the base initial learning rate, drop factor, and
        # epochs to drop every
        self.initAlpha = initAlpha
        self.factor = factor
        self.dropEvery = dropEvery
    def __call__(self, epoch):
        # compute the learning rate for the current epoch
        exp = np.floor((1 + epoch) / self.dropEvery)
        alpha = self.initAlpha * (self.factor ** exp)
        # return the learning rate
        return float(alpha)

class PolynomialDecay(LearningRateDecay):
    def __init__(self, maxEpochs=100, initAlpha=0.01, power=1.0):
            # store the maximum number of epochs, base learning rate,
            # and power of the polynomial
            self.maxEpochs = maxEpochs
            self.initAlpha = initAlpha
            self.power = power
    def __call__(self, epoch):
            # compute the new learning rate based on polynomial decay
            decay = (1 - (epoch / float(self.maxEpochs))) ** self.power
            alpha = self.initAlpha * decay
            # return the new learning rate
            return float(alpha)

def retinal_model(dummyXtrain, dummyYtrain, dummyXval, dummyYval, params):
    model = Sequential()
    model.add(Conv2D(params['first_neuron'], kernel_size=(3, 3), input_shape=(256, 256, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(params['second_neuron'], (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(params['third_neuron'], (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #model.add(Conv2D(64, (3, 3), activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(params['third_neuron'], activation='relu'))
    model.add(Dropout(params['dropout']))
    model.add(Dense(2, activation='softmax'))

    optimizer = params['optimizer'](learning_rate=params['lr'])

    #model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=params['optimizer'](lr_normalizer(params['lr'], params['optimizer'])))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)

    # Calculate class weightings
    counter = Counter(train_generator.classes)
    max_val = float(max(counter.values()))
    class_weights = {class_id: max_val/num_images for class_id, num_images in counter.items()}
    print('Using class weights:', class_weights)

    # Generate model name
    model_time = time.strftime('%Y%m%d_%H%M%S')
    model_name = 'custom_model_' + model_time

    # Callbacks
    ## Checkpoints
    best_model_name = 'checkpoints/' + model_name + '_best.h5'
    checkpoint = ModelCheckpoint(
        filepath=best_model_name,
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max',
    )

    # LR
    schedule = PolynomialDecay(maxEpochs=params['epochs'], initAlpha=params['lr'], power=params['lr_power']) # POLYNOMIAL
    lr_schedule = LearningRateScheduler(schedule)

    ## Tensorboard
    tensorboard_callback = TensorBoard(log_dir='logs/' + model_name, histogram_freq=1, write_graph=True, write_images=True, profile_batch=0)

    # Fit the model
    out = model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples // params['batch_size'],
        epochs=params['epochs'],
        class_weight=class_weights,
        callbacks=[checkpoint, tensorboard_callback, lr_schedule],
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // params['batch_size'])

    return out, model


def augment_image_test(generator, image):
    """
    Generate augmented data for one image
    """

    # Test on a single image
    img = load_img(image)
    #img = load_img('../../retinalimages/train/ABCA4/739ad8983bb4464803564dbf4a83313c_L_2012-03-06.png')
    img = img_to_array(img)
    img = img.reshape((1,) + img.shape)

    # Extract 20 generated images, saved to ../preview/
    i = 0
    for _ in generator.flow(img, batch_size=1, save_to_dir='../preview', save_prefix='ABCA4', save_format='jpeg'):
        i += 1
        if i > 20:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=5, type=int, help='Epochs to run for')
    parser.add_argument('--batchsize', default=32, type=int, help='Batch size')
    parser.add_argument('--preview', action='store_true', help='Generate a batch of augmented images then quit')
    parser.add_argument('--image', help='Image to generate augmented images from')
    args = parser.parse_args()

    # Data generator
    datagen = ImageDataGenerator(
        rotation_range=90,
        #width_shift_range=0.2,
        #height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.5,
        horizontal_flip=True,
        vertical_flip=True)

    if args.preview:
        if not args.image:
            print('--image required when --preview set')
            sys.exit(1)
        augment_image_test(datagen, args.image)
        sys.exit(0)

    # Create a small conv model
    #print(model.summary())

    ## LR decay
    #initial_lr = 0.0001
    #lr_schedule = ExponentialDecay(
    #    initial_lr,
    #    decay_steps=100000,
    #    decay_rate=0.96,
    #    staircase=True
    #)

    # Compile it, using rmsprop (?)
    #optimiser = RMSprop(learning_rate=1e-4)
    #model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimiser)
    #optimiser = RMSprop(learning_rate=lr_schedule)
    #optimiser = Adam(learning_rate=1e-3)
    #model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimiser)

    batch_size = args.batchsize

    # Train data generator, same as before
    train_datagen = datagen

    # Test data generator, just rescales the pixel values to 0-1
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Train dir
    train_generator = train_datagen.flow_from_directory(
        '../../retinalimages/train',
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode='categorical')
    # Test dir
    validation_generator = test_datagen.flow_from_directory(
        '../../retinalimages/validation',
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode='categorical')

    ## LR rate
    #schedule = StepDecay(initAlpha=1e-1, factor=0.25, dropEvery=15) # STEP
    #schedule = PolynomialDecay(maxEpochs=args.epochs, initAlpha=1e-4, power=1) # LINEAR
#    schedule = PolynomialDecay(maxEpochs=args.epochs, initAlpha=1e-3, power=3) # POLYNOMIAL
    #lr_callback = LearningRateScheduler(schedule)


    params = {
        'lr': [0.1, 0.001, 0.0001, 0.00001, 0.000001],
        'lr_power': [1,3,5,9,15],
        'first_neuron': [16, 32, 64, 128],
        'second_neuron': [16, 32, 64, 128],
        'third_neuron': [8, 16, 32, 64],
        'batch_size': [16, 32, 64, 128],
        'epochs': [20],
        'dropout': (0, 0.5, 0.7, 0.9),
        'optimizer': [Adam, RMSprop],
        'weight_regulizer': [None],
    }

    # Talos
    scan_obj = talos.Scan(
        next(train_generator),
        next(validation_generator),
        x_val=next(train_generator),
        y_val=next(validation_generator),
        params=params,
        model=retinal_model,
        experiment_name='eye2gene',
        print_params=True,
    )

    scan_obj.data.head()
    scan_obj.learning_entropy
    scan_obj.details

    ana_obj = talos.Analyze(scan_obj)
    print(ana_obj.best_params('val_acc', ['acc', 'loss', 'val_loss']))

    # Save model
    #model.load_weights(best_model_name)
    #model.save('trained_models/' + model_name + '.h5')

    """ # Generate some performance graphs of loss and accuracy
    figure, plots = plt.subplots(2, 1, figsize=(5, 10))
    ax1 = plots[0]
    ax2 = plots[1]
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.title.set_text('Model accuracy')
    ax1.set_ylabel('accuracy')
    ax1.set_xlabel('epoch')
    ax1.legend(['train', 'val'], loc='upper left')

    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.title.set_text('Model Loss')
    ax2.set_ylabel('loss')
    ax2.set_xlabel('epochs')
    ax2.legend(['train', 'val'], loc='upper right')

# Save figure
    figure.savefig('trained_models/' + model_name + '.png')
    print('Output graph saved to trained_models/' + model_name + '.png')

# Visualisation fun
    plot_model(
        model,
        to_file="model.png",
        show_shapes=True,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=False,
        dpi=96,
    )

    schedule.plot(np.arange(0, args.epochs))
    plt.savefig('trained_models/' + model_name + '_learn.png')
    """

