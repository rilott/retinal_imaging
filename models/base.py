'''
Model baseclass

'''

# pylint: disable=no-member

import json
import os
import time
import random
import sys
from collections import Counter

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.summary import create_file_writer

from data import load_data
from models.learning_rates import PolynomialDecay

class Model:
    """ Base class for a Model """

    def __init__(self, model_config):
        for k, v in model_config.items():
            setattr(self, k, v)

        self.name = 'base'
        self._config = model_config # Save raw config

        self.callbacks = list()

        # Datagen attrs
        self.train_datagen = None
        self.validation_datagen = None
        self.train_generator = None
        self.validation_generator = None

        # Set time to obj creation time
        self.train_start = time.strftime('%d%m%Y-%H%M%S')

        if not os.path.exists(self.model_save_dir):
            raise Exception('Save location {} does not exist'.format(self.model_save_dir))

    def setup(self):
        """ Set up functions now that child class is declared """
        self.load_data()
        self.set_layers()
        self.set_callbacks()
        self.save_config()

    def filename(self):
        """ Human readable filename """
        return '{}-{}-{}e-{}bs-{}lr.h5'.format(
            self.train_start,
            self.name,
            self.epochs,
            self.batch_size,
            self.lr,
        )

    def set_callbacks(self, checkpoints=True, tensorboard=True):
        """ Set any model callbacks here """

        if checkpoints:
            if not os.path.exists('checkpoints'):
                os.mkdir('checkpoints')

            checkpoint = ModelCheckpoint(
                filepath='checkpoints/' + self.filename(),
                monitor='val_accuracy',
                verbose=1,
                save_best_only=True,
                mode='max'
            )
            self.callbacks.append(checkpoint)

        if tensorboard:
            log_dir = os.path.join(self.model_log_dir, self.filename()[:-3])
            self.file_writer = create_file_writer(log_dir + '/metrics')
            self.file_writer.set_as_default()
            tensorboard_callback = TensorBoard(
                log_dir=log_dir,
                write_graph=True,
                write_images=True,
                histogram_freq=0,
                profile_batch=0,
            )
            self.callbacks.append(tensorboard_callback)

        lr_schedule = None
        config = self.lr_schedule_config
        if config:
            if config.get('lr_schedule') == 'polynomial':
                lr_schedule = PolynomialDecay(maxEpochs=self.epochs, initAlpha=self.lr, power=config.get('lr_power'))
            elif config.get('lr_schedule') == 'linear':
                lr_schedule = PolynomialDecay(maxEpochs=self.epochs, initAlpha=self.lr, power=1)


        if lr_schedule:
            lr_callback = LearningRateScheduler(lr_schedule)
            self.callbacks.append(lr_callback)


    def data_generator(self, validation=False):
        ''' Create a data generator '''

        if hasattr(self, 'preprocess_func'):
            if self.preprocess_func == None:
                rescale = 1./255
                preprocess = None
            else:
                rescale = None
                preprocess = self.preprocess_func
        else:
            rescale = 1./255
            preprocess = None

        if self.verbose:
            print('Rescale set to: ', rescale)

        if validation:
            return ImageDataGenerator(
                    rescale=rescale,
                    validation_split=self.validation_split,
                    preprocessing_function=preprocess)

        print('Applying augmentations: ', self.augmentations)

        return ImageDataGenerator(
            **self.augmentations,
            #featurewise_center=True,
            #-featurewise_std_normalization=True,
            #rotation_range=90,
            #width_shift_range=0.05,
            #height_shift_range=0.05,
            #zoom_range=1.5,
            #shear_range=0.2,
            rescale=rescale,
            #-horizontal_flip=True,
            #-vertical_flip=True,
            validation_split=self.validation_split,
            preprocessing_function=preprocess,
            #-fill_mode='reflect',
        )

    def generate_preview(self):
        """
        Generate preview of augmented data
        """

        # Load data generator
        train_datagen = self.data_generator()
        # Train data generator on data to generate statistics
        train_datagen.fit(self.train_data[0])

        # Load data
        if self.data_dir:
            train_generator = train_datagen.flow_from_directory(
                self.data_dir,
                target_size=self.input_shape,
                batch_size=self.batch_size,
                class_mode='categorical',
                shuffle=False,
            )
        else:
            train_generator = train_datagen.flow(
                x=self.train_data[0],
                y=self.train_data[1],
                batch_size=self.batch_size,
                shuffle=False,
            )

        # Get batch
        img = next(train_generator)[0]

        # Clear preview directory
        if not os.path.exists('preview'):
            os.mkdir('preview')
        else:
            for f in os.listdir('preview'):
                os.remove(os.path.join('preview', f))

        # Extract batch of generated images, saved to ../preview/
        i = 0
        for _ in train_datagen.flow(img, batch_size=1, save_to_dir='preview', save_format='jpeg', shuffle=False):
            i += 1
            if i > self.batch_size:
                break

    def load_data(self):
        """ Load data """

        train_ratio = 1 - self.validation_split
        validation_ratio = self.validation_split / 2
        test_ratio = self.validation_split / 2
        preprocess = self.preprocess_func

        # Load full dataset and split ourselves into train/val/test, or use provided directories
        # to load from
        if self.data_dir or (self.val_dir and self.train_dir):

            # Train data generator if we have any dataset-wide augmentations that need stats
            if not any(['featurewise' in key for key in self.augmentations.keys()]):
                return

            # Load all data
            x_data, y_data, _ = load_data(self.data_dir, self.classes, rescale=False, preprocess=self.preprocess_func, verbose=self.verbose)

            # Split into just train/test, train/val is taken care of by tensorflow
            x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1) # 10% for testing
            self.train_data = (x_train, y_train)
            self.val_data = (list(), list())
            self.test_data = (x_test, y_test)

            """
            # https://datascience.stackexchange.com/a/53161
            x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=1 - train_ratio)
            x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio / (test_ratio + validation_ratio), shuffle=False)

            self.train_data = (x_train, y_train)
            self.val_data = (x_val, y_val)
            self.test_data = ([], [])
            """
        else:
            # Training data
            x_train, y_train, _ = load_data(self.train_dir, self.classes, rescale=False, preprocess=self.preprocess_func, verbose=self.verbose)

            # Validation data
            if not self.val_dir:
                x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=self.validation_split)
            else:
                x_val, y_val, _ = load_data(self.val_dir, self.classes, rescale=False, preprocess=None, verbose=self.verbose)

            self.train_data = (x_train, y_train)
            self.val_data = (x_val, y_val)

        print('train:', len(self.train_data[1]))
        print('val:', len(self.val_data[1]))

    def compile(self):
        ''' Sensible compile defaults '''
        optimizer = Adam(lr=self.lr)
        self.model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)

    def train(self, workers=None):
        """ Train the model """

        if not self.model:
            print('Model not instantiated')
            return None

        # Load data generator
        self.train_datagen = self.data_generator()
        self.validation_datagen = self.data_generator(validation=True)

        # Train data generator if we have any dataset-wide augmentations that need stats
        if any(['featurewise' in key for key in self.augmentations.keys()]):
            print('Training datagen')
            self.train_datagen.fit(self.train_data[0])

        # Generate a seed
        seed = random.randint(0, 2**32-1)

        if self.data_dir:
            # Load data
            self.train_generator = self.train_datagen.flow_from_directory(
                self.data_dir,
                target_size=self.input_shape,
                batch_size=self.batch_size,
                class_mode='categorical',
                classes=self.classes,
                subset='training',
                seed=seed
            )

            self.validation_generator = self.validation_datagen.flow_from_directory(
                self.data_dir,
                target_size=self.input_shape,
                batch_size=self.batch_size,
                class_mode='categorical',
                classes=self.classes,
                subset='validation',
                seed=seed,
            )

            # Calc class weights
            counter = Counter(self.train_generator.classes)
            max_val = float(max(counter.values()))
            class_weights = {class_id: max_val/num_images for class_id, num_images in counter.items()}
        else:

            if self.val_dir:
                self.train_generator = self.train_datagen.flow_from_directory(
                    self.train_dir,
                    target_size=self.input_shape,
                    batch_size=self.batch_size,
                    class_mode='categorical',
                    classes=self.classes,
                )
                self.validation_generator = self.validation_datagen.flow_from_directory(
                    self.val_dir,
                    target_size=self.input_shape,
                    batch_size=self.batch_size,
                    class_mode='categorical',
                    classes=self.classes,
                )

                # Calc class weights
                counter = Counter(self.train_generator.classes)
                max_val = float(max(counter.values()))
                class_weights = {class_id: max_val/num_images for class_id, num_images in counter.items()}
            else:
                self.train_generator = self.train_datagen.flow(
                    x=self.train_data[0],
                    y=self.train_data[1],
                    batch_size=self.batch_size,
                )
                self.validation_generator = self.validation_datagen.flow(
                    x=self.val_data[0],
                    y=self.val_data[1],
                    batch_size=self.batch_size,
                )

                # This might not work properly..
                self.train_generator.samples = len(self.train_data[0])
                self.validation_generator.samples = len(self.val_data[0])
                y_ints = [y.argmax() for y in self.train_data[1]]
                class_weights = class_weight.compute_class_weight(
                    class_weight='balanced',
                    classes=np.unique(y_ints),
                    y=y_ints
                )
                class_weights = dict(enumerate(class_weights))

        use_multiprocessing = True if workers != None else False
        print('Class weights:', class_weights)

        # Train
        history = self.model.fit(
            self.train_generator,
            epochs=self.epochs,
            verbose=self.verbose,
            callbacks=self.callbacks,
            validation_data=self.validation_generator,
            class_weight=class_weights,
            steps_per_epoch=(self.train_generator.samples // self.batch_size),
            validation_steps=self.validation_generator.samples // self.batch_size,
            use_multiprocessing=use_multiprocessing,
            workers=workers,
        )

        # Flush the tensorboard events
        self.file_writer.flush()
        self.file_writer.close()

        return history

    def predict(self, x_test):
        """ Generate prediction for single image """
        print('Predicting not implemented for', self.name)
        return self.model.predict(x_test, verbose=self.verbose)

    def evaluate(self):
        """ Evaluate the model on the validation data """
        return self.model.evaluate(self.validation_generator, verbose=self.verbose)

    def print_summary(self):
        """ Print the model summary """
        if self.model:
            self.model.summary()

    def save_location(self):
        return os.path.join(self.model_save_dir, self.filename())

    def save_config(self):
        # Save training config
        with open(self.save_location()[:-3] + '.json', 'w') as config_file:
            config_file.write(json.dumps(self._config))

    def save(self, save_dir=None):
        """ Save Keras model to disk """

        if self.model:
            if self.verbose:
                print('Saving to', self.save_location())

            # Save model and weights
            self.model.save(self.save_location())
