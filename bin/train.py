"""
Train various different architectures of neural network using Keras
"""

import sys
import argparse
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from models.vgg16 import VGG16
from models.inception_resnetv2 import InceptionResnetV2
from models.inceptionv3 import InceptionV3
from models.custom import Custom
from models.nasnetlarge import NASNetLarge

model_choices = [
    'vgg16',
    'inception_resnetv2',
    'inceptionv3',
    'custom',
    'nasnetlarge',
]

def parse_augs(augs):
    if not augs:
        return dict()

    pairs = augs.split(',')
    parsed_augs = dict()
    for setting in pairs:
        var, val = setting.split('=')

        # Integer/Float parsing
        try:
            val = int(val)
        except ValueError:
            try:
                val = float(val)
            except ValueError:
                pass

        # Bool parsing
        if str(val).lower() == 'true':
            val = True
        elif str(val).lower() == 'false':
            val = False

        parsed_augs[var] = val
    return parsed_augs


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--augmentations', help='Comma separated values containing augmentations e.g horitzontal_flip=True,zoom=0.3')
    parser.add_argument('--batch-size', help='Batch size', default=32, type=int)
    parser.add_argument('--classes', default=['ABCA4', 'USH2A'], help='List of classes', nargs='+')
    parser.add_argument('--dropout', default=0.7, type=float, help='Dropout change 0-1')
    parser.add_argument('--epochs', type=int, help='Number of epochs to train', default=10)
    parser.add_argument('--lr', dest='learning_rate', default=1e-04, help='Learning rate', type=float)
    parser.add_argument('--lr-schedule', choices=['linear', 'poly'], help='Learning rate scheduler')
    parser.add_argument('--lr-power', type=int, help='Power of lr decay, only used when using polynomial learning rate scheduler', default=1)
    parser.add_argument('--model', help='Name of model to train', choices=model_choices)
    parser.add_argument('--model-save-dir', default='trained_models/', help='Save location for trained models')
    parser.add_argument('--model-log-dir', default='logs/', help='Save location for model logs (used by tensorboard)')
    parser.add_argument('--no-weights', action='store_true', help="Don't download and use any pretrained model weights, random init")
    parser.add_argument('--preview', action='store_true', help='Preview a batch of augmented data and exit')
    parser.add_argument('--split', help='Training/Test split (%% of data to keep for training, will be halved for validation and testing)', type=float, default=0.2)
    parser.add_argument('--data-dir', help='Full dataset directory (will be split into train/val/test)')
    parser.add_argument('--train-dir', help='Training data (validation is taken from this)')
    parser.add_argument('--val-dir', help='Validation data (can be supplied if you do not want it taken from training data')
    parser.add_argument('--workers', default=8, type=int, help='Number of workers to use when training (multiprocessing)')
    parser.add_argument('--test-dir', help='Testing data')
    parser.add_argument('--verbose', action='store_true', help='Verbose')

    args = parser.parse_args()

    if not args.data_dir:
        if not args.train_dir:
            print('Need to supply --train-dir')
            sys.exit(1)

    # Set tf to grow into GPU memory, not pre-allocate
    gpus = tf.config.experimental.list_physical_devices('GPU')

    if len(gpus) == 0:
        print('No GPUs found!')
        sys.exit(1)

    if args.verbose:
        print('GPUs: ', gpus)

    try:
        figure, plots = plt.subplots(2, 1, figsize=(5, 10))
    except Exception as e:
        if args.verbose:
            print('Unable to create figures, disabling graphical output')
        matplotlib.use('Agg')
        pass

    # Parse augmentations
    try:
        augmentations = parse_augs(args.augmentations)
    except Exception as e:
        print('Error parsing augmentations, make sure it is in csv format, with each value being setting=value')
        print(e)
        exit(1)

    # Set number of channels needed for our input data
    channels = 1
    if args.model == 'vgg16' or args.model == 'inception_resnetv2':
        channels = 3

    model_config = {
        'augmentations': augmentations,
        'classes': args.classes,
        'dropout': args.dropout,
        'model_name': args.model,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.learning_rate,
        'verbose': args.verbose,
        'data_dir': args.data_dir,
        'train_dir': args.train_dir,
        'val_dir': args.val_dir,
        'test_dir': args.test_dir,
        'validation_split': args.split,
        'input_shape': (256, 256),
        'use_imagenet_weights': (not args.no_weights),
        'model_log_dir': args.model_log_dir,
        'model_save_dir': args.model_save_dir,
    }

    if args.lr_schedule == 'poly':
        lr_schedule_config = {
            'lr_schedule': 'polynomial',
            'initial_lr': model_config.get('lr'),
            'lr_power': args.lr_power
        }
    elif args.lr_schedule == 'linear':
        lr_schedule_config = {
            'lr_schedule': 'linear',
            'inital_lr': model_config.get('lr'),
            'lr_power': 1,
        }
    else:
        lr_schedule_config = None

    model_config['lr_schedule_config'] = lr_schedule_config

    if args.verbose:
        print(model_config)



    # Create model
    if args.model == 'vgg16':
        model = VGG16(model_config)
    elif args.model == 'inception_resnetv2':
        model = InceptionResnetV2(model_config)
    elif args.model == 'inceptionv3':
        model = InceptionV3(model_config)
    elif args.model == 'custom':
        model = Custom(model_config)
    elif args.model == 'nasnetlarge':
        model = NASNetLarge(model_config)
    else:
        print('Unknown/No model selected!')
        sys.exit(1)

    model.compile()

    if args.verbose:
        model.print_summary()

    if args.preview:
        print('## Generating a batch of augmented images')
        model.generate_preview()
        sys.exit(0)

    if model:
        print('## Training on train data ##')
        history = model.train(workers=args.workers)
        print('## Training complete ##')

    #print('## Check our training was ok by evaluating again on the validation data')
    #score = model.predict_generator(model.val_generator)
    #print(score)
    #print('Val loss:', score[0])
    #print('Val accuracy:', score[1])

    print('## Evaluating on test data##')
    score = model.evaluate()
    print('Validation loss:', score[0])
    print('Validation accuracy:', score[1])
    model.accuracy = np.round(score[1]*100)

    model.save()
    print('## Model saved ##')

    #print('## Predicting ##')
    #model.test_generator.reset()
    #predictions = model.predict(args.test_data)

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
    ax2.legend(['train', 'val'], loc='upper left')
    print('Output graph saved to ', os.path.join(args.model_save_dir, model.filename()) + '.png')
    figure.savefig(os.path.join(args.model_save_dir, model.filename()) + '.png')
