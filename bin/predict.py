'''
Script to infer labels on data from a pre-saved keras model, using folder-structured testing data
'''

import argparse
import os
import csv

import PIL
from PIL import Image
import numpy as np
import cv2

import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def predict(model, classes, directory, batch_size, preprocess, size):
    """
    Predict using image generators
    """

    if not preprocess:
        rescale_val = 1./255
    else:
        rescale_val = None

    datagen = ImageDataGenerator(
        rescale=rescale_val,
        preprocessing_function=preprocess
    )

    prediction_generator = datagen.flow_from_directory(
        directory,
        classes=classes,
        target_size=(size, size),
        batch_size=batch_size,
        shuffle=False
    )

    preds = model.predict(
        prediction_generator,
        verbose=1,
    )

    preds_cls_idx = preds.argmax(axis=-1)
    prednames = [labels[k] for k in preds_cls_idx]
    filenames_to_cls = list(zip(prediction_generator.filenames, prednames))

    return filenames_to_cls

def predict_new(model, classes, directory, batch_size, preprocess, size):
    if not preprocess:
        rescale_val = 1./255
    else:
        rescale_val = None

    datagen = ImageDataGenerator(
        rescale=rescale_val,
        preprocessing_function=preprocess
    )

    prediction_generator = datagen.flow_from_directory(
        directory,
        target_size=(size, size),
        class_mode=None,
        batch_size=batch_size,
        shuffle=False,
    )

    preds = model.predict(
        prediction_generator,
        use_multiprocessing=True,
        workers=30,
        verbose=1,
    )

    preds_cls_idx = preds.argmax(axis=-1)
    prednames = [classes[k] for k in preds_cls_idx]
    filenames_to_cls = list(zip(prediction_generator.filenames, prednames))

    return filenames_to_cls




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('image_dir')
    parser.add_argument('model')
    parser.add_argument('--batch-size', default=32, help='Batch size', type=int)
    parser.add_argument('--classes', default=['ABCA4', 'USH2A'], help='List of classes', nargs='+')
    parser.add_argument('--size', type=int, default=256, help='Shape of input e.g 256 for (256,256)')
    parser.add_argument('--preprocess', choices=['inceptionv3', 'inception_resnetv2'], help='Preprocessing to perform on images')
    parser.add_argument('--new', action='store_true', help='Set if predicting on a flat folder of new data')
    parser.add_argument('--manual', action='store_true', help='Manually iterate over images and run prediction on each one. If not selected Keras flow_from_directory is used to perform predictions')
    parser.add_argument('--output', help='File to output CSV results to')
    args = parser.parse_args()

    path = args.image_dir
    print('Loading model')
    loaded_model = load_model(args.model)
    print('Model loaded')

    if args.preprocess == 'inceptionv3':
        preprocess = tensorflow.keras.applications.inception_v3.preprocess_input
    elif args.preprocess == 'inception_resnetv2':
        preprocess = tensorflow.keras.applications.inception_resnet_v2.preprocess_input
    else:
        preprocess = None

    labels = args.classes

    if args.new:
        print('New data mode selected')
        predictions = predict_new(loaded_model, args.classes, args.image_dir, args.batch_size, preprocess=preprocess, size=args.size)
        predictions = np.array(predictions)
        print(np.unique(predictions[:,1], return_counts=True))
        exit(0)
    else:
        predictions = predict(loaded_model, args.classes, args.image_dir, args.batch_size, preprocess=preprocess, size=args.size)

    if not args.manual:
        correct = 0
        images = 0
        correct_map = {label : 0 for label in labels}
        total_map = {label: 0 for label in labels}
        for p in predictions:
            gene = p[0].split('/')[0]
            pred = p[1]
            if gene == pred:
                correct_map[gene] += 1
                correct += 1
            total_map[gene] += 1
            images += 1

        for label in labels:
            print('{}: {} / {}'.format(label, correct_map[label], total_map[label]))
        print('Percentage correct (generator): {:.2f}, {}/{}'.format(correct / len(predictions) * 100, correct, images))
    else:
        count = 0
        images = 0
        results = list()
        for d in os.listdir(path):

            if not os.path.isdir(os.path.join(path, d)):
                continue

            for f in os.listdir(os.path.join(path, d)):

                # Load image
                try:
                    image = Image.open(os.path.join(os.path.join(path, d), f))
                    images += 1
                except PIL.UnidentifiedImageError:
                    print('Unable to load image file', os.path.join(os.path.join(path, d), f))
                    results.append((f, 0, 0, 'ERROR'))
                    print(f, 'ERROR')
                    continue

                # Convert to grayscale
                if image.mode == 'RGB':
                    image = image.convert('L')

                # Convert to numpy array
                image = np.array(image, dtype='float32')

                # Squeeze extra dimensions
                if len(image.shape) == 3:
                    image = np.squeeze(image)

                # Resize
                if image.shape != (args.size, args.size):
                    image = cv2.resize(image, dsize=(args.size, args.size), interpolation=cv2.INTER_CUBIC)

                # Make grayscale 3 channel input (might be able to bin this)
                image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
                image = image[np.newaxis, :, :, :]

                # Do any image preprocessing
                if preprocess:
                    image = preprocess(image)
                else:
                    image /= 255

                # Get network prediction
                prediction = loaded_model.predict(image)
                prediction_class = labels[prediction.argmax(axis=-1)[0]]
                print(f, prediction[0], prediction_class)
                results.append((f, prediction[0][0], prediction[0][1], prediction_class))

                # Save result
                if d == prediction_class:
                    count += 1

        print('Percentage correct (manual): {:.2f}, {}/{}'.format((count / images * 100), count, images))


    if args.output:
        with open(args.output, 'w') as csvout:
            writer = csv.writer(csvout)
            headers = ['file'] + args.classes + ['label']
            writer.writerow(headers)
            writer.writerows(results)