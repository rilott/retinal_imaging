""" Data helper functions """
import os
import random
import numpy as np
import PIL

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical

def load_data(directory, classes, rescale=True, preprocess=None, verbose=False):
    """ Helper function to load data in a Keras friendly format """

    if not os.path.exists(directory):
        raise FileNotFoundError(directory + ' not found')

    # Get directories
    directories = os.listdir(directory)

    # Count files
    num_images = 0
    for d in directories:
        if d not in classes or not os.path.isdir(os.path.join(directory, d)):
            continue
        num_images += len([name for name in os.listdir(os.path.join(directory, d)) if os.path.isfile(os.path.join(os.path.join(directory, d), name))])

    # Create numpy array with the correct size (pending actually loading images)
    x = np.empty((num_images, 256, 256, 3), dtype='float32')
    y = list()

    filen = 0
    failed = 0
    for d in directories:
        # Skip any class directories we don't want
        if d not in classes or not os.path.isdir(os.path.join(directory, d)):
            if verbose:
                print('Skipping', d)
            continue

        if verbose:
            print('Loading directory', d)

        for f in os.listdir(os.path.join(directory, d)):
            try:
                # Load image
                img = load_img(
                    os.path.join(os.path.join(directory, d), f),
                    color_mode='rgb',
                    target_size=[256, 256]
                )
            except PIL.UnidentifiedImageError:
                failed += 1
                if verbose:
                    print('Failed to load {}'.format(os.path.join(os.path.join(directory, d), f)))
                continue

            # Convert to numpy array
            img = img_to_array(img)

            # Apply any preprocess function and rescaling
            if preprocess is not None:
                img = preprocess(img)
            else:
                if rescale:
                    img /= 255

            # Append label to y            
            y.append(classes.index(d))

            # Save img to x
            x[filen, ...] = img

            # Increment img number
            filen += 1
    
    # Remove empty rows of x due to failed image reads
    if failed > 0:
        x = x[:-failed,...]
    
    # Convert y to categorical
    y = to_categorical(y)    

    return x, y, 0
