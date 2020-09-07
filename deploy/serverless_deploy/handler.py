import os
import sys

current_location = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_location, 'vendored'))

import base64
import boto3
from tflite_runtime.interpreter import Interpreter
from io import BytesIO
from PIL import Image
import json
import numpy as np

bucket_name = 'eye2geneclassifier'
labels = ['ABCA4', 'USH2A']
model_name_file = 'model.txt'
model_name_file_path = '/tmp/' + model_name_file

class ErrorException(Exception):
    ''' Error exception '''

    def __init__(self, message, event=None):
        super().__init__(message)
        self.event = event


def get_image(event):
    ''' Extract image '''

    image_data = event.get('image_data')

    if 'body' in event:
        try:
            image_data = json.loads(event.get('body')).get('image_data')
        except Exception as e:
            raise ErrorException('Unable to parse JSON from body: {}'.format(e))

    if not image_data:
        raise ErrorException('Unable to load image_data', event=event)

    # Try to eliminate the need for PIL
    try:
        image = Image.open(BytesIO(base64.b64decode(image_data)))
        image = np.array(image, dtype='float32')
    except Exception as e:
        raise ErrorException('Failed to decode base64 image: {}'.format(str(e)))

    # Replicate to 3 channels if 1
    print(image.shape)
    if image.ndim == 2:
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)

    # Check shape
    if image.shape != (256, 256, 3):
        raise ErrorException('Image size was not (256,256,3), instead {}'.format(image.shape))

    # Perform rescaling
    image = (image - 127.5) / 127.5

    # Add 4th dimension
    image = image[np.newaxis, :, :, :]

    return image

def download_model():
    ''' Download model from S3 bucket '''

def error(message, event=None):
    ''' Return a nicely formatted error '''
    return {
        'status': 'failed',
        'label': None,
        'message': message,
        'event': event
    }

def handler(event, context):
    if not event:
        return error('Event was None')

    # Load model.txt
    s3_client = boto3.client('s3')
    s3_client.download_file(bucket_name, model_name_file, model_name_file_path)
    model_name = open(model_name_file_path, 'r').read().strip()
    model_path = '/tmp/' + model_name

    if not os.path.exists(model_path):
        s3_client.download_file(bucket_name, model_name, model_path)
        download_model()
        print('Downloaded model')

    # Create interpreter
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print('Loaded model')

    # Extract image
    try:
        image = get_image(event)
        print('Image loaded')
    except Exception as e:
        return error('Failed to load image: ' + str(e), event)

    # Input data and predict
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    prediction_data = interpreter.get_tensor(output_details[0]['index'])
    prediction_idx = np.argmax(prediction_data[0])
    predicted_label = labels[prediction_idx]
    print('Prediction complete')

    return {
        'statusCode': 200,
        'body': json.dumps({
            'label': predicted_label,
            'message': 'Classified as ' + predicted_label,
            'data': prediction_data[0].tolist()
        }),
        'headers': {'Content-Type': 'application/json'}
    }
