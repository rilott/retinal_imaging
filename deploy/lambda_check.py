''' Test lambda function '''
import base64
import argparse
import os
import json
import time
import requests
#from PIL import Image
#from io import BytesIO

# Args
parser = argparse.ArgumentParser()
parser.add_argument('image')
args = parser.parse_args()

# Lambda function URL
FUNCTION_URL = 'https://e0jpk3czz0.execute-api.eu-central-1.amazonaws.com/default/rosstest'
FUNCTION_URL = 'https://dfce7ghhwh.execute-api.eu-central-1.amazonaws.com/dev/ping' #serverless framework

if os.path.exists(args.image):

    # Read in image
    with open(args.image, 'rb') as image_file:
        imstr = base64.b64encode(image_file.read())
    #imstr = base64.b64encode(imbuf.getvalue()).decode('ascii')

    # Create body
    data = {'image_data': imstr.decode('ascii')}

    # Send to lambda function
    start = time.time()
    resp = requests.post(FUNCTION_URL, json=data)
    end = time.time() - start
    print('Time taken: {:.2f} seconds'.format(end))

    # Print response
    print(json.dumps(resp.json(), indent=2))
