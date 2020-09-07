# Eye2Gene

You can still run this using docker, but it's not too hard to just run as a normal python script

# Training
## nvidia libs 

(if you have used system packages to install cuda then you probably don't need this, try without it first)

Make sure you have included the cuda toolkit in your `LD_LIBRARY_PATH`.
Mainly you need `cuda/lib64`, `cuda-10.1`, `cuda-10.1/extras`, `cuda-10.1/extras/CUPTI/lib64` (tensorboard)

You also need to add `cuda-10.1/bin` to your path if you want things like `nvidia-smi`

e.g here is my setup:
```
export WORKDIR=/mnt/new_root/rilott
export PATH=$WORKDIR/cuda-10.1/bin:~/Python-3.8.3/:~/.local/bin:$PATH
export LD_LIBRARY_PATH=$WORKDIR/cuda-10.1/lib64:$WORKDIR/cuda/lib64:$WORKDIR/cuda-10.1/extras/CUPTI/lib64:$LD_LIBRARY_PATH
```

## Python packages

I was using Python 3.8, which gives me access to tensorflow 2.2.0 at time of writing

Use a virtual environment, or be sure to use `--user` when running `pip` as follows:
```
pip3 install --user -r requirements.txt
```

## Running

You can train a network by running `train.py` in the `bin/` directory
e.g `python bin/train.py --model inceptionv3 --epochs 50 --lr 1e-4 --batch-size 8 --lr-schedule poly --lr-power 2 --split 0.3 --data-dir ../data/alldata --model-save-dir trainedmodels/ --model-log-dir logs/`

```
usage: train.py [-h] [--augmentations AUGMENTATIONS] [--batch-size BATCH_SIZE] [--classes CLASSES [CLASSES ...]] [--epochs EPOCHS] [--lr LEARNING_RATE] [--lr-schedule {linear,poly}] [--lr-power LR_POWER]
                [--model {vgg16,inception_resnetv2,inceptionv3,custom,nasnetlarge}] [--model-save-dir MODEL_SAVE_DIR] [--model-log-dir MODEL_LOG_DIR] [--no-weights] [--preview] [--split SPLIT] [--data-dir DATA_DIR]
                [--train-dir TRAIN_DIR] [--val-dir VAL_DIR] [--test-dir TEST_DIR] [--verbose]

optional arguments:
  -h, --help            show this help message and exit
  --augmentations AUGMENTATIONS
                        Comma separated values containing augmentations e.g horitzontal_flip=True,zoom=0.3
  --batch-size BATCH_SIZE
                        Batch size
  --classes CLASSES [CLASSES ...]
                        List of classes
  --epochs EPOCHS       Number of epochs to train
  --lr LEARNING_RATE    Learning rate
  --lr-schedule {linear,poly}
                        Learning rate scheduler
  --lr-power LR_POWER   Power of lr decay, only used when using polynomial learning rate scheduler
  --model {vgg16,inception_resnetv2,inceptionv3,custom,nasnetlarge}
                        Name of model to train
  --model-save-dir MODEL_SAVE_DIR
                        Save location for trained models
  --model-log-dir MODEL_LOG_DIR
                        Save location for model logs (used by tensorboard)
  --no-weights          Don't download and use any pretrained model weights, random init
  --preview             Preview a batch of augmented data and exit
  --split SPLIT         Training/Test split (% of data to keep for training, will be halved for validation and testing)
  --data-dir DATA_DIR   Full dataset directory (will be split into train/val/test)
  --train-dir TRAIN_DIR
                        Training data (validation is taken from this)
  --val-dir VAL_DIR     Validation data (can be supplied if you do not want it taken from training data
  --test-dir TEST_DIR   Testing data
  --verbose             Verbose
```

# Prediction

There is another script located at `bin/predict.py` which can be given a directory of images (in a structure keras can read), and a trained model. The script will then output percentages of correct predictions. Don't worry about `--batch-size` below, it seems this just affects how many images are predicted at once.

You'll need to specify the type of model using `--preprocess`, so that the correct preprocessing steps are applied to the input images. This currently needs to be done manually, but one day could be automatic.

```python
usage: predict.py [-h] [--batch-size BATCH_SIZE] [--classes CLASSES [CLASSES ...]] [--size SIZE] [--preprocess {inceptionv3,inception_resnetv2}] [--new] image_dir model

positional arguments:
  image_dir
  model

optional arguments:
  -h, --help            show this help message and exit
  --batch-size BATCH_SIZE
                        Batch size
  --classes CLASSES [CLASSES ...]
                        List of classes
  --size SIZE           Shape of input e.g 256 for (256,256)
  --preprocess {inceptionv3,inception_resnetv2}
                        Preprocessing to perform on images
  --new                 Set if predicting on a flat folder of new data
```

# Docker
## Start Docker
Obtain image or ask for image from me:
```
docker pull pontikos/keras_pytorch_cv2:tanga
```
You can also download it from here:
```
https://drive.google.com/open?id=1HrTYtBVoFtq6G-aSdg-K9pY7g8hEOgXF
```
Run image:
```
bash docker_run.sh
```

## Run docker on neuromancer
```
nvidia-docker run  -v `pwd -P`/images:/eye2gene/data/ -v `pwd -P`:/eye2gene -w /eye2gene/ -it  nvcr.io/nvidia/tensorflow:18.03-py2   bash
```

## Train
```
 python model.py --train data/train/AF --valid data/test/AF --epochs 100
```

## Predict
```
 python model.py --predict data/test/AF --weights weight.hdf5 > errors.csv
```

## Saliency


## Occlusion

