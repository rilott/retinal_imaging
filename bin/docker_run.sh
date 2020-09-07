#nvidia-docker run -v /media/raid:/eye2gene/data -v `pwd -P`:/eye2gene -it pontikos/keras_pytorch_cv2:tanga bash

#docker run -v /media/raid:/eye2gene/data -v `pwd -P`:/eye2gene -it eye2gene:latest bash

#nvidia-docker run  -v /media/raid:/eye2gene/data -v `pwd -P`:/eye2gene -w /eye2gene/ -it pontikos/keras2_cv2_mitosis:tanga  bash

#nvidia-docker run  -v data:/eye2gene/data -v `pwd -P`:/eye2gene -w /eye2gene/ -it pontikos/keras_pytorch_cv2:tanga  bash

nvidia-docker run  -v `pwd -P`/images:/eye2gene/data/ -v `pwd -P`:/eye2gene -w /eye2gene/ -it  nvcr.io/nvidia/tensorflow:18.03-py2 bash

