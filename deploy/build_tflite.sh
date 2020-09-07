docker build -t tflite_amazonlinux .
docker run -d --name=tflite_amazonlinux tflite_amazonlinux
docker cp tflite_amazonlinux:/usr/local/lib64/python3.7/site-packages .
docker cp tflite_amazonlinux:/usr/local/lib/python3.7/site-packages site-packages2/
mv -r site-packages2/* site-packages/
rmdir site-packages2/
docker stop tflite_amazonlinux
