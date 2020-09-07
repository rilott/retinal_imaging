docker start tflite_amazonlinux
docker exec -it tflite_amazonlinux pip3.7 install boto3 pillow
docker cp tflite_amazonlinux:/usr/local/lib64/python3.7/site-packages .
docker cp tflite_amazonlinux:/usr/local/lib/python3.7/site-packages site-packages2
mv site-packages2/* site-packages/
rmdir site-packages2/
rm -rf vendored/
mv site-packages/ vendored/
docker stop tflite_amazonlinux
