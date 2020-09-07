import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from data import load_data

parser = argparse.ArgumentParser()
parser.add_argument('--train')
parser.add_argument('--test')
parser.add_argument('--batch', type=int)
args = parser.parse_args()

# Set tf to grow into GPU memory, not pre-allocate
gpus = tf.config.experimental.list_physical_devices('GPU')
print('GPUs: ', gpus)

# Generator
datagen = ImageDataGenerator(
    rescale=1./255,
    featurewise_center=True,
)

# Load data
train = datagen.flow_from_directory(args.train, class_mode='categorical', batch_size=args.batch)
test = datagen.flow_from_directory(args.test, class_mode='categorical', batch_size=args.batch)

rows = 10
columns = args.batch
fig, ax = plt.subplots(rows, columns, figsize=(10,10))
for i in range(rows):
    x_batch, y_batch = train.next()
    print(x_batch.shape)
    for j in range(x_batch.shape[0]):
        ax[i][j].imshow(x_batch[j,:,:,:])
        ax[i][j].get_xaxis().set_visible(False)
        ax[i][j].get_yaxis().set_visible(False)
#plt.show()


model = tf.keras.applications.InceptionResNetV2(
    include_top=False,
    weights='imagenet',
    input_shape=(256,256,3),
    pooling='max',
)
outputs = tf.keras.layers.Dense(2, activation='softmax')(model.output)
model = tf.keras.Model(model.inputs, outputs)
optimiser = tf.keras.optimizers.Adam(learning_rate=1e-5)
model.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['accuracy'])

EPOCHS = 10
H = model.fit(x=train, epochs=EPOCHS)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(x=test, batch_size=args.batch)
print(classification_report(testY.argmax(axis=1),
            predictions.argmax(axis=1), target_names=le.classes_))
# plot the training loss and accuracy
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
print(history)
