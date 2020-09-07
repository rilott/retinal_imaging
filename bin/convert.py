import tensorflow as tf
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('modelfile')
    parser.add_argument('output')
    args = parser.parse_args()


    model = tf.keras.models.load_model(args.modelfile)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tfmodel = converter.convert()
    open(args.output, "wb").write(tfmodel)
